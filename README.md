# 1BRC on CUDA with cuda-oxide

This is my attempt at the [One Billion Row Challenge](https://github.com/gunnarmorling/1brc)
on the GPU using [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — NVLabs' experimental
library for writing CUDA kernels directly in Rust. The goal is to read a 13 GB text file of
weather station temperature readings, compute min/mean/max per station across 1 billion rows,
and do it as fast as possible.

If you have recommendations or spot something I missed, please let me know — still learning
a lot about GPU programming through this :)

## Build and run

Before running go to the cuda-oxide repo and setup your environment.

```
cargo oxide run
```

Place the measurement file at `./1brc/measurements.txt` before running.

## Performance over time (warm page cache, RTX 4090)

| Stage | Kernel | H2D* | Total | vs baseline |
|-------|--------|------|-------|-------------|
| [Baseline — nsys profiling](#nsys-profiling--figuring-out-where-time-actually-goes) | 0.527s | 1.362s | 1.889s | — |
| [Attempt 1 — slot lookup cache](#attempt-1--slot-lookup-cache-made-it-worse) | 0.685s | 1.362s | 2.047s | **-30%** ↓ |
| [Attempt 2 — block-level aggregation](#attempt-2--block-level-shared-memory-aggregation-current) | 0.492s | 1.362s | 1.854s | +6.6% |
| [Attempt 2 — best observed](#attempt-2--block-level-shared-memory-aggregation-current) | 0.459s | 1.362s | 1.821s | +12.9% |
| [Attempt 3 — table sharding (4 tables)](#attempt-3--table-sharding-across-4-independent-hash-tables) | 0.426s | 1.362s | 1.788s | **+19.2%** |

\* H2D varies with OS page cache warmth (cold cache: 10–16s, warm: ~1.4s). Best observed used throughout for kernel comparison clarity.

**Spoiler-alert its the H2D that demolishes times**


## Design

### Input loading

The file is memory-mapped with `memmap2`. The OS pages it in lazily as the H2D copy reads
through it, so there is no separate "read into RAM" step — the CUDA driver DMAs directly from
OS-paged memory. Using `std::fs::read` instead roughly doubles transfer time because it forces
a full sequential read into a heap `Vec` before the GPU copy can even start. H2D time varies
a lot depending on whether the OS page cache is warm; a cold-cache run can take 10–16s while a
warm-cache run sits around 1.4s.

### Parallel line parsing

1 048 576 threads (4 096 blocks × 256 threads) divide the byte range evenly. Because lines are
variable length, each thread skips forward to the first `\n` after its nominal start position so
every thread begins on a complete line boundary. Thread 0 always starts at byte 0. Each thread
then processes lines sequentially until it reaches the end of its chunk.

Temperatures are stored as integer tenths (`-12.3 °C → -123`), avoiding any floating-point in
the kernel. The format is always `[-]D[D].D`, so the parser is a tight hand-rolled loop.

### GPU hash table

A flat open-addressing table of `TABLE_SIZE = 16 384` slots stores per-station aggregates.
Each slot is backed by parallel arrays:

| Array | Type | Purpose |
|-------|------|---------|
| `table_name_lens` | `i32` | sentinel + name length |
| `table_names` | `u8[NAME_MAX]` | station name bytes |
| `table_sums` | `i64` | sum of readings (tenths) |
| `table_cnts` | `i32` | count of readings |
| `table_mins` | `i32` | minimum reading (tenths) |
| `table_maxs` | `i32` | maximum reading (tenths) |

Slot ownership uses a three-state protocol on `table_name_lens`:

- `0` — empty
- `-1` — being written (lock held by claiming thread)
- `> 0` — valid, holds the name length

A thread claiming an empty slot does `CAS(0 → -1, AcqRel/Acquire)`. On success it copies the
name bytes then does a `Release` store of the real length. Readers do an `Acquire` load; once
they see `> 0` the name bytes are guaranteed visible. Threads that see `-1` spin in place.

Station names are hashed with FNV-1a for the initial slot, with linear probing on collision.

Aggregate updates (`sum`, `cnt`, `min`, `max`) use `DeviceAtomicI64`/`DeviceAtomicI32` with
`Relaxed` ordering — the slot is already claimed before any update is issued.

### cuda-oxide specifics

`DeviceAtomicI32` / `DeviceAtomicI64` are not `Copy`, so they cannot live in `DeviceBuffer<T>`.
The workaround: allocate `DeviceBuffer<i32>` on the host, pass `cu_deviceptr() as *mut i32` to
the kernel, and reinterpret inside the kernel with `DeviceAtomicI32::from_ptr`.

`#[kernel]` is required on every entry-point function; `thread::index_1d()` and related
functions are panicking stubs outside that attribute.

`Option<T>` with `T` exactly 128 bytes triggers a PTX backend layout bug (`field 0 (size 128)
at byte 8 exceeds total size 128`). `parse_line` avoids this by returning
`Option<(usize, i16, usize)>` and reading name bytes directly from the `raw_text` slice.

### Readback and output

All six table arrays are copied back to the host with `to_host_vec`. Non-empty slots are
collected, sorted alphabetically, and printed in the standard 1BRC format:

```
{Abha=-23.0/18.0/59.2, Abidjan=-16.2/26.0/67.3, ...}
```

---

## Optimization journey

### nsys profiling — figuring out where time actually goes

Before touching anything, I ran `nsys profile --trace=cuda,osrt --stats=true` to get a ground
truth on where time was being spent. This is what came back on a cold-cache run:

```
Time (%)  Total Time       Num Calls  Name
--------  ---------------  ---------  --------------------
    95.3   16006637005 ns          3  cuMemcpyHtoDAsync_v2
     4.1     688288065 ns          8  cuStreamSynchronize
     0.3      51934923 ns          7  cuMemFree_v2
     0.0       1230798 ns          1  cuModuleLoadData
     0.0       1027090 ns          6  cuMemcpyDtoHAsync_v2
     0.0        256210 ns          1  cuLaunchKernel
```

`cuLaunchKernel` is 256 µs — basically nothing. The kernel itself runs entirely inside
`cuStreamSynchronize`. On a warm-cache run, H2D drops to ~1.4s and the kernel lands around
0.527s. That became the baseline to beat.

### Understanding the actual bottleneck

The kernel issues exactly 4 atomic operations per row:

```
fetch_add (sum)   ×1 000 000 000
fetch_add (cnt)   ×1 000 000 000
fetch_min (min)   ×1 000 000 000
fetch_max (max)   ×1 000 000 000
                  ──────────────
                  4 000 000 000 atomic operations total
```

All 4 billion of those target a hash table with only ~10 000 occupied slots. With 1 048 576
threads in flight simultaneously, roughly **100 threads are contending for the same cache line
at any given moment**. Global atomics on the same cache line serialize at the L2 — they queue
up and execute one at a time. That's the wall.

The GDDR6X on the 4090 can push ~1 TB/s, so raw memory bandwidth isn't the issue. The 13 GB
file is streamed sequentially with high spatial locality and the hardware prefetcher handles it
fine. The problem is purely the atomic traffic hammering 10K hot spots.

### Attempt 1 — slot lookup cache (made it worse)

The first idea was to cache the result of `find_or_claim_slot` in shared memory so that threads
seeing the same station name repeatedly within a block could skip the CAS and global memory
probe. I dropped `NUM_BLOCKS` from 4096 to 512 to give each block a bigger chunk of the file
(more repeated names) and added a `SharedArray<i32, 4096>` mapping `hash → global_slot`.

```
Kernel: 0.685s  (was 0.527s)
```

Worse. The logic felt right but it targeted the wrong thing. `find_or_claim_slot` CAS
operations only fire **once per unique station in total** — at most ~10K times across the entire
run. That's not what's taking time. The 4 billion aggregate atomics (`fetch_add`, `fetch_min`,
`fetch_max`) are completely untouched by the cache. The cache lookup also read the global name
table to verify a hit, which is its own global memory access on every row. Net result: more
work, slower kernel.

### Attempt 2 — block-level shared memory aggregation (current)

The correct thing to reduce is the aggregate atomics, not the slot lookup. The idea: instead of
hitting global memory 4 times per row, accumulate into shared memory within each block and
flush to global once at the end.

Each block now runs three phases:

**Phase 1 — init.** All 256 threads cooperatively zero five `SharedArray`s of 2048 slots each:
`SH_SLOTS` (which global slot owns each local slot), `SH_SUMS`, `SH_CNTS`, `SH_MINS`,
`SH_MAXS`. That's 48 KB of shared memory per block, initialized to identity values before any
rows are processed.

**Phase 2 — accumulate.** For each row, we still call `find_or_claim_slot` to get the global
slot (unavoidable — that's where names live). Then we map it to a local slot with
`global_slot & 2047`. If that local slot is free, we claim it with a `CAS(-1 → global_slot)`
on `SH_SLOTS` and start accumulating into the shared arrays with shared-memory atomics. If it's
already claimed by our station, we just accumulate. If it's claimed by a different station
(collision), we fall back to global atomics for that row.

**Phase 3 — flush.** After a `sync_threads()`, each thread cooperatively iterates over its
portion of the 2048 local slots and issues 4 global atomics per occupied slot:
`fetch_add(sum)`, `fetch_add(cnt)`, `fetch_min(min)`, `fetch_max(max)`.

```
Kernel: 0.459s  (was 0.527s, ~13% faster)
```

Real improvement, but not dramatic. Two things limit it:

**Coverage is only ~20%.** With 4096 blocks and ~244K rows per block, every block's chunk is
large enough that all 10K unique stations show up in it (each station appears ~24 times per
block on average). So all 10K stations are competing for 2048 local slots simultaneously. Only
about 2033 of them win a slot — the other ~8000 always fall back to global atomics. That means
roughly 80% of rows are still hitting the same contended global atomics as before.

**Occupancy drops from ~100% to 33%.** At 48 KB of shared memory per block, the RTX 4090 can
only fit 2 blocks per SM at a time (96 KB shared memory per SM). That's 16 active warps
instead of the ~48 it can normally juggle. Fewer warps means less ability to hide stalls by
switching to another warp while one is waiting. This eats into the gains from the reduced
atomic count.

### Attempt 3 — table sharding across 4 independent hash tables

The shared memory approach reduced contention but couldn't cover all 10K stations. The next
idea was to attack the contention from the other direction: instead of fewer writes per slot,
have fewer threads competing for each slot.

The kernel now assigns each block to one of 4 independent tables based on its block ID
(`blockIdx % 4`). GPU memory holds 4× everything — 4 tables of 16K slots each. All 4096
blocks still run at the same time across the 128 SMs, but now only 1024 of them are writing
to any given table. That drops contention from ~100 threads per slot down to ~25.

```rust
let table_id = (thread::blockIdx_x() % NUM_TABLES as u32) as usize;
let tbl_sums  = unsafe { table_sums.add(table_id * TABLE_SIZE) };
// ... same for name_lens, names, cnts, mins, maxs
```

After the kernel finishes, all 4 tables are copied back to the host and merged on the CPU
in a tight loop — scan each table's 16K slots, accumulate into a `HashMap` with
`sum +=`, `cnt +=`, `min = min(...)`, `max = max(...)`. Four tables of ~10K entries each
is ~40K iterations, which takes under a millisecond and doesn't even show up in the timings.

```
Kernel: 0.426s  (was 0.459s, ~7% faster)
```

Also tried 8 tables. The results were essentially the same as 4 — 0.429–0.440s. At that
point the contention cost is no longer the dominant term; the kernel is spending most of its
remaining time on parsing, hashing, and `find_or_claim_slot` probes, which sharding doesn't
touch.

### Things that didn't pan out

**Larger global table (`TABLE_SIZE` 32K).** Doubling the table halves the expected probe
chain length, but at 61% load factor the chains are already very short. The bigger
`table_names` array (3.2 MB instead of 1.6 MB) pushed up L2 working set enough to cost
more than the shorter probes saved. Reverted.

**Narrower types for mins/maxs (`i16`) and name lengths (`i8`).** The values fit —
temperature tenths max out around ±999 which is well inside `i16`, and station name lengths
max at 100 which fits in `i8` with room for the -1 sentinel. But CUDA's PTX atomic
instructions only exist for 32-bit and 64-bit operands. There are no native `atomicMin` /
`atomicMax` for 16-bit integers, and no 8-bit atomics at all. cuda-oxide only exposes
`DeviceAtomicI32` and `DeviceAtomicI64`, so it's simply not an option at the moment. The
memory saved (a few hundred KB across all arrays) is pretty irrelevant next to the 13 GB input.

**Doubling the local table (`LOCAL_SIZE` 4096).** This would double station coverage from
~20% to ~37%, but the 5 shared arrays at 4096 slots need 96 KB per block — and static
shared memory has a hard 48 KB per-block limit in CUDA. Dynamic shared memory can go
higher (up to 99 KB on sm_89) but requires `cudaFuncSetAttribute` to opt in, which
cuda-oxide doesn't expose. The attempt compiled fine but failed at PTX JIT load time with
error 218.

---

## Brick wall

The current best kernel time is **0.426s** and I don't think there's much left to squeeze out
of this approach without either changing the algorithm or patching cuda-oxide.

Table sharding was the last meaningful lever. Going from 4 to 8 tables made no measurable
difference — 0.429s vs 0.426s, within noise. That tells you the contention component of the
kernel is now small enough that it's no longer the bottleneck. The remaining ~0.43s is
dominated by things sharding can't help: parsing bytes, running the FNV hash, and probing
`find_or_claim_slot`. Those touch every row unconditionally.

Here's where the approaches stacked up against each other and why each one ran out of road:

**Shared memory aggregation** — the right idea, wrong scale. To cover all 10K stations locally
you'd need 200 KB of shared memory per block. The hardware gives you 48 KB for static
declarations, full stop. Dynamic shared memory can go to 99 KB on Ada Lovelace but needs
`cudaFuncSetAttribute` to opt in, which cuda-oxide doesn't expose. Even if you got to 99 KB
you'd only cover ~4900 stations and crush occupancy to 1 block per SM. The 48 KB limit means
2048 slots covering ~20% of stations, with occupancy already halved to 33%.

**Table sharding** — worked, but hits a ceiling fast. 4 tables gave a real 6% drop in kernel
time. 8 tables gave nothing. Once you've diluted the contention enough that it's no longer the
slowest thing in the kernel, adding more tables just adds overhead without removing anything.

For now, **~0.43s kernel on a 13 GB file** (1.79s total) is where this sits. Happy with it as a learning
exercise in GPU atomics, shared memory limits, and contention patterns. If you've got ideas
for pushing it further I'd love to hear them :)
