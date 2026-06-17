use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicI32, DeviceAtomicI64};
use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;
use memmap2::Mmap;

const TABLE_SIZE: usize = 1 << 14;
const NUM_TABLES: usize = 4;
const NAME_MAX: usize = 100;
const BLOCK_SIZE: u32 = 256;
const NUM_BLOCKS: u32 = 4096;
const LOCAL_SIZE: usize = 1 << 11; // 2048 slots × 24 B = 48 KB, the static shared mem limit

#[cuda_module]
mod kernels {
    use super::*;

    // Each block owns its shared memory, so the static-mut aliasing is safe.
    #[allow(static_mut_refs)]
    #[kernel]
    pub fn process_measurements(
        raw_text: &[u8],
        table_name_lens: *mut i32,
        table_names: *mut u8,
        table_sums: *mut i64,
        table_cnts: *mut i32,
        table_mins: *mut i32,
        table_maxs: *mut i32,
    ) {
        // Per-block shared aggregation table; flushed to global at the end.
        static mut SH_SLOTS: SharedArray<i32, LOCAL_SIZE> = SharedArray::UNINIT;
        static mut SH_SUMS: SharedArray<i64, LOCAL_SIZE> = SharedArray::UNINIT;
        static mut SH_CNTS: SharedArray<i32, LOCAL_SIZE> = SharedArray::UNINIT;
        static mut SH_MINS: SharedArray<i32, LOCAL_SIZE> = SharedArray::UNINIT;
        static mut SH_MAXS: SharedArray<i32, LOCAL_SIZE> = SharedArray::UNINIT;

        let local_tid = thread::threadIdx_x() as usize;
        let block_size = thread::blockDim_x() as usize;

        // Cooperatively init the shared arrays.
        let mut i = local_tid;
        while i < LOCAL_SIZE {
            unsafe {
                SH_SLOTS[i] = -1;
                SH_SUMS[i] = 0;
                SH_CNTS[i] = 0;
                SH_MINS[i] = i32::MAX;
                SH_MAXS[i] = i32::MIN;
            }
            i += block_size;
        }
        thread::sync_threads();

        // Base pointers for arithmetic.
        let sh_slots = unsafe { SH_SLOTS.as_ptr() as *mut i32 };
        let sh_sums = unsafe { SH_SUMS.as_ptr() as *mut i64 };
        let sh_cnts = unsafe { SH_CNTS.as_ptr() as *mut i32 };
        let sh_mins = unsafe { SH_MINS.as_ptr() as *mut i32 };
        let sh_maxs = unsafe { SH_MAXS.as_ptr() as *mut i32 };

        // Shard across NUM_TABLES tables to cut atomic contention between blocks.
        let table_id = (thread::blockIdx_x() % NUM_TABLES as u32) as usize;
        let tbl_name_lens = unsafe { table_name_lens.add(table_id * TABLE_SIZE) };
        let tbl_names = unsafe { table_names.add(table_id * TABLE_SIZE * NAME_MAX) };
        let tbl_sums = unsafe { table_sums.add(table_id * TABLE_SIZE) };
        let tbl_cnts = unsafe { table_cnts.add(table_id * TABLE_SIZE) };
        let tbl_mins = unsafe { table_mins.add(table_id * TABLE_SIZE) };
        let tbl_maxs = unsafe { table_maxs.add(table_id * TABLE_SIZE) };

        let tid = thread::index_1d().get();
        let n_threads = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let total = raw_text.len();
        let chunk = (total + n_threads - 1) / n_threads;
        let start = tid * chunk;
        let end = ((tid + 1) * chunk).min(total);

        // Idle threads still must reach the sync_threads() below.
        if start < total {
            if let Some(mut pos) = line_start(raw_text, tid, start, total) {
                while pos < end {
                    let Some((name_len, temp, line_len)) = parse_line(&raw_text[pos..], end - pos)
                    else {
                        break;
                    };
                    let name = &raw_text[pos..pos + name_len];
                    let h = station_hash(name);
                    let global_slot = find_or_claim_slot(name, h, tbl_name_lens, tbl_names);
                    let local_slot = global_slot & (LOCAL_SIZE - 1);

                    let slot_atom = unsafe { DeviceAtomicI32::from_ptr(sh_slots.add(local_slot)) };
                    let stored = slot_atom.load(AtomicOrdering::Acquire);

                    // Use the shared slot: claim if empty, reuse if ours, else go global.
                    let use_local = if stored == global_slot as i32 {
                        true
                    } else if stored == -1 {
                        slot_atom
                            .compare_exchange(
                                -1,
                                global_slot as i32,
                                AtomicOrdering::AcqRel,
                                AtomicOrdering::Acquire,
                            )
                            .is_ok()
                    } else {
                        false
                    };

                    if use_local {
                        unsafe {
                            DeviceAtomicI64::from_ptr(sh_sums.add(local_slot))
                                .fetch_add(temp as i64, AtomicOrdering::Relaxed);
                            DeviceAtomicI32::from_ptr(sh_cnts.add(local_slot))
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            DeviceAtomicI32::from_ptr(sh_mins.add(local_slot))
                                .fetch_min(temp as i32, AtomicOrdering::Relaxed);
                            DeviceAtomicI32::from_ptr(sh_maxs.add(local_slot))
                                .fetch_max(temp as i32, AtomicOrdering::Relaxed);
                        }
                    } else {
                        update_aggregates(
                            global_slot,
                            temp,
                            tbl_sums,
                            tbl_cnts,
                            tbl_mins,
                            tbl_maxs,
                        );
                    }

                    pos += line_len;
                }
            }
        }

        thread::sync_threads();

        // Flush occupied shared slots to global.
        let mut i = local_tid;
        while i < LOCAL_SIZE {
            let gs = unsafe { *sh_slots.add(i) };
            if gs >= 0 {
                let gs = gs as usize;
                unsafe {
                    DeviceAtomicI64::from_ptr(tbl_sums.add(gs))
                        .fetch_add(*sh_sums.add(i), AtomicOrdering::Relaxed);
                    DeviceAtomicI32::from_ptr(tbl_cnts.add(gs))
                        .fetch_add(*sh_cnts.add(i), AtomicOrdering::Relaxed);
                    DeviceAtomicI32::from_ptr(tbl_mins.add(gs))
                        .fetch_min(*sh_mins.add(i), AtomicOrdering::Relaxed);
                    DeviceAtomicI32::from_ptr(tbl_maxs.add(gs))
                        .fetch_max(*sh_maxs.add(i), AtomicOrdering::Relaxed);
                }
            }
            i += block_size;
        }
    }

    fn line_start(raw_text: &[u8], tid: usize, start: usize, total: usize) -> Option<usize> {
        if tid == 0 {
            return Some(0);
        }
        let mut p = start;
        while p < total && raw_text[p] != b'\n' {
            p += 1;
        }
        if p + 1 < total { Some(p + 1) } else { None }
    }

    fn station_hash(name: &[u8]) -> usize {
        let mut h: usize = 0xcbf29ce484222325_usize;
        for &b in name {
            h ^= b as usize;
            h = h.wrapping_mul(0x00000100000001B3);
        }
        h
    }

    fn update_aggregates(
        slot: usize,
        temp: i16,
        table_sums: *mut i64,
        table_cnts: *mut i32,
        table_mins: *mut i32,
        table_maxs: *mut i32,
    ) {
        unsafe {
            DeviceAtomicI64::from_ptr(table_sums.add(slot))
                .fetch_add(temp as i64, AtomicOrdering::Relaxed);
            DeviceAtomicI32::from_ptr(table_cnts.add(slot)).fetch_add(1, AtomicOrdering::Relaxed);
            DeviceAtomicI32::from_ptr(table_mins.add(slot))
                .fetch_min(temp as i32, AtomicOrdering::Relaxed);
            DeviceAtomicI32::from_ptr(table_maxs.add(slot))
                .fetch_max(temp as i32, AtomicOrdering::Relaxed);
        }
    }

    // table_name_lens sentinels: 0 = empty, -1 = being written, >0 = name length.
    // CAS(0 → -1) claims a slot; the length is stored after the name bytes are copied.
    fn find_or_claim_slot(
        name: &[u8],
        h: usize,
        table_name_lens: *mut i32,
        table_names: *mut u8,
    ) -> usize {
        let mut slot = h & (TABLE_SIZE - 1);

        loop {
            let len_atom = unsafe { DeviceAtomicI32::from_ptr(table_name_lens.add(slot)) };
            match len_atom.load(AtomicOrdering::Acquire) {
                0 => {
                    if len_atom
                        .compare_exchange(0, -1, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
                        .is_ok()
                    {
                        for (j, &b) in name.iter().enumerate() {
                            unsafe {
                                *table_names.add(slot * NAME_MAX + j) = b;
                            }
                        }
                        len_atom.store(name.len() as i32, AtomicOrdering::Release);
                        return slot;
                    }
                }
                stored if stored > 0 => {
                    if stored as usize == name.len() && slot_name_matches(table_names, slot, name) {
                        return slot;
                    }
                    slot = (slot + 1) & (TABLE_SIZE - 1);
                }
                -1 => {} // being written, spin
                _ => slot = (slot + 1) & (TABLE_SIZE - 1),
            }
        }
    }

    fn slot_name_matches(table_names: *mut u8, slot: usize, name: &[u8]) -> bool {
        let base = slot * NAME_MAX;
        for (j, &b) in name.iter().enumerate() {
            if unsafe { *table_names.add(base + j) } != b {
                return false;
            }
        }
        true
    }

    fn parse_line(text: &[u8], limit: usize) -> Option<(usize, i16, usize)> {
        let max = text.len().min(limit);
        let mut sep = 0;
        while sep < max && text[sep] != b';' {
            sep += 1;
        }
        if sep >= max {
            return None;
        }
        let (temp, consumed) = parse_temp(&text[sep + 1..max])?;
        Some((sep.min(NAME_MAX), temp, sep + 1 + consumed))
    }

    // "-12.3\n" → (-123, 6); integer tenths, no floats.
    fn parse_temp(text: &[u8]) -> Option<(i16, usize)> {
        let mut i = 0;
        let neg = !text.is_empty() && text[0] == b'-';
        if neg {
            i += 1;
        }
        let mut val: i16 = 0;
        while i < text.len() && text[i] != b'.' && text[i] != b'\n' {
            val = val * 10 + (text[i] - b'0') as i16;
            i += 1;
        }
        if i < text.len() && text[i] == b'.' {
            i += 1;
            if i < text.len() && text[i] != b'\n' {
                val = val * 10 + (text[i] - b'0') as i16;
                i += 1;
            }
        }
        while i < text.len() && text[i] != b'\n' {
            i += 1;
        }
        if i >= text.len() {
            return None;
        }
        Some((if neg { -val } else { val }, i + 1))
    }
}

// Split into chunks aligned to newline boundaries so no record straddles a seam.
fn line_chunks(data: &[u8], target: usize) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let n = data.len();
    let mut start = 0;
    while start < n {
        let mut end = (start + target).min(n);
        if end < n {
            // Back up to the last newline.
            while end > start && data[end - 1] != b'\n' {
                end -= 1;
            }
            if end == start {
                // No newline in range (one huge line); extend forward.
                end = (start + target).min(n);
                while end < n && data[end - 1] != b'\n' {
                    end += 1;
                }
            }
        }
        chunks.push((start, end));
        start = end;
    }
    chunks
}

// Fault the mmap in parallel so the disk gets a deep request queue instead of one
// sequential fault stream. One byte per 4 KB page forces it resident; on a warm
// cache this is just a cheap scan.
fn prefault(data: &[u8], threads: usize) {
    let n = data.len();
    let band = n.div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let start = t * band;
            let end = (start + band).min(n);
            if start >= end {
                break;
            }
            let slice = &data[start..end];
            s.spawn(move || {
                let mut acc: u8 = 0;
                let mut i = 0;
                while i < slice.len() {
                    acc = acc.wrapping_add(slice[i]);
                    i += 4096;
                }
                std::hint::black_box(acc);
            });
        }
    });
}

fn fmt_val(tenths: i32) -> String {
    let sign = if tenths < 0 { "-" } else { "" };
    let abs = tenths.unsigned_abs();
    format!("{sign}{}.{}", abs / 10, abs % 10)
}

fn fmt_mean(sum: i64, cnt: i32) -> String {
    fmt_val((sum as f64 / cnt as f64).round() as i32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_total = std::time::Instant::now();

    let ctx = CudaContext::new(0)?;
    let s0 = ctx.default_stream();

    let file = std::fs::File::open("./1brc/measurements.txt")?;
    let mmap = unsafe { Mmap::map(&file)? };

    // PREFAULT_THREADS overrides the thread count; 0 disables prefault.
    const NUM_OF_THREADS: usize = 16;
    let prefault_threads: usize = std::env::var("PREFAULT_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(NUM_OF_THREADS);
    let t_disk = std::time::Instant::now();
    if prefault_threads > 0 {
        prefault(&mmap[..], prefault_threads);
    }
    eprintln!(
        "Disk read:     {:>8.3}s  ({prefault_threads} threads)",
        t_disk.elapsed().as_secs_f64()
    );

    let t0 = std::time::Instant::now();

    // Aggregation state, allocated once and accumulated across all chunks via atomics.
    // Init on s0 before forking s1 so the inits are visible to s1's kernels.
    let total_slots = TABLE_SIZE * NUM_TABLES;
    let name_lens_dev = DeviceBuffer::<i32>::zeroed(&s0, total_slots)?;
    let names_dev = DeviceBuffer::<u8>::zeroed(&s0, total_slots * NAME_MAX)?;
    let sums_dev = DeviceBuffer::<i64>::zeroed(&s0, total_slots)?;
    let cnts_dev = DeviceBuffer::<i32>::zeroed(&s0, total_slots)?;
    let mins_dev = DeviceBuffer::from_host(&s0, &vec![i32::MAX; total_slots])?;
    let maxs_dev = DeviceBuffer::from_host(&s0, &vec![i32::MIN; total_slots])?;

    // s1 runs concurrently, after s0's inits.
    let s1 = s0.fork()?;
    let streams = [&s0, &s1];

    let module = kernels::load(&ctx)?;
    let config = LaunchConfig {
        grid_dim: (NUM_BLOCKS, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // CHUNK_MB sets the per-chunk read size (default 256 MB).
    const CHUNK_MB: usize = 256;
    let chunk_mb: usize = std::env::var("CHUNK_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(CHUNK_MB);
    let chunks = line_chunks(&mmap, chunk_mb << 20);
    eprintln!("chunk size:    {chunk_mb} MB  ({} chunks)", chunks.len());

    // Keep every chunk's buffer alive: dropping one mid-loop calls cuMemFree, which
    // synchronizes the whole device and kills the overlap. They free together at the end.
    let mut keep_alive = Vec::with_capacity(chunks.len());

    for (i, &(start, end)) in chunks.iter().enumerate() {
        let stream = streams[i % 2];
        // This chunk's H2D overlaps the other stream's kernel; no sync between chunks.
        let text_dev = DeviceBuffer::from_host(stream, &mmap[start..end])?;
        module.process_measurements(
            stream,
            config,
            &text_dev,
            name_lens_dev.cu_deviceptr() as *mut i32,
            names_dev.cu_deviceptr() as *mut u8,
            sums_dev.cu_deviceptr() as *mut i64,
            cnts_dev.cu_deviceptr() as *mut i32,
            mins_dev.cu_deviceptr() as *mut i32,
            maxs_dev.cu_deviceptr() as *mut i32,
        )?;
        keep_alive.push(text_dev);
    }

    s0.synchronize()?;
    s1.synchronize()?;
    eprintln!("Pipeline:      {:>8.3}s", t0.elapsed().as_secs_f64());

    let t2 = std::time::Instant::now();
    let name_lens = name_lens_dev.to_host_vec(&s0)?;
    let names = names_dev.to_host_vec(&s0)?;
    let sums = sums_dev.to_host_vec(&s0)?;
    let cnts = cnts_dev.to_host_vec(&s0)?;
    let mins = mins_dev.to_host_vec(&s0)?;
    let maxs = maxs_dev.to_host_vec(&s0)?;
    eprintln!("D2H transfer:  {:>8.3}s", t2.elapsed().as_secs_f64());

    let t3 = std::time::Instant::now();

    // Combine the NUM_TABLES partial tables on the CPU.
    let mut station_map: std::collections::HashMap<String, (i32, i32, i64, i32)> =
        std::collections::HashMap::with_capacity(10_000);

    for t in 0..NUM_TABLES {
        let base = t * TABLE_SIZE;
        for s in 0..TABLE_SIZE {
            let idx = base + s;
            if name_lens[idx] > 0 {
                let len = name_lens[idx] as usize;
                let name = String::from_utf8_lossy(&names[idx * NAME_MAX..idx * NAME_MAX + len])
                    .into_owned();
                let e = station_map
                    .entry(name)
                    .or_insert((i32::MAX, i32::MIN, 0i64, 0i32));
                e.0 = e.0.min(mins[idx]);
                e.1 = e.1.max(maxs[idx]);
                e.2 += sums[idx];
                e.3 += cnts[idx];
            }
        }
    }

    let mut stations: Vec<(String, i32, i32, i64, i32)> = station_map
        .into_iter()
        .map(|(name, (min, max, sum, cnt))| (name, min, max, sum, cnt))
        .collect();
    stations.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let entries: Vec<String> = stations
        .iter()
        .map(|(name, min, max, sum, cnt)| {
            format!(
                "{name}={}/{}/{}",
                fmt_val(*min),
                fmt_mean(*sum, *cnt),
                fmt_val(*max)
            )
        })
        .collect();
    println!("{{{}}}", entries.join(", "));
    eprintln!("Sort + output: {:>8.3}s", t3.elapsed().as_secs_f64());

    eprintln!("─────────────────────────────");
    eprintln!("Total:         {:>8.3}s", t_total.elapsed().as_secs_f64());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem_speed() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let t_read = std::time::Instant::now();
        let data = std::fs::read("./1brc/measurements.txt")?; // forces the full disk read now
        eprintln!("disk read:   {:>8.3}s", t_read.elapsed().as_secs_f64());

        let t_copy = std::time::Instant::now();
        let text_dev = DeviceBuffer::from_host(&stream, &data)?;
        stream.synchronize()?;
        eprintln!("H2D only:    {:>8.3}s", t_copy.elapsed().as_secs_f64());
        Ok(())
    }
}
