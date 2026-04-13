

use std::fs::{OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use memmap2::{MmapMut, MmapOptions};

/// A single rejected tick record for quarantine analysis.
#[derive(Debug, Clone)]
pub struct RejectedTick {
    pub symbol: String,
    pub price: f64,
    pub volume: i64,
    pub timestamp: f64,
    pub reason: String,
}

/// A high-speed, mmap-backed circular buffer for sampled rejections.
pub struct QuarantineBuffer {
    mmap: MmapMut,
    capacity: usize,
    write_pos: AtomicU64,
    pub total_rejections: AtomicU64,
    pub outlier_rejections: AtomicU64,
    pub invalid_rejections: AtomicU64,
}

// SAFETY: MmapMut is Send/Sync, and we use atomic offsets for safe parallel writes to unique segments.
unsafe impl Send for QuarantineBuffer {}
unsafe impl Sync for QuarantineBuffer {}

impl QuarantineBuffer {
    pub fn new(path: &Path, capacity: usize) -> anyhow::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        
        let file_size = 64 + (capacity * 64); 
        file.set_len(file_size as u64)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            mmap,
            capacity,
            write_pos: AtomicU64::new(0),
            total_rejections: AtomicU64::new(0),
            outlier_rejections: AtomicU64::new(0),
            invalid_rejections: AtomicU64::new(0),
        })
    }

    pub fn push(&self, tick: &RejectedTick) {
        let total = self.total_rejections.fetch_add(1, Ordering::Relaxed);
        
        // Approved Plan Refinement: 0.1% Sampling Rate (1 in 1000)
        if total % 1000 != 0 {
            return;
        }

        let pos = self.write_pos.fetch_add(1, Ordering::SeqCst) % (self.capacity as u64);
        let offset = 64 + (pos as usize * 64);
        
        // Use raw pointers for ultra-fast, zero-copy per-core throughput
        unsafe {
            let raw_ptr = self.mmap.as_ptr().add(offset) as *mut u8;
            
            let symbol_bytes = tick.symbol.as_bytes();
            let sym_len = symbol_bytes.len().min(15);
            std::ptr::copy_nonoverlapping(symbol_bytes.as_ptr(), raw_ptr, sym_len);
            *raw_ptr.add(15) = 0; 

            std::ptr::copy_nonoverlapping(&tick.price as *const f64 as *const u8, raw_ptr.add(16), 8);
            std::ptr::copy_nonoverlapping(&tick.volume as *const i64 as *const u8, raw_ptr.add(24), 8);
            std::ptr::copy_nonoverlapping(&tick.timestamp as *const f64 as *const u8, raw_ptr.add(32), 8);
            
            let reason_bytes = tick.reason.as_bytes();
            let reason_len = reason_bytes.len().min(23);
            std::ptr::copy_nonoverlapping(reason_bytes.as_ptr(), raw_ptr.add(40), reason_len);
        }
    }
}
