
use pyo3::prelude::*;
use socket2::{Socket, Domain, Type, Protocol};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;

#[pyclass]
pub struct RustPulse {
    running: Arc<AtomicBool>,
    interface: String,
    port: u16,
}

#[pymethods]
impl RustPulse {
    #[new]
    fn new(interface: String, port: u16) -> Self {
        RustPulse {
            running: Arc::new(AtomicBool::new(false)),
            interface,
            port,
        }
    }

    fn start(&mut self, shm_path: String, cpu_core: usize) -> PyResult<()> {
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let interface = self.interface.clone();
        let port = self.port;

        thread::spawn(move || {
            // 🚀 SILICON LOCKDOWN: Pin to core (Linux specific)
            #[cfg(target_os = "linux")]
            unsafe {
                let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(cpu_core, &mut cpuset);
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
            }

            // 1. Initialize SHM Mapping
            // SHM is typically in /dev/shm/ on Linux
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&shm_path)
                .expect("Failed to open SHM file");
            
            let mut mmap = unsafe { MmapOptions::new().map_mut(&file).expect("Failed to mmap SHM") };

            // 2. Initialize Socket (Raw or UDP)
            let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
                .expect("Failed to create socket");
            
            let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().unwrap();
            socket.bind(&addr.into()).expect("Failed to bind socket");
            socket.set_nonblocking(true).expect("Failed to set non-blocking");

            let mut buf = [std::mem::MaybeUninit::<u8>::uninit(); 2048];
            
            while running.load(Ordering::Relaxed) {
                match socket.recv_from(&mut buf) {
                    Ok((n, _addr)) => {
                        if n >= 32 {
                            // 🚀 RUST SPEED: Raw pointer mapping to SHM
                            let receive_ts_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as i64;

                            let head_ptr = mmap.as_mut_ptr() as *mut i64;
                            let current_head = unsafe { *head_ptr };
                            let idx = (current_head % 100000) as usize;
                            
                            let tick_ptr = unsafe { mmap.as_mut_ptr().add(8 + idx * 40) };
                            
                            unsafe {
                                // Safety: We know buf[0..32] is initialized because n >= 32
                                let init_buf = std::mem::transmute::<&[std::mem::MaybeUninit<u8>], &[u8]>(&buf[..32]);
                                std::ptr::copy_nonoverlapping(init_buf.as_ptr(), tick_ptr, 32);
                                std::ptr::copy_nonoverlapping(
                                    &receive_ts_ns as *const i64 as *const u8,
                                    tick_ptr.add(32),
                                    8
                                );
                                *head_ptr = current_head + 1;
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::yield_now();
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(())
    }

    fn stop(&mut self) -> PyResult<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
}

#[pymodule]
fn bsopt_pulse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPulse>()?;
    Ok(())
}
