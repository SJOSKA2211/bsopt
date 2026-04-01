use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use socket2::{Socket, Domain, Type, Protocol};
use crate::quarantine::{QuarantineBuffer, RejectedTick};
use crate::generated::market_tick_generated::src::data::fbs::{root_as_market_tick_fb};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct NativeIngestEngine {
    addr: SocketAddr,
    quarantine: Arc<QuarantineBuffer>,
    pub processed_count: Arc<AtomicU64>,
}

impl NativeIngestEngine {
    pub fn new(addr: SocketAddr, quarantine: Arc<QuarantineBuffer>) -> Self {
        Self {
            addr,
            quarantine,
            processed_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub async fn run(&self) -> anyhow::Result<()> {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        #[cfg(all(unix, not(target_os = "macos")))]
        socket.set_reuse_port(true)?;
        socket.bind(&self.addr.into())?;
        socket.set_nonblocking(true)?;
        
        let udp = UdpSocket::from_std(socket.into())?;
        let mut buf = vec![0u8; 1500]; // Standard MTU size

        println!("NativeIngestEngine listening on {}", self.addr);

        loop {
            let (len, _) = udp.recv_from(&mut buf).await?;
            let data = &buf[..len];

            // 1. Zero-copy Parsing (FlatBuffers)
            if let Ok(tick) = root_as_market_tick_fb(data) {
                let price = tick.price();
                let symbol_opt = tick.symbol();
                let symbol = symbol_opt.unwrap_or("UNKNOWN");
                
                // 2. High-speed Validation (Native)
                if price <= 0.0 || price > 1_000_000.0 {
                    self.quarantine.push(&RejectedTick {
                        symbol: symbol.to_string(),
                        price,
                        volume: tick.volume(),
                        timestamp: tick.timestamp(),
                        reason: "INVALID_PRICE".to_string(),
                    });
                    continue;
                }

                // 3. Increment counters
                self.processed_count.fetch_add(1, Ordering::Relaxed);
            } else {
                self.quarantine.TotalRejections.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
