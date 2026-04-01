import socket
import time
import sys
import os

# Root of generated FlatBuffers
gen_root = os.path.abspath("src/ingestion/generated")
sys.path.append(gen_root)

try:
    import flatbuffers
    # Namespace is src.data.fbs
    from src.data.fbs import MarketTickFB
    print(f"Successfully imported MarketTickFB from {gen_root}")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    # Fallback for manual inspection of generated structure
    import glob
    print(f"Searching for MarketTickFB.py in {gen_root}:")
    print(glob.glob(f"{gen_root}/**/MarketTickFB.py", recursive=True))
    sys.exit(1)

def create_tick(builder, symbol, price, volume, timestamp):
    sym = builder.CreateString(symbol)
    MarketTickFB.MarketTickFBStart(builder)
    MarketTickFB.MarketTickFBAddSymbol(builder, sym)
    MarketTickFB.MarketTickFBAddPrice(builder, price)
    MarketTickFB.MarketTickFBAddVolume(builder, volume)
    MarketTickFB.MarketTickFBAddTimestamp(builder, timestamp)
    return MarketTickFB.MarketTickFBEnd(builder)

def run_load_gen(target_addr=("127.0.0.1", 5555), duration=5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    builder = flatbuffers.Builder(1024)
    
    # Pre-build a template tick to avoid overhead in the hot loop
    builder.Clear()
    tick_offset = create_tick(builder, "AAPL", 150.0, 100, time.time())
    builder.Finish(tick_offset)
    packet = builder.Output()
    
    print(f"Starting 1M+ ticks/sec load generator towards {target_addr}...")
    start_time = time.time()
    count = 0
    
    try:
        while time.time() - start_time < duration:
            sock.sendto(packet, target_addr)
            count += 1
            if count % 1000000 == 0:
                print(f"Sent {count} ticks...")
    except KeyboardInterrupt:
        pass
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    print(f"\nFinal Results:")
    print(f"Total Ticks Sent: {count}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"Average Throughput: {int(count/total_elapsed)} ticks/sec")

if __name__ == "__main__":
    run_load_gen(duration=3)
    time.sleep(1)
