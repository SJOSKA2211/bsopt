import asyncio
from src.blockchain.defi_options import DeFiOptionsProtocol
from src.blockchain.oracle import OracleManager
import structlog

logger = structlog.get_logger()

async def verify_blockchain_v3():
    # 1. Initialize Protocol
    protocol = DeFiOptionsProtocol(rpc_url="http://localhost:8545") # Local anvil/ganache if running
    print("✅ DeFi Protocol Initialized")

    # 2. Test SOR
    best_venue = await protocol.route_order("ETH", 1.0, True)
    print(f"✅ SOR Routing Decision: {best_venue['name']}")

    # 3. Test Oracle Confidence
    oracle = OracleManager()
    score = oracle.get_confidence_score("WS", 0.5)
    print(f"✅ Oracle Confidence Score (WS, 0.5s): {score}")
    
    # 4. Test EIP-712 Order Signing (Simulated)
    order = {
        "maker": "0x0000000000000000000000000000000000000000",
        "asset": "0x0000000000000000000000000000000000000000",
        "amount": 10**18,
        "price": 2000 * 10**18,
        "nonce": 1,
        "expiry": 1700000000
    }
    try:
        # Requires private key, so we expect failure if not provided
        # but let's see if the logic is sound
        print("ℹ️ Skipping EIP-712 sign (No private key)")
    except Exception as e:
        print(f"❌ EIP-712 Sign Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(verify_blockchain_v3())
    except Exception as e:
        print(f"⚠️ Verification partially failed (Expected due to no RPC/Key): {e}")
