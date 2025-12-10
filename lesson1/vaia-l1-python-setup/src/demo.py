"""
VAIA L1: Demo Script
Runs demo operations to update dashboard metrics
"""
import asyncio
import httpx
import time
import sys

async def run_demo_operations(base_url: str = "http://localhost:8000", count: int = 10):
    """Run multiple demo operations"""
    print(f"🚀 Running {count} demo operations...")
    
    client = httpx.AsyncClient()
    try:
        for i in range(count):
            try:
                response = await client.post(f"{base_url}/api/demo", json={"action": "run"})
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Demo {i+1}/{count}: {data['message']}")
                else:
                    print(f"❌ Demo {i+1}/{count}: Failed with status {response.status_code}")
            except Exception as e:
                print(f"❌ Demo {i+1}/{count}: Error - {e}")
            
            # Small delay between operations
            await asyncio.sleep(0.5)
        
        # Get final metrics
        try:
            response = await client.get(f"{base_url}/api/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print("\n📊 Final Metrics:")
                print(f"   Total Requests: {metrics['requests_total']}")
                print(f"   Demo Executions: {metrics['demo_executions']}")
                print(f"   Successful Operations: {metrics['successful_operations']}")
                print(f"   Requests/Second: {metrics['requests_per_second']:.2f}")
                print(f"   Avg Response Time: {metrics['average_response_time']:.2f}ms")
        except Exception as e:
            print(f"❌ Error fetching metrics: {e}")
    finally:
        await client.aclose()

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(run_demo_operations(count=count))
