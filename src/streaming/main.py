from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from strawberry.fastapi import GraphQLRouter
from src.streaming.graphql.schema import schema
from src.shared.observability import logging_middleware
import asyncio
import json
import time


app = FastAPI(title="BS-Opt Market Data Service")
app.middleware("http")(logging_middleware)

graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/marketdata")
async def websocket_marketdata(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # You can receive messages if the client sends any (e.g., subscribe/unsubscribe)
            # data = await websocket.receive_text()
            # print(f"Received from client: {data}")

            # Send mock market data
            mock_data = {
                "symbol": "AAPL",
                "time": int(time.time()),
                "open": 150 + (time.time() % 10),
                "high": 160 + (time.time() % 10),
                "low": 140 + (time.time() % 10),
                "close": 150 + (time.time() % 10)
            }
            await websocket.send_text(json.dumps(mock_data))
            await asyncio.sleep(1)  # Send data every second
    except WebSocketDisconnect:
        print("Client disconnected from /marketdata websocket")
    except Exception as e:
        print(f"Error in /marketdata websocket: {e}")
