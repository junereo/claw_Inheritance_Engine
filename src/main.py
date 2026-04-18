from __future__ import annotations

import uvicorn
import os

def main() -> int:
    """
    Inheritance Engine Server Entry Point.
    Replaces the legacy terminal UI with a FastAPI REST server.
    """
    print("Starting PromToon Inheritance Engine API Server...")
    print("Host: 0.0.0.0, Port: 8000")
    
    # We import the app here to avoid circular imports if any
    from .api import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
