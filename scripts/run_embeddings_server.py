#!/usr/bin/env python3
"""
Thin uvicorn launcher for app.embeddings_server.

Reads EMBEDDINGS_HOST / EMBEDDINGS_PORT (defaults 0.0.0.0:8081) so the
systemd unit / Quadlet on the infra side can drop this in with no flags.

  python scripts/run_embeddings_server.py
  EMBEDDINGS_PORT=80 python scripts/run_embeddings_server.py
"""

import os

import uvicorn


def main() -> None:
    host = os.getenv("EMBEDDINGS_HOST", "0.0.0.0")
    port = int(os.getenv("EMBEDDINGS_PORT", "8081"))
    uvicorn.run(
        "app.embeddings_server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
