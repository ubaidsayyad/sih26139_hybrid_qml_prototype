"""
run_server.py
-------------
Production-grade WSGI Server Launcher for the SIH26139 Hybrid QML Platform.

Uses Waitress (multi-threaded, robust on Windows, zero-reloader process restarts)
to serve the Flask application reliably for live demonstrations and testing.

Usage:
    python run_server.py
"""

import os
import sys
import socket

# Ensure prototype root is in Python sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from waitress import serve


def get_local_ip():
    """Retrieve the host machine's primary local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    threads = int(os.environ.get("WAITRESS_THREADS", 8))
    local_ip = get_local_ip()

    print("=" * 72)
    print("  HYBRID QUANTUM-CLASSICAL MACHINE LEARNING PLATFORM (SIH26139)")
    print("  Production WSGI Server (Waitress) · Multi-Threaded Engine")
    print("=" * 72)
    print(f"  * Host Binding:      {host}:{port}")
    print(f"  * Local Web URL:     http://localhost:{port}")
    print(f"  * Network Access:    http://{local_ip}:{port}")
    print(f"  * Worker Threads:    {threads} concurrent request workers")
    print(f"  * Debug / Reloader:  DISABLED (Immune to file write restart loop)")
    print(f"  * System Health:     http://localhost:{port}/health")
    print("=" * 72)
    print("  Server is READY for live demonstration and high-concurrency requests.\n")

    # Serve the WSGI application with multi-threading
    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        connection_limit=1000,
        channel_timeout=30,
        ident="HybridQML-WSGI"
    )


if __name__ == "__main__":
    main()
