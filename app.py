"""Live Web UI Server for the Turtle Trend Following US Stock Ranking Screener."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import BASE_DIR, DB_PATH, OUTPUT_DIR
from dashboard import DASHBOARD_HTML_PATH, generate_dashboard_html
from database import get_ticker_metadata, load_price_histories
from indicators import compute_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("turtle_ui_server")


class TurtleDashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving output/ directory and live /api/run-scan endpoint."""

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.path = "/dashboard.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/run-scan":
            try:
                cmd = [sys.executable, str(BASE_DIR / "screener.py")]
                subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
                payload = {"status": "ok", "scan_date": "latest"}
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                err = json.dumps({"status": "error", "message": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            return

        self.send_error(404, "Endpoint not found")


def ensure_dashboard_built() -> Path:
    """Build or refresh output/dashboard.html from SQLite before starting server."""
    ticker_meta = get_ticker_metadata(db_path=DB_PATH)
    raw_histories = load_price_histories(tickers=sorted(ticker_meta.keys()), db_path=DB_PATH)
    ind_frames = {tkr: compute_indicators(df) for tkr, df in raw_histories.items() if len(df) >= 30}
    return generate_dashboard_html(indicator_frames=ind_frames, db_path=DB_PATH, output_path=DASHBOARD_HTML_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Turtle Screener Web UI Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve dashboard on (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on startup")
    args = parser.parse_args()

    logger.info("Building dashboard payload from SQLite database...")
    dash_path = ensure_dashboard_built()
    logger.info("Dashboard ready at %s", dash_path)

    handler = partial(TurtleDashboardHandler, directory=str(OUTPUT_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/dashboard.html"
    logger.info("Turtle Screener Web UI running at %s", url)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web server.")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
