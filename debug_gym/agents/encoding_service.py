"""
Sentence encoding service that can be shared across multiple RAG agents.
This service hosts the sentence encoder as a separate process/service to avoid
loading multiple copies of the model in memory.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import List, Optional

import numpy as np
import requests

from debug_gym.agents.utils import SentenceEncoder
from debug_gym.logger import DebugGymLogger


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread pool server to handle multiple requests concurrently."""

    daemon_threads = True


class EncodingServiceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the encoding service."""

    def __init__(self, encoder, *args, **kwargs):
        self.encoder = encoder
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests (health checks)."""
        try:
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))
            else:
                self.send_error(404, "Endpoint not found")
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")

    def do_POST(self):
        """Handle POST requests for encoding."""
        try:
            if self.path == "/encode":
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))

                texts = data.get("texts", [])
                batch_size = data.get("batch_size", 16)
                is_query = data.get("is_query", False)

                if not texts:
                    self.send_error(400, "No texts provided")
                    return

                # Encode the texts
                if is_query:
                    embeddings = self.encoder.encode_sentence_querying(
                        texts, batch_size=batch_size
                    )
                else:
                    embeddings = self.encoder.encode_sentence(
                        texts, batch_size=batch_size
                    )

                # Convert to list for JSON serialization
                embeddings_list = embeddings.tolist()

                response_data = {
                    "embeddings": embeddings_list,
                    "shape": list(embeddings.shape),
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            else:
                self.send_error(404, "Endpoint not found")

        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")

    def log_message(self, format, *args):
        """Override to use proper logging instead of stderr."""
        # Use a simple logger for HTTP server messages
        logger = DebugGymLogger("EncodingService")
        logger.info(f"EncodingService: {format % args}")


class EncodingService:
    """Sentence encoding service that can be shared across multiple processes."""

    def __init__(self, model_name: str, port: int = 8765, host: str = "localhost"):
        self.model_name = model_name
        self.port = port
        self.host = host
        self.encoder = None
        self.server = None
        self.server_thread = None
        self.logger = DebugGymLogger(__name__)

    def start_service(self):
        """Start the encoding service."""
        self.logger.info(f"Initializing sentence encoder with model: {self.model_name}")
        self.encoder = SentenceEncoder(model_name=self.model_name)

        # Create a handler class with the encoder
        def handler_factory(*args, **kwargs):
            return EncodingServiceHandler(self.encoder, *args, **kwargs)

        self.server = ThreadedHTTPServer((self.host, self.port), handler_factory)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        self.logger.info(f"Encoding service started on {self.host}:{self.port}")

    def stop_service(self):
        """Stop the encoding service."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
        self.logger.info("Encoding service stopped")


class EncodingServiceClient:
    """Client for interacting with the encoding service."""

    def __init__(self, host: str = "localhost", port: int = 8765, timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.logger = DebugGymLogger(__name__)

    def is_service_available(self) -> bool:
        """Check if the encoding service is available."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def wait_for_service(self, max_wait_time: int = 60) -> bool:
        """Wait for the service to become available."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            if self.is_service_available():
                return True
            time.sleep(1)
        return False

    def encode_sentence(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode sentences using the service."""
        data = {"texts": texts, "batch_size": batch_size, "is_query": False}

        response = requests.post(
            f"{self.base_url}/encode", json=data, timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Encoding service error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return np.array(result["embeddings"])

    def encode_sentence_querying(
        self, texts: List[str], batch_size: int = 16
    ) -> np.ndarray:
        """Encode query sentences using the service."""
        data = {"texts": texts, "batch_size": batch_size, "is_query": True}

        response = requests.post(
            f"{self.base_url}/encode", json=data, timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Encoding service error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return np.array(result["embeddings"])


def start_encoding_service_standalone(
    model_name: str, port: int = 8765, host: str = "localhost"
):
    """Standalone function to start the encoding service."""
    service = EncodingService(model_name, port, host)

    try:
        service.start_service()
        print(f"Encoding service running on {host}:{port}")
        print("Press Ctrl+C to stop the service")

        # Keep the service running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down encoding service...")
        service.stop_service()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start sentence encoding service")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Embedding-0.6B", help="Model name"
    )
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")

    args = parser.parse_args()
    start_encoding_service_standalone(args.model, args.port, args.host)
