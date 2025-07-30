"""
Retrieval service that can be shared across multiple RAG agents.
This service hosts the vector index and retrieval logic as a separate process/service
to avoid loading multiple copies of the index in memory.

The service handles sentence encoding internally using local SentenceTransformer models,
providing a simplified architecture without external encoding service dependencies.
"""

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import List, Optional, Tuple

import numpy as np
import requests
import yaml

from debug_gym.agents.shared_cache import get_shared_cache_manager
from debug_gym.agents.utils import FaissRetriever, SentenceEncoder
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.logger import DebugGymLogger


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread pool server to handle multiple requests concurrently."""

    daemon_threads = True
    timeout = 60
    allow_reuse_address = True
    request_queue_size = 10

    def server_bind(self):
        """Override to set socket options."""
        import socket

        HTTPServer.server_bind(self)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class RetrievalServiceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the retrieval service."""

    def __init__(self, retrieval_manager, *args, **kwargs):
        self.retrieval_manager = retrieval_manager
        self.logger = DebugGymLogger("RetrievalService")
        super().__init__(*args, **kwargs)

    def log_request(self, code="-", size="-"):
        """Override to reduce logging noise."""
        pass

    def do_GET(self):
        """Handle GET requests (health checks)."""
        try:
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))
            elif self.path == "/indexes":
                # List available indexes
                indexes = list(self.retrieval_manager.indexes.keys())
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"indexes": indexes}).encode("utf-8"))
            else:
                self.send_error(404, "Endpoint not found")
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")

    def do_POST(self):
        """Handle POST requests for retrieval operations."""
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            if self.path == "/retrieve":
                self._handle_retrieve(data)
            elif self.path == "/build_index":
                self._handle_build_index(data)
            elif self.path == "/check_index":
                self._handle_check_index(data)
            else:
                self.send_error(404, "Endpoint not found")

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            try:
                self.send_error(500, f"Internal server error: {str(e)}")
            except:
                pass

    def _handle_retrieve(self, data):
        """Handle retrieval requests."""
        index_key = data.get("index_key")
        query_text = data.get("query_text")
        num_retrievals = data.get("num_retrievals", 1)

        if not index_key or not query_text:
            self.send_error(400, "index_key and query_text are required")
            return

        self.logger.info(
            f"Processing retrieval request for index '{index_key}', num_retrievals={num_retrievals}"
        )

        try:
            relevant_examples = self.retrieval_manager.retrieve(
                index_key, query_text, num_retrievals
            )

            response_data = {"relevant_examples": relevant_examples}
            response_bytes = json.dumps(response_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Connection", "close")
            self.end_headers()

            self.wfile.write(response_bytes)
            self.wfile.flush()

            try:
                self.connection.shutdown(1)
            except:
                pass

            self.logger.info("Retrieval request completed successfully")

        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            self.send_error(500, f"Retrieval error: {str(e)}")

    def _handle_build_index(self, data):
        """Handle index building requests."""
        index_key = data.get("index_key")
        experience_trajectory_path = data.get("experience_trajectory_path")
        rag_indexing_method = data.get("rag_indexing_method")
        sentence_encoder_model = data.get("sentence_encoder_model")
        rag_indexing_batch_size = data.get("rag_indexing_batch_size", 16)
        use_cache = data.get("use_cache", True)

        if not all(
            [
                index_key,
                experience_trajectory_path,
                rag_indexing_method,
                sentence_encoder_model,
            ]
        ):
            self.send_error(400, "Missing required parameters for index building")
            return

        self.logger.info(f"Building index '{index_key}'")

        try:
            success = self.retrieval_manager.build_index(
                index_key=index_key,
                experience_trajectory_path=experience_trajectory_path,
                rag_indexing_method=rag_indexing_method,
                sentence_encoder_model=sentence_encoder_model,
                rag_indexing_batch_size=rag_indexing_batch_size,
                use_cache=use_cache,
            )

            response_data = {"success": success, "index_key": index_key}
            response_bytes = json.dumps(response_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Connection", "close")
            self.end_headers()

            self.wfile.write(response_bytes)
            self.wfile.flush()

            try:
                self.connection.shutdown(1)
            except:
                pass

            self.logger.info(f"Index building completed successfully for '{index_key}'")

        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            self.send_error(500, f"Index building error: {str(e)}")

    def _handle_check_index(self, data):
        """Handle index existence check requests."""
        index_key = data.get("index_key")

        if not index_key:
            self.send_error(400, "index_key is required")
            return

        try:
            exists = self.retrieval_manager.has_index(index_key)

            response_data = {"exists": exists, "index_key": index_key}
            response_bytes = json.dumps(response_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Connection", "close")
            self.end_headers()

            self.wfile.write(response_bytes)
            self.wfile.flush()

            try:
                self.connection.shutdown(1)
            except:
                pass

        except Exception as e:
            self.logger.error(f"Error checking index: {str(e)}")
            self.send_error(500, f"Index check error: {str(e)}")


class RetrievalManager:
    """Manages multiple retrieval indexes and handles retrieval operations."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = DebugGymLogger(__name__)
        self.indexes = (
            {}
        )  # index_key -> {"retriever": FaissRetriever, "data_input": List[str], "data_label": List[str]}

        # Thread lock for index operations to prevent race conditions
        self.index_lock = threading.RLock()

        # Cache configuration
        self.cache_dir = self.config.get("rag_cache_dir", ".rag_cache")
        self.use_cache = self.config.get("rag_use_cache", True)

        if self.use_cache:
            self.cache_manager = get_shared_cache_manager(self.cache_dir)
        else:
            self.cache_manager = None

        # Sentence encoder configuration
        self.sentence_encoder_model = self.config.get(
            "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
        )

        # Initialize encoder
        self._initialize_encoder()

    def has_index(self, index_key: str) -> bool:
        """Check if an index exists."""
        with self.index_lock:
            return index_key in self.indexes

    def _initialize_encoder(self):
        """Initialize local sentence encoder."""
        self.logger.info(
            f"Initializing local sentence encoder with model: {self.sentence_encoder_model}"
        )
        self.encoder = SentenceEncoder(model_name=self.sentence_encoder_model)

    def parse_indexing_method(self, method: str):
        """Parse the indexing method from the configuration."""
        assert method is not None, "rag_indexing_method must be provided"

        method, step = method.rsplit("-", 1) if "-" in method else (method, "1")
        assert method in [
            "observation",
            "tool_name",
            "tool_call",
            "tool_call_with_reasoning",
        ], f"Invalid rag_indexing_method: {method}"
        assert step.isdigit(), f"Invalid step value: {step}"
        step = int(step)
        assert step > 0, "Step must be a positive integer."
        return [method, step]

    def load_experience_trajectory_from_file(
        self, file_path: str, max_examples: int = None
    ):
        """Load experience trajectories from a JSONL file."""
        experience_trajectories = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if max_examples and line_num > max_examples:
                        break
                    try:
                        experience_json = json.loads(line.strip())
                        satisfied_criteria = experience_json.get(
                            "satisfied_criteria", []
                        )
                        if (
                            "follows_proper_debugging_workflow"
                            not in satisfied_criteria
                            or "has_successful_outcome" not in satisfied_criteria
                        ):
                            continue
                        experience_trajectories.append(experience_json["messages"])
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping invalid JSON on line {line_num}")
        except Exception as e:
            self.logger.error(f"Error loading experience trajectories from file: {e}")

        return experience_trajectories

    def build_retrieval_dataset(self, experience_trajectories, rag_indexing_method):
        """Build a dataset for retrieval based on the loaded experience trajectories and the indexing method."""

        def find_last_k_messages_with_role(trajectory, role, k):
            """Find the last k messages with the specified role in the trajectory."""
            if isinstance(role, str):
                role = [role]
            messages = [msg for msg in trajectory if msg["role"] in role]
            return messages[-k:] if len(messages) >= k else messages

        method, step = rag_indexing_method
        data_input, data_label = [], []

        for trajectory in experience_trajectories:
            for i in range(len(trajectory)):
                if trajectory[i]["role"] != "assistant":
                    continue
                if "tool_calls" not in trajectory[i] or not trajectory[i]["tool_calls"]:
                    continue
                if (
                    "function" not in trajectory[i]["tool_calls"][0]
                    or not trajectory[i]["tool_calls"][0]["function"]
                ):
                    continue

                _label = {"tool_calls": trajectory[i]["tool_calls"][0]["function"]}
                if "content" in trajectory[i]:
                    _label["content"] = trajectory[i]["content"]
                label = json.dumps(_label)

                for __step in range(1, step + 1):
                    match method:
                        case "observation":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], ["user", "tool"], __step
                            )
                            if not input_list:
                                continue
                            input_list = [msg["content"] for msg in input_list]
                            input_text = " <STEP_DELIMITER> ".join(input_list)
                        case "tool_name":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], "assistant", __step
                            )
                            if not input_list:
                                continue
                            tool_name_list = []
                            for msg in input_list:
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    if (
                                        "function" in msg["tool_calls"][0]
                                        and msg["tool_calls"][0]["function"]
                                    ):
                                        tool_name = msg["tool_calls"][0].get("name", "")
                                        if tool_name:
                                            tool_name_list.append(tool_name)
                            if not tool_name_list:
                                continue
                            input_text = " <STEP_DELIMITER> ".join(tool_name_list)
                        case "tool_call":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], "assistant", __step
                            )
                            if not input_list:
                                continue
                            tool_call_list = []
                            for msg in input_list:
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    if (
                                        "function" in msg["tool_calls"][0]
                                        and msg["tool_calls"][0]["function"]
                                    ):
                                        tool_call = json.dumps(
                                            msg["tool_calls"][0]["function"]
                                        )
                                        tool_call_list.append(tool_call)
                            if not tool_call_list:
                                continue
                            input_text = " <STEP_DELIMITER> ".join(tool_call_list)
                        case "tool_call_with_reasoning":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], "assistant", __step
                            )
                            if not input_list:
                                continue
                            tool_call_with_reasoning_list = []
                            for msg in input_list:
                                tmp = {}
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    if (
                                        "function" in msg["tool_calls"][0]
                                        and msg["tool_calls"][0]["function"]
                                    ):
                                        tmp["tool_calls"] = msg["tool_calls"][0][
                                            "function"
                                        ]
                                if "content" in msg:
                                    tmp["content"] = msg["content"]
                                if tmp:
                                    tool_call_with_reasoning_list.append(
                                        json.dumps(tmp)
                                    )
                            if not tool_call_with_reasoning_list:
                                continue
                            input_text = " <STEP_DELIMITER> ".join(
                                tool_call_with_reasoning_list
                            )
                        case _:
                            raise ValueError(
                                f"Invalid rag_indexing_method: {method}. Supported methods: observation, tool_name, tool_call, tool_call_with_reasoning"
                            )

                    data_input.append(filter_non_utf8(input_text))
                    data_label.append(filter_non_utf8(label))

        self.logger.info(
            f"Built retrieval dataset with {len(data_input)} examples using method: {method}, max step: {step}"
        )
        return data_input, data_label

    def _generate_cache_key(
        self, experience_trajectory_path, rag_indexing_method, sentence_encoder_model
    ):
        """Generate a human-readable cache key."""
        trajectory_filename = os.path.basename(experience_trajectory_path)
        if trajectory_filename.endswith(".jsonl"):
            trajectory_filename = trajectory_filename[:-6]

        method, step = rag_indexing_method
        indexing_str = f"{method}-{step}"

        model_name = (
            sentence_encoder_model.split("/")[-1]
            if "/" in sentence_encoder_model
            else sentence_encoder_model
        )

        def sanitize_for_filename(s):
            return re.sub(r"[^\w\-.]", "_", s)

        trajectory_clean = sanitize_for_filename(trajectory_filename)
        indexing_clean = sanitize_for_filename(indexing_str)
        model_clean = sanitize_for_filename(model_name)

        cache_key = f"{trajectory_clean}_{indexing_clean}_{model_clean}"
        return cache_key

    def build_index(
        self,
        index_key: str,
        experience_trajectory_path: str,
        rag_indexing_method: str,
        sentence_encoder_model: str,
        rag_indexing_batch_size: int = 16,
        use_cache: bool = True,
    ) -> bool:
        """Build a retrieval index."""
        with self.index_lock:
            try:
                # Check if index already exists (double-check pattern)
                if index_key in self.indexes:
                    self.logger.info(
                        f"Index '{index_key}' already exists, skipping build"
                    )
                    return True

                self.logger.info(f"Building index '{index_key}'...")

                # Update encoder if a different model is requested
                if sentence_encoder_model != self.sentence_encoder_model:
                    self.logger.info(
                        f"Switching to encoder model: {sentence_encoder_model}"
                    )
                    self.sentence_encoder_model = sentence_encoder_model
                    self.encoder = SentenceEncoder(model_name=sentence_encoder_model)

                # Parse indexing method
                parsed_method = self.parse_indexing_method(rag_indexing_method)

                # Load experience trajectories
                experience_trajectories = self.load_experience_trajectory_from_file(
                    experience_trajectory_path
                )

                # Build retrieval dataset
                data_input, data_label = self.build_retrieval_dataset(
                    experience_trajectories, parsed_method
                )

                if not data_input:
                    self.logger.warning(f"No data found for index '{index_key}'")
                    return False

                # Compute or load embeddings
                input_representations = None

                if use_cache and self.cache_manager:
                    cache_key = self._generate_cache_key(
                        experience_trajectory_path,
                        parsed_method,
                        sentence_encoder_model,
                    )

                    def compute_embeddings(data_input):
                        """Callback function to compute embeddings."""
                        return self.encoder.encode_sentence(
                            data_input, batch_size=rag_indexing_batch_size
                        )

                    data_input, input_representations = (
                        self.cache_manager.load_or_create_cache(
                            cache_key=cache_key,
                            indexing_method=parsed_method,
                            encoder_model=sentence_encoder_model,
                            data_input=data_input,
                            compute_callback=compute_embeddings,
                        )
                    )
                else:
                    self.logger.info("Computing input representations...")
                    input_representations = self.encoder.encode_sentence(
                        data_input, batch_size=rag_indexing_batch_size
                    )

                # Build index
                encoding_dim = input_representations.shape[1]
                retriever = FaissRetriever(encoding_dim)
                retriever.add(input_representations)

                # Store index
                self.indexes[index_key] = {
                    "retriever": retriever,
                    "data_input": data_input,
                    "data_label": data_label,
                }

                self.logger.info(
                    f"Built index '{index_key}' with {len(data_input)} examples, embedding dim: {encoding_dim}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Error building index '{index_key}': {str(e)}")
                return False

    def retrieve(
        self, index_key: str, query_text: str, num_retrievals: int = 1
    ) -> List[str]:
        """Retrieve relevant examples from the specified index."""
        if index_key not in self.indexes:
            raise ValueError(f"Index '{index_key}' not found")

        index_data = self.indexes[index_key]
        retriever = index_data["retriever"]
        data_label = index_data["data_label"]

        if retriever is None or num_retrievals <= 0:
            return []

        # Check query length to prevent potential memory issues
        # Most sentence transformers have token limits around 512-8192 tokens
        # Roughly estimate ~4 chars per token as a safety check
        max_query_chars = 32000  # Conservative limit for ~8k tokens
        if len(query_text) > max_query_chars:
            self.logger.warning(
                f"Query text too long ({len(query_text)} chars > {max_query_chars}), "
                f"truncating to prevent encoding issues"
            )
            query_text = query_text[:max_query_chars]

        try:
            # Encode the query - this can fail due to GPU memory issues or long queries
            query_representation = self.encoder.encode_sentence(
                [query_text], batch_size=1
            )[0]
        except Exception as e:
            # Handle various encoding errors including GPU memory issues
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["cuda", "memory", "gpu", "out of memory", "oom"]
            ):
                self.logger.warning(f"GPU memory error during query encoding: {e}")
            elif "token" in error_msg and (
                "limit" in error_msg or "length" in error_msg or "maximum" in error_msg
            ):
                self.logger.warning(f"Query too long for encoding model: {e}")
            else:
                self.logger.warning(f"Error encoding query text: {e}")

            # Return empty list when encoding fails
            return []

        try:
            # Retrieve similar examples
            distances, indices = retriever.retrieve(
                np.array([query_representation]), topk=num_retrievals
            )

            # Extract the examples
            relevant_examples = []
            for i, idx in enumerate(indices[0]):
                if idx < len(data_label):
                    relevant_examples.append(data_label[idx])

            return relevant_examples

        except Exception as e:
            self.logger.warning(f"Error during retrieval: {e}")
            return []


class RetrievalService:
    """Retrieval service that can be shared across multiple processes."""

    def __init__(self, config: dict, port: int = 8766, host: str = "localhost"):
        self.config = config
        self.port = port
        self.host = host
        self.retrieval_manager = None
        self.server = None
        self.server_thread = None
        self.logger = DebugGymLogger(__name__)

    def start_service(self):
        """Start the retrieval service."""
        self.logger.info("Initializing retrieval manager...")
        self.retrieval_manager = RetrievalManager(self.config)

        # Create a handler class with the retrieval manager
        def handler_factory(*args, **kwargs):
            return RetrievalServiceHandler(self.retrieval_manager, *args, **kwargs)

        self.server = ThreadedHTTPServer((self.host, self.port), handler_factory)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        self.logger.info(f"Retrieval service started on {self.host}:{self.port}")

    def stop_service(self):
        """Stop the retrieval service."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
        self.logger.info("Retrieval service stopped")


class RetrievalServiceClient:
    """Client for interacting with the retrieval service."""

    def __init__(self, host: str = "localhost", port: int = 8766, timeout: int = 120):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.logger = DebugGymLogger(__name__)

    def is_service_available(self) -> bool:
        """Check if the retrieval service is available."""
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

    def check_index(self, index_key: str) -> bool:
        """Check if an index exists on the retrieval service."""
        data = {"index_key": index_key}

        try:
            response = requests.post(
                f"{self.base_url}/check_index",
                json=data,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return False

            result = response.json()
            return result.get("exists", False)

        except Exception as e:
            self.logger.error(f"Error checking index: {e}")
            return False

    def build_index(
        self,
        index_key: str,
        experience_trajectory_path: str,
        rag_indexing_method: str,
        sentence_encoder_model: str,
        rag_indexing_batch_size: int = 16,
        use_cache: bool = True,
    ) -> bool:
        """Build an index on the retrieval service."""
        data = {
            "index_key": index_key,
            "experience_trajectory_path": experience_trajectory_path,
            "rag_indexing_method": rag_indexing_method,
            "sentence_encoder_model": sentence_encoder_model,
            "rag_indexing_batch_size": rag_indexing_batch_size,
            "use_cache": use_cache,
        }

        try:
            response = requests.post(
                f"{self.base_url}/build_index",
                json=data,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Retrieval service error: {response.status_code} - {response.text}"
                )

            result = response.json()
            return result.get("success", False)

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to retrieval service: {e}")
            raise RuntimeError(f"Failed to connect to retrieval service: {e}")
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout error from retrieval service: {e}")
            raise RuntimeError(f"Retrieval service timeout: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error from retrieval service: {e}")
            raise

    def retrieve(
        self, index_key: str, query_text: str, num_retrievals: int = 1
    ) -> List[str]:
        """Retrieve relevant examples from the retrieval service."""
        data = {
            "index_key": index_key,
            "query_text": query_text,
            "num_retrievals": num_retrievals,
        }

        try:
            response = requests.post(
                f"{self.base_url}/retrieve",
                json=data,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                self.logger.warning(
                    f"Retrieval service error: {response.status_code} - {response.text}"
                )
                return []

            result = response.json()
            return result.get("relevant_examples", [])

        except requests.exceptions.ConnectionError as e:
            self.logger.warning(f"Connection error to retrieval service: {e}")
            return []
        except requests.exceptions.Timeout as e:
            self.logger.warning(f"Timeout error from retrieval service: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"Unexpected error from retrieval service: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error from retrieval service: {e}")
            raise

    def list_indexes(self) -> List[str]:
        """List available indexes."""
        try:
            response = requests.get(f"{self.base_url}/indexes", timeout=10)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Retrieval service error: {response.status_code} - {response.text}"
                )
            result = response.json()
            return result.get("indexes", [])
        except Exception as e:
            self.logger.error(f"Error listing indexes: {e}")
            return []


def start_retrieval_service_standalone(
    config: dict, port: int = 8766, host: str = "localhost"
):
    """Standalone function to start the retrieval service."""
    service = RetrievalService(config, port, host)

    try:
        service.start_service()
        print(f"Retrieval service running on {host}:{port}")
        print("Press Ctrl+C to stop the service")

        # Keep the service running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down retrieval service...")
        service.stop_service()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start retrieval service")
    parser.add_argument("--port", type=int, default=8766, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    start_retrieval_service_standalone(config, args.port, args.host)
