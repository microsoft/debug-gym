#!/usr/bin/env python3
"""
Script to pre-generate input-representation caches for RAG agents.
This allows you to prepare caches ahead of time before running multiple agents in parallel.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the debug_gym directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_gym.agents.rag_agent import RAGAgent
from debug_gym.logger import DebugGymLogger


class CacheGenerator:
    """Generates input-representation caches for RAG agents by reusing RAGAgent code."""

    def __init__(
        self,
        experience_trajectory_path: str,
        rag_indexing_method: str,
        sentence_encoder_model: str,
        cache_dir: str = ".rag_cache",
        use_encoding_service: bool = False,
        encoding_service_host: str = "localhost",
        encoding_service_port: int = 8765,
        max_examples: int = None,
        batch_size: int = 16,
    ):
        self.logger = DebugGymLogger("CacheGenerator")

        # Create a minimal config for the RAG agent
        config = {
            "experience_trajectory_path": experience_trajectory_path,
            "rag_indexing_method": rag_indexing_method,
            "rag_indexing_batch_size": batch_size,
            "sentence_encoder_model": sentence_encoder_model,
            "rag_cache_dir": cache_dir,
            "rag_use_cache": True,
            "rag_use_encoding_service": use_encoding_service,
            "rag_encoding_service_host": encoding_service_host,
            "rag_encoding_service_port": encoding_service_port,
            # Required by base agent
            "output_path": "/tmp/cache_generator_output",
            "random_seed": 42,
            "memory_size": 100,
        }

        self.max_examples = max_examples
        self.batch_size = batch_size

        # Create a mock environment (RAGAgent needs it but we won't use it)
        class MockEnv:
            pass

        self.logger.info("Initializing RAG agent for cache generation...")

        # Initialize the RAG agent (this will load data and build the dataset)
        try:
            self.rag_agent = RAGAgent(config=config, env=MockEnv(), logger=self.logger)
        except Exception as e:
            # If initialization fails, we might need to handle max_examples differently
            self.logger.warning(f"Initial RAG agent creation failed: {e}")
            self.logger.info("Trying with manual data loading...")

            # Create agent but override the data loading
            self.rag_agent = self._create_agent_with_custom_loading(config, MockEnv())

    def _create_agent_with_custom_loading(self, config, env):
        """Create RAG agent with custom data loading for max_examples support."""
        # Create agent without auto-initialization
        agent = object.__new__(RAGAgent)

        # Initialize parent classes manually
        from debug_gym.agents.debug_agent import DebugAgent

        DebugAgent.__init__(agent, config, env, None, self.logger)

        # Set RAG-specific attributes
        agent.rag_num_retrievals = config.get("rag_num_retrievals", 1)
        agent.rag_indexing_method = agent.parse_indexing_method(
            config.get("rag_indexing_method")
        )
        agent.sentence_encoder_model = config.get(
            "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
        )
        agent.cache_dir = config.get("rag_cache_dir", ".rag_cache")
        agent.use_cache = config.get("rag_use_cache", True)
        agent.use_encoding_service = config.get("rag_use_encoding_service", True)
        agent.encoding_service_host = config.get(
            "rag_encoding_service_host", "localhost"
        )
        agent.encoding_service_port = config.get("rag_encoding_service_port", 8765)

        # Initialize shared cache manager
        from debug_gym.agents.shared_cache import get_shared_cache_manager

        if agent.use_cache:
            agent.cache_manager = get_shared_cache_manager(agent.cache_dir)
        else:
            agent.cache_manager = None

        agent.experience_trajectory_path = config.get("experience_trajectory_path")

        # Load experience trajectories with max_examples support
        agent.load_experience_trajectory_from_file(
            agent.experience_trajectory_path, self.max_examples
        )

        # Build retrieval dataset
        agent.build_retrieval_dataset()

        # Initialize encoder
        agent._initialize_encoder()

        return agent

    def generate_cache(self):
        """Generate and save the input-representation cache."""
        if not hasattr(self.rag_agent, "data_input") or not self.rag_agent.data_input:
            self.logger.error(
                "No data to process. Check your experience trajectory file and indexing method."
            )
            return False

        cache_key = self.rag_agent._generate_cache_key()
        self.logger.info(f"Generating cache with key: {cache_key}")
        self.logger.info(f"Processing {len(self.rag_agent.data_input)} examples")

        def compute_embeddings(data_input):
            """Callback function to compute embeddings."""
            self.logger.info(
                f"Computing embeddings for {len(data_input)} inputs with batch_size={self.batch_size}"
            )
            start_time = time.time()
            embeddings = self.rag_agent.encoder.encode_sentence(
                data_input, batch_size=self.batch_size
            )
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Embedding computation completed in {elapsed_time:.2f} seconds"
            )
            return embeddings

        try:
            # Use the RAG agent's cache manager to generate and save cache
            data_input, input_representations = (
                self.rag_agent.cache_manager.load_or_create_cache(
                    cache_key=cache_key,
                    indexing_method=self.rag_agent.rag_indexing_method,
                    encoder_model=self.rag_agent.sentence_encoder_model,
                    data_input=self.rag_agent.data_input,
                    compute_callback=compute_embeddings,
                )
            )

            self.logger.info(
                f"Successfully generated cache with {len(data_input)} examples"
            )
            self.logger.info(f"Embedding dimensions: {input_representations.shape}")
            self.logger.info(f"Cache saved to: {self.rag_agent.cache_dir}")

            # Print cache info
            cache_info = self.rag_agent.cache_manager.get_cache_info()
            self.logger.info(
                f"Cache memory usage: {cache_info['memory_usage_mb']:.2f} MB"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to generate cache: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate input-representation caches for RAG agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "experience_trajectory_path",
        help="Path to the experience trajectory JSONL file",
    )
    parser.add_argument(
        "rag_indexing_method",
        help="RAG indexing method (e.g., 'tool_name-1', 'tool_call-2', 'observation-3')",
    )
    parser.add_argument(
        "sentence_encoder_model",
        help="Sentence encoder model name (e.g., 'Qwen/Qwen3-Embedding-0.6B')",
    )

    # Optional arguments
    parser.add_argument(
        "--cache-dir",
        default=".rag_cache",
        help="Directory to store the generated cache",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for encoding"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of trajectory examples to process",
    )
    parser.add_argument(
        "--use-encoding-service",
        action="store_true",
        help="Use encoding service instead of local encoder",
    )
    parser.add_argument(
        "--encoding-service-host", default="localhost", help="Encoding service host"
    )
    parser.add_argument(
        "--encoding-service-port", type=int, default=8765, help="Encoding service port"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.experience_trajectory_path):
        print(
            f"Error: Experience trajectory file not found: {args.experience_trajectory_path}"
        )
        return 1

    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 80)
    print("RAG Cache Generator")
    print("=" * 80)
    print(f"Experience trajectory: {args.experience_trajectory_path}")
    print(f"Indexing method: {args.rag_indexing_method}")
    print(f"Encoder model: {args.sentence_encoder_model}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")
    if args.use_encoding_service:
        print(
            f"Encoding service: {args.encoding_service_host}:{args.encoding_service_port}"
        )
    print("=" * 80)

    try:
        # Create cache generator
        generator = CacheGenerator(
            experience_trajectory_path=args.experience_trajectory_path,
            rag_indexing_method=args.rag_indexing_method,
            sentence_encoder_model=args.sentence_encoder_model,
            cache_dir=args.cache_dir,
            use_encoding_service=args.use_encoding_service,
            encoding_service_host=args.encoding_service_host,
            encoding_service_port=args.encoding_service_port,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
        )

        # Generate cache
        success = generator.generate_cache()

        if success:
            print("\n🎉 Cache generation completed successfully!")
            return 0
        else:
            print("\n❌ Cache generation failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
