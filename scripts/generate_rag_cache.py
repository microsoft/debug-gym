#!/usr/bin/env python3
"""
Script to pre-generate input-representation caches for RAG agents.
This allows you to prepare caches ahead of time before running multiple agents in parallel.
Note: This script now works with the integrated retrieval service architecture.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the debug_gym directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_gym.agents.retrieval_service import RetrievalManager
from debug_gym.logger import DebugGymLogger


class CacheGenerator:
    """Generates input-representation caches using the retrieval service components."""

    def __init__(
        self,
        experience_trajectory_path: str,
        rag_indexing_method: str,
        sentence_encoder_model: str,
        cache_dir: str = ".rag_cache",
        max_examples: int = None,
        batch_size: int = 16,
    ):
        self.logger = DebugGymLogger("CacheGenerator")

        # Create config for the retrieval manager
        config = {
            "rag_cache_dir": cache_dir,
            "rag_use_cache": True,
            "sentence_encoder_model": sentence_encoder_model,
        }

        self.experience_trajectory_path = experience_trajectory_path
        self.rag_indexing_method = rag_indexing_method
        self.sentence_encoder_model = sentence_encoder_model
        self.max_examples = max_examples
        self.batch_size = batch_size

        self.logger.info("Initializing retrieval manager for cache generation...")
        self.retrieval_manager = RetrievalManager(config)

    def generate_cache(self):
        """Generate and save the input-representation cache."""
        # First, we need to load the experience trajectory data
        experience_data = self._load_experience_data()

        if not experience_data:
            self.logger.error(
                "No data to process. Check your experience trajectory file and indexing method."
            )
            return False

        self.logger.info(f"Processing {len(experience_data)} examples")

        # Use retrieval manager to build index (this will cache embeddings)
        index_name = f"cache_gen_{self.rag_indexing_method}_{self.sentence_encoder_model.replace('/', '_')}"

        self.logger.info(f"Building index: {index_name}")
        success = self.retrieval_manager.build_index(
            index_name, experience_data, self.rag_indexing_method
        )

        if success:
            self.logger.info("Cache generation completed successfully!")
            return True
        else:
            self.logger.error("Cache generation failed!")
            return False

    def _load_experience_data(self):
        """Load experience trajectory data."""
        try:
            import json

            self.logger.info(
                f"Loading experience data from: {self.experience_trajectory_path}"
            )

            with open(self.experience_trajectory_path, "r") as f:
                data = json.load(f)

            # Extract input data based on indexing method
            if self.rag_indexing_method == "history":
                # For history indexing, we want the complete problem-solving sequences
                experience_data = []
                for episode in data:
                    if "history" in episode:
                        experience_data.append(str(episode["history"]))
                    elif "trajectory" in episode:
                        experience_data.append(str(episode["trajectory"]))
                    else:
                        # Fallback: use the entire episode as a string
                        experience_data.append(str(episode))

            elif self.rag_indexing_method == "action":
                # For action indexing, extract individual actions
                experience_data = []
                for episode in data:
                    if "history" in episode:
                        for step in episode["history"]:
                            if "action" in step:
                                experience_data.append(str(step["action"]))
                    elif "trajectory" in episode:
                        for step in episode["trajectory"]:
                            if "action" in step:
                                experience_data.append(str(step["action"]))

            else:
                self.logger.warning(
                    f"Unknown indexing method: {self.rag_indexing_method}, using full episodes"
                )
                experience_data = [str(episode) for episode in data]

            # Apply max_examples limit if specified
            if self.max_examples and len(experience_data) > self.max_examples:
                self.logger.info(f"Limiting to first {self.max_examples} examples")
                experience_data = experience_data[: self.max_examples]

            self.logger.info(f"Loaded {len(experience_data)} data points")
            return experience_data

        except Exception as e:
            self.logger.error(f"Failed to load experience data: {e}")
            return []


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
    print("=" * 80)

    try:
        # Create cache generator
        generator = CacheGenerator(
            experience_trajectory_path=args.experience_trajectory_path,
            rag_indexing_method=args.rag_indexing_method,
            sentence_encoder_model=args.sentence_encoder_model,
            cache_dir=args.cache_dir,
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
