#!/usr/bin/env python3
"""
Script to start the encoding service for RAG agents.
This should be run before starting multiple RAG agents for parallel execution.
"""

import argparse
import os
import sys

# Add the debug_gym directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_gym.agents.encoding_service import start_encoding_service_standalone


def main():
    parser = argparse.ArgumentParser(
        description="Start sentence encoding service for RAG agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Model name for sentence encoding",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to run the service on"
    )
    parser.add_argument(
        "--host", default="localhost", help="Host to bind the service to"
    )

    args = parser.parse_args()

    print(f"Starting encoding service with model: {args.model}")
    print(f"Service will be available at http://{args.host}:{args.port}")
    print("Make sure to configure your RAG agents with:")
    print(f"  rag_use_encoding_service: true")
    print(f"  rag_encoding_service_host: {args.host}")
    print(f"  rag_encoding_service_port: {args.port}")
    print()

    start_encoding_service_standalone(args.model, args.port, args.host)


if __name__ == "__main__":
    main()
