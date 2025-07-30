#!/usr/bin/env python3
"""
Script to start the retrieval service.
"""

import argparse

import yaml

from debug_gym.agents.retrieval_service import start_retrieval_service_standalone


def main():
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
            config = config.get("rag_agent", {})

    start_retrieval_service_standalone(config, args.port, args.host)


if __name__ == "__main__":
    main()
