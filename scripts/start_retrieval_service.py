#!/usr/bin/env python3
"""
Script to start the retrieval service with hang detection support.
"""

import argparse

import yaml

from debug_gym.agents.retrieval_service import start_retrieval_service_standalone


def main():
    parser = argparse.ArgumentParser(
        description="Start retrieval service with hang detection"
    )
    parser.add_argument("--port", type=int, default=8766, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument(
        "--no-hang-detection",
        action="store_true",
        help="Disable hang detection and auto-restart",
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            config = config.get("rag_agent", {})

    enable_hang_detection = not args.no_hang_detection

    print(f"Starting retrieval service on {args.host}:{args.port}")
    if enable_hang_detection:
        print(
            "Hang detection enabled - service will auto-restart if it becomes unresponsive"
        )
    else:
        print("Hang detection disabled")

    start_retrieval_service_standalone(
        config, args.port, args.host, enable_hang_detection=enable_hang_detection
    )


if __name__ == "__main__":
    main()
