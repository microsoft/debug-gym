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
    parser.add_argument(
        "--hang-timeout",
        type=int,
        help="Timeout in seconds before considering service hung (default: 300)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        help="Interval in seconds between hang detection checks (default: 150)",
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        help="Delay in seconds before restarting hung service (default: 2)",
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            config = config.get("rag_agent", {})

    # Override config with command line arguments
    if args.hang_timeout is not None:
        config["hang_detection_timeout"] = args.hang_timeout
    if args.check_interval is not None:
        config["watchdog_check_interval"] = args.check_interval
    if args.restart_delay is not None:
        config["restart_delay"] = args.restart_delay

    enable_hang_detection = not args.no_hang_detection

    print(f"Starting retrieval service on {args.host}:{args.port}")
    if enable_hang_detection:
        hang_timeout = config.get("hang_detection_timeout", 300)
        check_interval = config.get("watchdog_check_interval", 150)
        restart_delay = config.get("restart_delay", 2)
        print(
            f"Hang detection enabled - service will auto-restart if unresponsive for {hang_timeout}s "
            f"(checks every {check_interval}s, restart delay: {restart_delay}s)"
        )
    else:
        print("Hang detection disabled")

    start_retrieval_service_standalone(
        config, args.port, args.host, enable_hang_detection=enable_hang_detection
    )


if __name__ == "__main__":
    main()
