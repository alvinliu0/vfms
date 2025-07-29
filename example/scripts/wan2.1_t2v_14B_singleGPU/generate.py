#!/usr/bin/env python3
"""
Example script for Wan2.1 14B text-to-video generation.

This script demonstrates how to use the Wan2.1 14B model for text-to-video generation.
It can be used as a standalone script or as a reference for integration.

Usage:
    python generate.py --prompt "A beautiful sunset over the ocean"
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gradio_client.wan2_1_t2v_14B_singleGPU_client import Wan2_1Client


def load_prompts_from_file(file_path: str) -> list:
    """
    Load prompts from a text file.

    Args:
        file_path: Path to the text file containing prompts

    Returns:
        List of prompts
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    except Exception as e:
        print(f"❌ Error loading prompts from {file_path}: {e}")
        return []


def load_negative_prompts_from_file(file_path: str) -> list:
    """
    Load negative prompts from a text file.

    Args:
        file_path: Path to the text file containing negative prompts

    Returns:
        List of negative prompts
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            negative_prompts = [line.strip() for line in f if line.strip()]
        return negative_prompts
    except Exception as e:
        print(f"❌ Error loading negative prompts from {file_path}: {e}")
        return []


def main():
    """Main function for the generation script."""
    parser = argparse.ArgumentParser(
        description="Wan2.1 14B Text-to-Video Generation Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single video
  python generate.py \\
    --token "your_api_token" \\
    --prompt "A beautiful sunset over the ocean"

  # Generate multiple videos from files
  python generate.py \\
    --token "your_api_token" \\
    --prompts-file prompts_example.txt \\
    --negative-prompts-file negative_prompts_example.txt

  # With custom parameters
  python generate.py \\
    --token "your_api_token" \\
    --prompt "A beautiful sunset over the ocean" \\
    --parameters '{"frame_num": 16, "sample_steps": 50}'
        """,
    )

    parser.add_argument(
        "--token",
        type=str,
        help="API token for authentication",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default="api_token.txt",
        help="File containing API token",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="Server endpoint URL",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single text prompt for video generation",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="File containing multiple prompts (one per line)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Single negative prompt",
    )
    parser.add_argument(
        "--negative-prompts-file",
        type=str,
        help="File containing multiple negative prompts (one per line)",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        help="JSON string of generation parameters",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test connection only",
    )

    args = parser.parse_args()

    # Initialize client
    client = Wan2_1Client(args.endpoint, "", args.output_dir)

    # Load API token
    api_token = args.token
    if not api_token:
        if client.load_api_token(args.token_file):
            api_token = client.api_token
        else:
            print("❌ No API token provided and token file not found")
            print("💡 Use --token to provide API token or create api_token.txt file")
            sys.exit(1)

    # Update client with API token
    client.api_token = api_token
    client.headers["Authorization"] = f"Bearer {api_token}"

    # Test connection if requested
    if args.test_connection:
        if client.test_connection():
            print("✅ Connection test successful!")
            sys.exit(0)
        else:
            print("❌ Connection test failed!")
            sys.exit(1)

    # Parse parameters if provided
    parameters = None
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in parameters: {e}")
            sys.exit(1)

    # Handle single prompt
    if args.prompt:
        print(f"🎬 Generating single video with prompt: {args.prompt}")
        result = client.generate_video(args.prompt, args.negative_prompt, parameters)
        
        if result:
            print(f"✅ Video generated successfully: {result}")
            sys.exit(0)
        else:
            print("❌ Video generation failed")
            sys.exit(1)

    # Handle multiple prompts from file
    elif args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        if not prompts:
            print("❌ No prompts loaded from file")
            sys.exit(1)

        negative_prompts = []
        if args.negative_prompts_file:
            negative_prompts = load_negative_prompts_from_file(args.negative_prompts_file)

        print(f"🎬 Generating {len(prompts)} videos from prompts file")
        results = client.batch_generate(prompts, negative_prompts, parameters)

        successful = [r for r in results if r]
        print(f"✅ Successfully generated {len(successful)}/{len(prompts)} videos")
        
        if successful:
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        print("❌ No prompt provided")
        print("💡 Use --prompt for single generation or --prompts-file for batch")
        sys.exit(1)


if __name__ == "__main__":
    main() 