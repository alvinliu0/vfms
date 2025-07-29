#!/usr/bin/env python3
"""
Local client for Wan2.1 text-to-video generation endpoint (14B model).

This script provides a convenient way to:
1. Load API token from a local file
2. Send text-to-video generation requests to the endpoint
3. Download and save generated videos locally
4. Handle authentication and error responses

Usage:
    python wan2.1_14B_client.py --prompt "A beautiful sunset" --output-dir ./results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gradio_client
import httpx
import requests


class Wan2_1Client:
    """
    Client for interacting with Wan2.1 text-to-video generation endpoint.
    Supports both Gradio client and direct HTTP requests.
    """

    def __init__(self, endpoint_url: str, api_token: str = "", 
                 output_dir: str = "./results"):
        """
        Initialize the Wan2.1 client.

        Args:
            endpoint_url: URL of the Gradio endpoint
            api_token: API token for authentication (optional)
            output_dir: Directory to save generated videos
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_token = api_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up headers for API requests
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"

        # Try to initialize Gradio client
        self.client = None
        self.api_name = "/predict"  # Default API endpoint
        
        # Initialize Gradio client with authentication
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

        client_kwargs = {
            "headers": self.headers if self.headers else None,
            "httpx_kwargs": {"timeout": httpx.Timeout(7200.0)},  # 2 hours timeout for generation
            "ssl_verify": True,  # set to False if you have self-signed certs
        }
        
        try:
            print(f"🔄 Initializing Gradio client with headers: {self.headers}")
            self.client = gradio_client.Client(self.endpoint_url, **client_kwargs)
            print(f"✅ Gradio client initialized with authentication for: {self.endpoint_url}")
        except Exception as e:
            print(f"⚠️ Gradio client initialization failed: {e}")
            print("💡 Please ensure the server is running and accessible")

        # Validate endpoint URL
        self._check_endpoint_url()

    def _check_endpoint_url(self):
        """Validate and clean up the endpoint URL."""
        if not self.endpoint_url.startswith(("http://", "https://")):
            self.endpoint_url = f"http://{self.endpoint_url}"
            print(f"🔧 Added protocol to endpoint URL: {self.endpoint_url}")

        # Remove trailing slash if present
        self.endpoint_url = self.endpoint_url.rstrip("/")

    def load_api_token(self, token_file: str = "api_token.txt") -> bool:
        """
        Load API token from a local file.

        Args:
            token_file: Path to the file containing the API token

        Returns:
            True if token loaded successfully, False otherwise
        """
        try:
            token_path = Path(token_file)
            if not token_path.exists():
                print(f"⚠️ Token file not found: {token_path}")
                return False

            with open(token_path, "r") as f:
                token = f.read().strip()
                if not token:
                    print("⚠️ Token file is empty")
                    return False

            self.api_token = token
            self.headers["Authorization"] = f"Bearer {token}"
            print(f"✅ API token loaded from {token_path}")
            return True

        except Exception as e:
            print(f"❌ Error loading API token: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test connection to the Gradio endpoint.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.client:
                print("❌ Gradio client not initialized")
                return False

            # Try to get the API info
            api_info = self.client.view_api()
            print(f"✅ Connection successful! API info: {api_info}")
            return True

        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def generate_video(
        self, prompt: str, negative_prompt: str = "", parameters: Optional[Dict[str, Any]] = None, save_metadata: bool = True
    ) -> Optional[str]:
        """
        Generate a video using the Wan2.1 endpoint.

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain elements
            parameters: Optional generation parameters
            save_metadata: Whether to save metadata files

        Returns:
            Path to the generated video file, or None if generation failed
        """
        try:
            if not self.client:
                print("❌ Gradio client not initialized")
                return None

            if not prompt:
                print("❌ No prompt provided")
                return None

            # Prepare parameters
            if parameters is None:
                parameters = {}

            # Map parameters to the expected format
            mapped_params = self._map_parameters(parameters)

            # Create generation parameters JSON
            generation_params = {
                "frame_num": mapped_params.get("frame_num", 16),
                "sample_steps": mapped_params.get("sample_steps", 50),
                "sample_guide_scale": mapped_params.get("sample_guide_scale", 7.5),
                "base_seed": mapped_params.get("base_seed", 42),
                "width": mapped_params.get("width", 832),
                "height": mapped_params.get("height", 480),
                "use_prompt_extend": mapped_params.get("use_prompt_extend", False),
            }

            # Convert to JSON string
            params_json = json.dumps(generation_params, indent=2)

            print(f"🎬 Generating video with prompt: {prompt}")
            print(f"📝 Negative prompt: {negative_prompt}")
            print(f"⚙️ Parameters: {generation_params}")

            # Call the Gradio endpoint
            result = self.client.predict(
                prompt,  # str in 'Prompt' Textbox component
                negative_prompt,  # str in 'Negative Prompt' Textbox component
                params_json,  # str in 'Generation Parameters JSON' Textbox component
                self.api_token,  # str in 'API Token (Optional)' Textbox component
                api_name="/predict",
            )

            print(f"📊 Generation result: {result}")

            # Parse the result
            if isinstance(result, tuple) and len(result) >= 2:
                video_path = result[0]
                status_message = result[1]

                if video_path and os.path.exists(video_path):
                    # Download the video file
                    local_filename = f"wan2.1_14B_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    local_path = self.output_dir / local_filename

                    if self._download_file(video_path, local_path):
                        print(f"✅ Video generated and saved to: {local_path}")

                        # Save metadata if requested
                        if save_metadata:
                            metadata = {
                                "prompt": prompt,
                                "negative_prompt": negative_prompt,
                                "parameters": generation_params,
                                "generated_at": datetime.now().isoformat(),
                                "source_video_path": video_path,
                                "local_video_path": str(local_path),
                                "model": "wan2.1_t2v_14B_singleGPU",
                            }

                            metadata_path = local_path.with_suffix(".json")
                            with open(metadata_path, "w") as f:
                                json.dump(metadata, f, indent=2)
                            print(f"📄 Metadata saved to: {metadata_path}")

                        return str(local_path)
                    else:
                        print("❌ Failed to download video file")
                        return None
                else:
                    print(f"❌ Video generation failed: {status_message}")
                    return None
            else:
                print(f"❌ Unexpected result format: {result}")
                return None

        except Exception as e:
            print(f"❌ Error during video generation: {e}")
            return None

    def _map_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map parameter names to the expected format.

        Args:
            parameters: Input parameters

        Returns:
            Mapped parameters
        """
        mapping = {
            "num_frames": "frame_num",
            "num_inference_steps": "sample_steps",
            "guidance_scale": "sample_guide_scale",
            "seed": "base_seed",
        }

        mapped = {}
        for key, value in parameters.items():
            if key in mapping:
                mapped[mapping[key]] = value
            else:
                mapped[key] = value

        return mapped

    def _download_file(self, url: str, local_path: Path) -> bool:
        """
        Download a file from URL to local path.

        Args:
            url: Source URL
            local_path: Local destination path

        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def batch_generate(self, prompts: list, negative_prompts: Optional[list] = None, parameters: Optional[Dict[str, Any]] = None) -> list:
        """
        Generate multiple videos in batch.

        Args:
            prompts: List of text prompts
            negative_prompts: Optional list of negative prompts
            parameters: Optional generation parameters

        Returns:
            List of generated video paths
        """
        results = []
        total = len(prompts)

        for i, prompt in enumerate(prompts, 1):
            print(f"🎬 Generating video {i}/{total}: {prompt}")

            negative_prompt = ""
            if negative_prompts and i <= len(negative_prompts):
                negative_prompt = negative_prompts[i - 1]

            result = self.generate_video(prompt, negative_prompt, parameters)
            results.append(result)

            if result:
                print(f"✅ Video {i} generated successfully")
            else:
                print(f"❌ Video {i} generation failed")

            # Add delay between generations to avoid overwhelming the server
            if i < total:
                print("⏳ Waiting 5 seconds before next generation...")
                time.sleep(5)

        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Wan2.1 14B Text-to-Video Generation Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with command-line token
  python wan2.1_t2v_14B_singleGPU_client.py \\
    --token "your_api_token" \\
    --prompt "A beautiful sunset over the ocean"

  # With custom parameters
  python wan2.1_t2v_14B_singleGPU_client.py \\
    --token "your_api_token" \\
    --prompt "A beautiful sunset over the ocean" \\
    --negative-prompt "blurry, low quality" \\
    --parameters '{"frame_num": 16, "sample_steps": 50}'

  # Test connection only
  python wan2.1_t2v_14B_singleGPU_client.py \\
    --token "your_api_token" \\
    --test-connection
        """,
    )

    parser.add_argument(
        "--token",
        type=str,
        help="API token for authentication (recommended)",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default="api_token.txt",
        help="File containing API token (fallback)",
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
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt to avoid elements",
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

    # Check if prompt is provided
    if not args.prompt:
        print("❌ No prompt provided")
        print("💡 Use --prompt to specify a text prompt")
        sys.exit(1)

    # Parse parameters if provided
    parameters = None
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in parameters: {e}")
            sys.exit(1)

    # Generate video
    result = client.generate_video(args.prompt, args.negative_prompt, parameters)

    if result:
        print(f"✅ Video generated successfully: {result}")
        sys.exit(0)
    else:
        print("❌ Video generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 