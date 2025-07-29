#!/usr/bin/env python3
"""
Local client for Wan2.1 text-to-video generation endpoint.

This script provides a convenient way to:
1. Load API token from a local file
2. Send text-to-video generation requests to the endpoint
3. Download and save generated videos locally
4. Handle authentication and error responses

Usage:
    python wan2.1_client.py --prompt "A beautiful sunset" --output-dir ./results
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
                print(f"❌ API token file not found: {token_file}")
                print("Please create a file named 'api_token.txt' with your API token on a single line.")
                return False

            with open(token_path, "r") as f:
                self.api_token = f.read().strip()

            if not self.api_token:
                print("❌ API token file is empty")
                return False

            print(f"✅ API token loaded successfully: {self.api_token[:10]}...")
            return True

        except Exception as e:
            print(f"❌ Error loading API token: {e}")
            return False

    def test_connection(self) -> bool:
        """Test connection to the endpoint using Gradio client."""
        # Test basic connectivity
        try:
            print(f"🔄 Testing basic connectivity to: {self.endpoint_url}")
            response = requests.get(self.endpoint_url, timeout=10, 
                                 headers=self.headers)
            print(f"✅ Endpoint is reachable (HTTP {response.status_code})")
        except Exception as e:
            print(f"❌ Endpoint is not reachable: {e}")
            return False

        # Test Gradio client
        if self.client is None:
            print("❌ Gradio client not available")
            print("💡 Please ensure the server is running and accessible")
            return False

        try:
            print("🔄 Testing Gradio API discovery...")
            api_info = self.client.view_api(print_info=False, 
                                          return_format="dict") or {}

            if api_info:
                available_functions = list(api_info.keys())
                print(f"Available functions: {available_functions}")

                # Look for actual function names (not endpoint categories)
                actual_functions = []
                for func_name in available_functions:
                    if func_name not in ["named_endpoints", "unnamed_endpoints"]:
                        actual_functions.append(func_name)

                if actual_functions:
                    print(f"✅ Found function: {actual_functions[0]}")
                    return True
                else:
                    print("⚠️ No specific functions found")
                    return True  # Still return True for basic connectivity
            else:
                print("⚠️ No API functions discovered")
                return True  # Still return True for basic connectivity

        except Exception as e:
            print(f"❌ Gradio API discovery failed: {e}")
            print("💡 Please ensure the server exposes API endpoints")
            return False

    def generate_video(
        self, prompt: str, negative_prompt: str = "", parameters: Optional[Dict[str, Any]] = None, save_metadata: bool = True
    ) -> Optional[str]:
        """
        Generate a video using the Wan2.1 endpoint.

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain elements
            parameters: Additional generation parameters
            save_metadata: Whether to save request metadata

        Returns:
            Path to the downloaded video file, or None if generation failed
        """
        try:
            # Default parameters optimized for H100 GPU - matching official generate.py
            if parameters is None:
                parameters = {
                    "frame_num": 16,  # Standard frame count
                    "sample_steps": 50,
                    "sample_guide_scale": 7.5,
                    "base_seed": int(time.time()) % 1000000,  # Random seed
                    "width": 832,
                    "height": 480,
                    "use_prompt_extend": False,
                }

            # Create unique output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.output_dir / f"generation_{timestamp}"
            output_folder.mkdir(parents=True, exist_ok=True)

            print(f"🎬 Generating video with prompt: '{prompt}'")
            print(f"📁 Output folder: {output_folder}")

            # Prepare the request
            request_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "parameters": json.dumps(parameters),
                "api_token": self.api_token,
            }

            # Save request metadata
            if save_metadata:
                metadata_file = output_folder / "request_metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(request_data, f, indent=2)
                print(f"📄 Request metadata saved to: {metadata_file}")

            # Try using gradio_client
            try:
                if self.client is None:
                    raise RuntimeError("gradio_client is not available")

                print("⏳ Sending request via gradio_client...")
                # Map parameters to correct names
                mapped_parameters = self._map_parameters(parameters)

                # Use gradio_client for prediction
                result = self.client.predict(
                    prompt,
                    negative_prompt,
                    json.dumps(mapped_parameters),
                    self.api_token,
                )

                print("✅ Generation completed!")
                print(f"📊 Result: {result}")

            except Exception as e:
                print(f"❌ gradio_client failed: {e}")
                print("💡 Please ensure the server is running and API endpoints are exposed")
                return None

            # Parse the result
            if isinstance(result, tuple) and len(result) >= 2:
                video_path, status_message = result

                # Save status message
                status_file = output_folder / "status.txt"
                with open(status_file, "w") as f:
                    f.write(status_message)

                # Download video if it's a URL
                if video_path and isinstance(video_path, str):
                    if video_path.startswith("http"):
                        # Download the video
                        local_video_path = output_folder / "generated_video.mp4"
                        self._download_file(video_path, local_video_path)
                        print(f"🎥 Video downloaded to: {local_video_path}")
                        return str(local_video_path)
                    elif os.path.exists(video_path):
                        # Copy the video if it's a local path
                        local_video_path = output_folder / "generated_video.mp4"
                        import shutil

                        shutil.copy2(video_path, local_video_path)
                        print(f"🎥 Video copied to: {local_video_path}")
                        return str(local_video_path)
                    else:
                        print(f"⚠️ Video path not found: {video_path}")
                        return None
                else:
                    print("⚠️ No video path in result")
                    return None
            else:
                print(f"⚠️ Unexpected result format: {result}")
                return None

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None

    def _map_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map old parameter names to new ones for backward compatibility."""
        mapped_params = parameters.copy()

        # Map old parameter names to new ones
        if "num_frames" in mapped_params and "frame_num" not in mapped_params:
            mapped_params["frame_num"] = mapped_params.pop("num_frames")

        if "num_inference_steps" in mapped_params and "sample_steps" not in mapped_params:
            mapped_params["sample_steps"] = mapped_params.pop("num_inference_steps")

        if "guidance_scale" in mapped_params and "sample_guide_scale" not in mapped_params:
            mapped_params["sample_guide_scale"] = mapped_params.pop("guidance_scale")

        if "seed" in mapped_params and "base_seed" not in mapped_params:
            mapped_params["base_seed"] = mapped_params.pop("seed")

        # Remove unsupported parameters
        unsupported_params = ["fps", "batch_size", "video_save_name"]
        for param in unsupported_params:
            if param in mapped_params:
                del mapped_params[param]

        return mapped_params

    def _download_file(self, url: str, local_path: Path) -> bool:
        """Download a file from URL to local path."""
        try:
            response = requests.get(url, stream=True)
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
            negative_prompts: List of negative prompts (optional)
            parameters: Generation parameters (same for all)

        Returns:
            List of paths to generated videos
        """
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)

        results = []
        for i, (prompt, neg_prompt) in enumerate(zip(prompts, negative_prompts)):
            print(f"\n--- Generating video {i+1}/{len(prompts)} ---")
            result = self.generate_video(prompt, neg_prompt, parameters)
            results.append(result)

            # Add delay between requests
            if i < len(prompts) - 1:
                time.sleep(2)

        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Wan2.1 Text-to-Video Generation Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with command line token
  python wan2.1_t2v_1.3B_singleGPU_client.py --token "your_token" --prompt "A beautiful sunset"

  # Usage with token from file (fallback)
  python wan2.1_t2v_1.3B_singleGPU_client.py --prompt "A beautiful sunset"

  # Test connection only
  python wan2.1_t2v_1.3B_singleGPU_client.py --token "your_token" --test-connection

  # With custom parameters
  python wan2.1_t2v_1.3B_singleGPU_client.py --token "your_token" --prompt "A beautiful sunset" --parameters '{"frame_num": 16, "sample_steps": 50}'
        """
    )
    parser.add_argument("--endpoint", default="http://localhost:8080", 
                       help="Endpoint URL")
    parser.add_argument("--token", 
                       help="API token for authentication (recommended over file)")
    parser.add_argument("--output-dir", default="./results", 
                       help="Output directory for results")
    parser.add_argument("--prompt", required=True, 
                       help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", default="", 
                       help="Negative prompt")
    parser.add_argument("--parameters", 
                       help="JSON string of generation parameters")
    parser.add_argument("--test-connection", action="store_true", 
                       help="Test connection only")

    args = parser.parse_args()

    # Load API token - prioritize command line argument
    api_token = args.token
    if not api_token:
        print("ℹ️ No API token provided via --token argument")
        print("💡 Trying to load from api_token.txt file...")
        try:
            token_path = Path("api_token.txt")
            if token_path.exists():
                with open(token_path, "r") as f:
                    token_content = f.read().strip()

                # Skip lines that start with # (comments) and empty lines
                token_lines = [line.strip() for line in token_content.split("\n") 
                             if line.strip() and not line.startswith("#")]

                if token_lines:
                    api_token = token_lines[0]
                    # Check if the token looks like a placeholder
                    if api_token in ["your_api_token_here", 
                                   "your_api_token_here_12345"]:
                        print("❌ API token appears to be a placeholder")
                        print("Please replace 'your_api_token_here' with your actual API token")
                        api_token = ""
                    else:
                        print(f"✅ API token loaded from file: {api_token[:10]}...")
                else:
                    print("❌ API token file is empty or contains only comments")
            else:
                print("⚠️ API token file not found: api_token.txt")
                print("💡 You can provide the token via --token argument or create api_token.txt")
        except Exception as e:
            print(f"⚠️ Error loading API token from file: {e}")
    else:
        print(f"✅ API token provided via command line: {api_token[:10]}...")

    # Initialize client with API token
    client = Wan2_1Client(args.endpoint, api_token=api_token, 
                          output_dir=args.output_dir)

    # Test connection
    if not client.test_connection():
        print("❌ Failed to connect to endpoint")
        sys.exit(1)

    if args.test_connection:
        print("✅ Connection test successful!")
        return

    # Parse parameters
    parameters = None
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid parameters JSON: {e}")
            sys.exit(1)

    # Generate video
    result = client.generate_video(args.prompt, args.negative_prompt, 
                                 parameters)

    if result:
        print(f"✅ Video generated successfully: {result}")
    else:
        print("❌ Video generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
