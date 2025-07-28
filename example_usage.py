#!/usr/bin/env python3
"""
Example usage of the Wan2.1 client with command-line API tokens.

This script demonstrates the secure way to use the video generation client
without storing API tokens in files.
"""

import subprocess
import sys
from pathlib import Path


def run_client_with_token(token: str, prompt: str, endpoint: str = "http://localhost:8080"):
    """
    Run the client with a command-line token.
    
    Args:
        token: API token for authentication
        prompt: Text prompt for video generation
        endpoint: Server endpoint URL
    """
    cmd = [
        "python", "gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py",
        "--token", token,
        "--prompt", prompt,
        "--endpoint", endpoint
    ]
    
    print(f"Running command: {' '.join(cmd[:4])}... [token hidden]")
    print(f"Prompt: {prompt}")
    print(f"Endpoint: {endpoint}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Success!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error occurred:")
        print(e.stderr)
        return False


def test_connection(token: str, endpoint: str = "http://localhost:8080"):
    """
    Test connection to the server.
    
    Args:
        token: API token for authentication
        endpoint: Server endpoint URL
    """
    cmd = [
        "python", "gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py",
        "--token", token,
        "--endpoint", endpoint,
        "--test-connection"
    ]
    
    print(f"Testing connection to {endpoint}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Connection successful!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Connection failed:")
        print(e.stderr)
        return False


def main():
    """Main function demonstrating secure API token usage."""
    print("🔐 Wan2.1 Client - Secure API Token Usage Example")
    print("=" * 50)
    
    # Example 1: Test connection
    print("\n1. Testing connection...")
    token = "your_api_token_here"  # Replace with actual token
    if test_connection(token):
        print("✅ Server is ready!")
    else:
        print("❌ Server connection failed. Please check if the server is running.")
        return
    
    # Example 2: Generate a simple video
    print("\n2. Generating a simple video...")
    prompt = "A beautiful sunset over the ocean, cinematic lighting, 4K quality"
    if run_client_with_token(token, prompt):
        print("✅ Video generation completed!")
    else:
        print("❌ Video generation failed.")
    
    # Example 3: Generate video with custom parameters
    print("\n3. Generating video with custom parameters...")
    cmd = [
        "python", "gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py",
        "--token", token,
        "--prompt", "A majestic eagle soaring through the mountains",
        "--negative-prompt", "blurry, low quality, distorted",
        "--parameters", '{"frame_num": 16, "sample_steps": 50, "sample_guide_scale": 7.5}',
        "--output-dir", "./example_results"
    ]
    
    print(f"Running advanced command: {' '.join(cmd[:4])}... [token hidden]")
    print("Parameters: frame_num=16, sample_steps=50, sample_guide_scale=7.5")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Advanced video generation completed!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Advanced video generation failed:")
        print(e.stderr)


if __name__ == "__main__":
    print("⚠️  IMPORTANT: Replace 'your_api_token_here' with your actual API token!")
    print("⚠️  This is just an example - modify the script with your real token.")
    print()
    
    # Check if client script exists
    client_script = Path("gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py")
    if not client_script.exists():
        print("❌ Client script not found!")
        print("Please ensure you're running this from the project root directory.")
        sys.exit(1)
    
    # Check if server is running (optional)
    print("💡 Make sure the Gradio host server is running before executing this example.")
    print("   You can start it with: python gradio_host/wan2.1_t2v_1.3B_singleGPU_host.py")
    print()
    
    # Uncomment the line below to run the example (after replacing the token)
    # main()
    
    print("🔒 Security Note:")
    print("- Never hardcode API tokens in scripts")
    print("- Use environment variables or command-line arguments")
    print("- Consider using a secrets management system for production")
    print("- This example is for demonstration purposes only")