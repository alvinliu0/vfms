#!/usr/bin/env python3
"""
Simple test script for i2v functionality.
"""

import sys
import os
from pathlib import Path

# Add the gradio_client directory to the path
sys.path.insert(0, str(Path(__file__).parent / "gradio_client"))

try:
    # Import using importlib to handle the dots in filename
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wan2_1_i2v_14B_singleGPU_client",
        Path(__file__).parent / "gradio_client" / "wan2.1_i2v_14B_singleGPU_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Wan2_1Client = module.Wan2_1Client
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure the client file exists in gradio_client/")
    sys.exit(1)


def test_i2v_connection(endpoint, token):
    """Test the i2v connection."""
    print("🔄 Testing i2v connection...")
    client = Wan2_1Client(endpoint, api_token=token)
    
    if client.test_connection():
        print("✅ Connection successful!")
        return client
    else:
        print("❌ Connection failed!")
        return None


def test_i2v_generation(client):
    """Test i2v generation with a simple image."""
    image_path = "Wan2.1/examples/i2v_input.JPG"
    prompt = "A cat wearing sunglasses"
    
    print("🎬 Testing i2v generation...")
    print(f"🖼️ Image: {image_path}")
    print(f"📝 Prompt: {prompt}")
    
    # Use parameters matching the official Wan example
    parameters = {
        "checkpoint": "480p",  # Use 480p to save time instead of 720p
    }
    
    result = client.generate_video(prompt, image_path, "", parameters)
    
    if result:
        print(f"✅ Generation successful! Video saved to: {result}")
    else:
        print("❌ Generation failed!")
    
    return result


if __name__ == "__main__":
    print("🚀 Starting i2v test...")
    
    # Get credentials from environment variables or command line
    endpoint = os.environ.get("I2V_ENDPOINT")
    token = os.environ.get("I2V_TOKEN")
    
    if not endpoint or not token:
        print("❌ Missing credentials!")
        print("💡 Set environment variables:")
        print("   export I2V_ENDPOINT='your_endpoint_url'")
        print("   export I2V_TOKEN='your_api_token'")
        print("   Or provide them as command line arguments:")
        print("   python test_i2v_simple.py <endpoint> <token>")
        sys.exit(1)
    
    # Test connection
    client = test_i2v_connection(endpoint, token)
    if client is None:
        sys.exit(1)
    
    # Test generation
    test_i2v_generation(client)
    
    print("✅ Test completed!") 