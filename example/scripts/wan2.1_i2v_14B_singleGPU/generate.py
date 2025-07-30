#!/usr/bin/env python3
"""
Wan2.1 Image-to-Video Generation Script

A simple script that handles CLI arguments for basic inputs and calls the client's
generate_video method for image-to-video generation. Supports both single and batch generation.

Usage:
    # Single generation:
    python example/scripts/wan2.1_i2v_14B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --prompt "your prompt" --image input.jpg
    
    # Batch generation (file-based) - handles prompts with commas
    python example/scripts/wan2.1_i2v_14B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" \
        --batch-prompts-file "example\scripts\wan2.1_i2v_14B_singleGPU\prompts_example.txt" \
        --images-file "example\scripts\wan2.1_i2v_14B_singleGPU\images_example.txt" \
        --batch-negative-prompts-file "example\scripts\wan2.1_i2v_14B_singleGPU\negative_prompts_example.txt"

    # Test connection only
    python example/scripts/wan2.1_i2v_14B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --test-connection

Model: Wan2.1 I2V 14B Single GPU
"""

import argparse
import json
import sys
import importlib.util
from pathlib import Path

# Add the project root to the path so we can import the client
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent  # Go up three levels from generate.py

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the client class using importlib for filename with dots
client_module_path = project_root / "gradio_client" / "wan2.1_i2v_14B_singleGPU_client.py"
spec = importlib.util.spec_from_file_location("wan2_1_client", client_module_path)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)
Wan2_1Client = client_module.Wan2_1Client


def parse_arguments():
    """Parse command line arguments for basic inputs."""
    parser = argparse.ArgumentParser(
        description="Wan2.1 Image-to-Video Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--token", required=True, help="API token for authentication")
    parser.add_argument("--endpoint", required=True, help="Server endpoint URL")
    
    # Generation mode (single vs batch) - only required if not testing connection
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--prompt", help="Text prompt for single video generation")
    group.add_argument("--batch-prompts-file", help="File containing prompts (one per line) for batch generation")
    
    # Image input (required for i2v)
    parser.add_argument("--image", help="Input image path for single generation")
    parser.add_argument("--images-file", help="File containing image paths (one per line) for batch generation")
    
    # Optional arguments
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted, ugly",
                       help="Negative prompt for single generation")
    parser.add_argument("--batch-negative-prompts-file",
                       help="File containing negative prompts (one per line) for batch generation")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--test-connection", action="store_true", help="Test connection only")
    
    # Generation parameters (i2v defaults)
    parser.add_argument("--frame-num", type=int, default=81, help="Number of frames (4n+1 format for i2v)")
    parser.add_argument("--sample-steps", type=int, default=40, help="Inference steps (20-100)")
    parser.add_argument("--sample-guide-scale", type=float, default=5.0, help="Guidance scale (1.0-20.0)")
    parser.add_argument("--base-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=832, 
                       help="Video width (must be divisible by 8)")
    parser.add_argument("--height", type=int, default=480, 
                       help="Video height (must be divisible by 8)")
    parser.add_argument("--checkpoint", choices=["480p", "720p"], default="480p",
                       help="Choose checkpoint model: 480p or 720p")
    parser.add_argument("--use-prompt-extend", action="store_true", help="Use prompt extension")
    
    args = parser.parse_args()
    
    # Validate arguments: if not testing connection, require either prompt+image or batch files
    if not args.test_connection:
        if not args.prompt and not args.batch_prompts_file:
            parser.error("Either --prompt or --batch-prompts-file is required when not using --test-connection")
        
        if args.prompt and not args.image:
            parser.error("--image is required when using --prompt for single generation")
        
        if args.batch_prompts_file and not args.images_file:
            parser.error("--images-file is required when using --batch-prompts-file for batch generation")
    
    return args


def load_prompts_from_file(file_path: str) -> list:
    """Load prompts from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    except Exception as e:
        print(f"❌ Error loading prompts from {file_path}: {e}")
        return []


def load_negative_prompts_from_file(file_path: str) -> list:
    """Load negative prompts from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            negative_prompts = [line.strip() for line in f if line.strip()]
        return negative_prompts
    except Exception as e:
        print(f"❌ Error loading negative prompts from {file_path}: {e}")
        return []


def load_image_paths_from_file(file_path: str) -> list:
    """Load image paths from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        return image_paths
    except Exception as e:
        print(f"❌ Error loading image paths from {file_path}: {e}")
        return []


def main():
    """
    Main function that handles CLI arguments and calls the client's generate_video or batch_generate method.
    """
    print("🎬 Wan2.1 Image-to-Video Generation Script")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    print(f"🔗 Connecting to endpoint: {args.endpoint}")
    client = Wan2_1Client(
        endpoint_url=args.endpoint,
        api_token=args.token,
        output_dir=args.output_dir
    )
    
    # Test connection
    print("🧪 Testing connection...")
    if not client.test_connection():
        print("❌ Connection test failed. Please check your endpoint URL and API token.")
        return False
    
    print("✅ Connection successful!")
    
    # If only testing connection, exit here
    if args.test_connection:
        print("✅ Connection test completed successfully!")
        return True
    
    # Prepare generation parameters
    generation_params = {
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_guide_scale": args.sample_guide_scale,
        "base_seed": args.base_seed,
        "width": args.width,
        "height": args.height,
        "checkpoint": args.checkpoint,
        "use_prompt_extend": args.use_prompt_extend,
    }
    
    # Determine generation mode
    if args.prompt and args.image:
        # Single generation mode
        print("\n📋 Single Generation Parameters:")
        print(f"   Prompt: {args.prompt}")
        print(f"   Image: {args.image}")
        print(f"   Negative Prompt: {args.negative_prompt}")
        print(f"   Frames: {generation_params['frame_num']}")
        print(f"   Steps: {generation_params['sample_steps']}")
        print(f"   Guidance Scale: {generation_params['sample_guide_scale']}")
        print(f"   Resolution: {generation_params['width']}x{generation_params['height']}")
        print(f"   Seed: {generation_params['base_seed']}")
        
        print("\n🚀 Starting single video generation...")
        print("⏳ This may take 1-2 hours. Please be patient...")
        
        try:
            # Call the client's generate_video method
            result_path = client.generate_video(
                prompt=args.prompt,
                image_path=args.image,
                negative_prompt=args.negative_prompt,
                parameters=generation_params
            )
            
            if result_path:
                print("\n✅ Generation completed successfully!")
                print(f"🎥 Video saved to: {result_path}")
                
                # Show output directory structure
                output_path = Path(result_path)
                print(f"📁 Full output directory: {output_path.parent}")
                
                # List files in the output directory
                if output_path.parent.exists():
                    print("📄 Generated files:")
                    for file in output_path.parent.iterdir():
                        if file.is_file():
                            print(f"   - {file.name}")
                
                return True
            else:
                print("❌ Generation failed. Please check the logs above for details.")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ Generation interrupted by user (Ctrl+C)")
            return False
        except Exception as e:
            print(f"\n❌ Generation failed with error: {e}")
            return False
    
    elif args.batch_prompts_file and args.images_file:
        # Batch generation mode
        print("\n📋 Batch Generation Parameters:")
        print(f"   Prompts file: {args.batch_prompts_file}")
        print(f"   Images file: {args.images_file}")
        print(f"   Frames: {generation_params['frame_num']}")
        print(f"   Steps: {generation_params['sample_steps']}")
        print(f"   Guidance Scale: {generation_params['sample_guide_scale']}")
        print(f"   Resolution: {generation_params['width']}x{generation_params['height']}")
        print(f"   Seed: {generation_params['base_seed']}")
        
        # Load prompts and images from files
        prompts = load_prompts_from_file(args.batch_prompts_file)
        image_paths = load_image_paths_from_file(args.images_file)
        
        if not prompts:
            print("❌ No prompts loaded from file")
            return False
        
        if not image_paths:
            print("❌ No image paths loaded from file")
            return False
        
        if len(prompts) != len(image_paths):
            print(f"❌ Number of prompts ({len(prompts)}) doesn't match number of images ({len(image_paths)})")
            return False
        
        # Load negative prompts if provided
        negative_prompts = []
        if args.batch_negative_prompts_file:
            negative_prompts = load_negative_prompts_from_file(args.batch_negative_prompts_file)
            if len(negative_prompts) != len(prompts):
                print(f"❌ Number of negative prompts ({len(negative_prompts)}) doesn't match number of prompts ({len(prompts)})")
                return False
        
        print(f"\n🚀 Starting batch generation of {len(prompts)} videos...")
        print("⏳ This may take several hours. Please be patient...")
        
        try:
            # Call the client's batch_generate method
            results = client.batch_generate(
                prompts=prompts,
                image_paths=image_paths,
                negative_prompts=negative_prompts,
                parameters=generation_params
            )
            
            # Count successful generations
            successful = [r for r in results if r]
            print(f"\n✅ Batch generation completed!")
            print(f"🎥 Successfully generated {len(successful)}/{len(prompts)} videos")
            
            if successful:
                print("📁 Generated videos:")
                for i, result in enumerate(results):
                    if result:
                        print(f"   {i+1}. {result}")
                
                return True
            else:
                print("❌ No videos were generated successfully. Please check the logs above for details.")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ Batch generation interrupted by user (Ctrl+C)")
            return False
        except Exception as e:
            print(f"\n❌ Batch generation failed with error: {e}")
            return False
    
    else:
        print("❌ Invalid arguments. Please check your command line arguments.")
        return False


if __name__ == "__main__":
    main() 