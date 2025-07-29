#!/usr/bin/env python3
"""
Wan2.1 Text-to-Video Generation Script

A simple script that handles CLI arguments for basic inputs and calls the client's
generate_video method for video generation. Supports both single and batch generation.

Usage:
    # Single generation:
    python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --prompt "your prompt"
    
    # Batch generation (comma-separated)
    python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts "prompt1,prompt2,prompt3"
    
    # Batch generation (file-based) - handles prompts with commas
    python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" \
        --batch-prompts-file "example\scripts\wan2.1_t2v_1.3B_singleGPU\prompts_example.txt" \
            --batch-negative-prompts-file "example\scripts\wan2.1_t2v_1.3B_singleGPU\negative_prompts_example.txt"

    # Test connection only
    python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --test-connection

Model: Wan2.1 T2V 1.3B Single GPU
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
client_module_path = project_root / "gradio_client" / "wan2.1_t2v_1.3B_singleGPU_client.py"
spec = importlib.util.spec_from_file_location("wan2_1_client", client_module_path)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)
Wan2_1Client = client_module.Wan2_1Client


def parse_arguments():
    """Parse command line arguments for basic inputs."""
    parser = argparse.ArgumentParser(
        description="Wan2.1 Text-to-Video Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--token", required=True, help="API token for authentication")
    parser.add_argument("--endpoint", required=True, help="Server endpoint URL")
    
    # Generation mode (single vs batch) - only required if not testing connection
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--prompt", help="Text prompt for single video generation")
    group.add_argument("--batch-prompts", help="Comma-separated prompts for batch generation")
    group.add_argument("--batch-prompts-file", help="File containing prompts (one per line) for batch generation")
    
    # Optional arguments
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted, ugly",
                       help="Negative prompt for single generation")
    parser.add_argument("--batch-negative-prompts", 
                       help="Comma-separated negative prompts for batch generation")
    parser.add_argument("--batch-negative-prompts-file",
                       help="File containing negative prompts (one per line) for batch generation")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--test-connection", action="store_true", help="Test connection only")
    
    # Generation parameters
    parser.add_argument("--frame-num", type=int, default=16, help="Number of frames (8-32)")
    parser.add_argument("--sample-steps", type=int, default=50, help="Inference steps (20-100)")
    parser.add_argument("--sample-guide-scale", type=float, default=7.5, help="Guidance scale (1.0-20.0)")
    parser.add_argument("--base-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=832, help="Video width (must be divisible by 8)")
    parser.add_argument("--height", type=int, default=480, help="Video height (must be divisible by 8)")
    parser.add_argument("--use-prompt-extend", action="store_true", help="Use prompt extension")
    
    args = parser.parse_args()
    
    # Validate arguments: if not testing connection, require either prompt or batch-prompts
    if not args.test_connection and not args.prompt and not args.batch_prompts and not args.batch_prompts_file:
        parser.error("Either --prompt, --batch-prompts, or --batch-prompts-file is required when not using --test-connection")
    
    # Validate batch arguments: can't use both file and string for prompts
    if args.batch_prompts and args.batch_prompts_file:
        parser.error("Cannot use both --batch-prompts and --batch-prompts-file")
    
    # Validate batch arguments: can't use both file and string for negative prompts
    if args.batch_negative_prompts and args.batch_negative_prompts_file:
        parser.error("Cannot use both --batch-negative-prompts and --batch-negative-prompts-file")
    
    return args


def main():
    """
    Main function that handles CLI arguments and calls the client's generate_video or batch_generate method.
    """
    print("🎬 Wan2.1 Video Generation Script")
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
        "use_prompt_extend": args.use_prompt_extend,
    }
    
    # Determine generation mode
    if args.prompt:
        # Single generation mode
        print("\n📋 Single Generation Parameters:")
        print(f"   Prompt: {args.prompt}")
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
    
    elif args.batch_prompts or args.batch_prompts_file:
        # Batch generation mode
        prompts = []
        
        if args.batch_prompts:
            # Parse comma-separated prompts
            prompts = [p.strip() for p in args.batch_prompts.split(",") if p.strip()]
        elif args.batch_prompts_file:
            # Read prompts from file
            try:
                with open(args.batch_prompts_file, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"❌ Error: Batch prompts file not found: {args.batch_prompts_file}")
                return False
            except Exception as e:
                print(f"❌ Error reading batch prompts file: {e}")
                return False
        
        if not prompts:
            print("❌ Error: No valid prompts provided in batch mode")
            return False
        
        # Handle negative prompts for batch
        negative_prompts = None
        if args.batch_negative_prompts:
            # Parse comma-separated negative prompts
            negative_prompts = [p.strip() for p in args.batch_negative_prompts.split(",") if p.strip()]
        elif args.batch_negative_prompts_file:
            # Read negative prompts from file
            try:
                with open(args.batch_negative_prompts_file, 'r', encoding='utf-8') as f:
                    negative_prompts = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"❌ Error: Batch negative prompts file not found: {args.batch_negative_prompts_file}")
                return False
            except Exception as e:
                print(f"❌ Error reading batch negative prompts file: {e}")
                return False
        
        # Validate negative prompts length
        if negative_prompts and len(negative_prompts) != len(prompts):
            print("⚠️ Warning: Number of negative prompts doesn't match number of prompts")
            print("   Using default negative prompt for unmatched prompts")
            # Extend negative prompts to match prompts length
            while len(negative_prompts) < len(prompts):
                negative_prompts.append(args.negative_prompt)
        
        print(f"\n📋 Batch Generation Parameters:")
        print(f"   Number of videos: {len(prompts)}")
        print(f"   Frames: {generation_params['frame_num']}")
        print(f"   Steps: {generation_params['sample_steps']}")
        print(f"   Guidance Scale: {generation_params['sample_guide_scale']}")
        print(f"   Resolution: {generation_params['width']}x{generation_params['height']}")
        print(f"   Seed: {generation_params['base_seed']}")
        
        print(f"\n📝 Prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"   {i}. {prompt}")
        
        if negative_prompts:
            print(f"\n🚫 Negative Prompts:")
            for i, neg_prompt in enumerate(negative_prompts, 1):
                print(f"   {i}. {neg_prompt}")
        
        print(f"\n🚀 Starting batch generation of {len(prompts)} videos...")
        print("⏳ This may take several hours. Please be patient...")
        
        try:
            # Call the client's batch_generate method
            result_paths = client.batch_generate(
                prompts=prompts,
                negative_prompts=negative_prompts,
                parameters=generation_params
            )
            
            # Count successful generations
            successful_generations = [path for path in result_paths if path]
            
            print(f"\n✅ Batch generation completed!")
            print(f"🎥 Successfully generated: {len(successful_generations)}/{len(prompts)} videos")
            
            if successful_generations:
                print(f"\n📁 Generated videos:")
                for i, result_path in enumerate(result_paths, 1):
                    if result_path:
                        print(f"   {i}. {result_path}")
                    else:
                        print(f"   {i}. ❌ Failed")
            
            return len(successful_generations) > 0
            
        except KeyboardInterrupt:
            print("\n⚠️ Batch generation interrupted by user (Ctrl+C)")
            return False
        except Exception as e:
            print(f"\n❌ Batch generation failed with error: {e}")
            return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Script completed successfully!")
    else:
        print("\n💥 Script failed. Please check the error messages above.")
        sys.exit(1) 