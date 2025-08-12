"""
Gradio app for Wan2.1 text-to-video generation with 1.3B model on single GPU.

This app provides a clean interface for text-to-video generation using the Wan2.1 model.
It follows the same patterns as the cosmos_transfer1_3.50.2.py app but adapted for
text-to-video generation using the actual Wan2.1 codebase.

Tested with gradio==5.38.2
"""

import datetime
import json
import os
import random
import string
import subprocess
import sys
import traceback
import zoneinfo
import threading
from typing import Optional

import gradio as gr


def sync_to_pbss(local_path: str, job_id: str, model_name: str):
    """
    Synchronize generated results to PBSS using s5cmd.
    
    Args:
        local_path: Local path to the generated video
        job_id: Unique job identifier
        model_name: Name of the model used for generation
    """
    try:
        # Get PBSS configuration from environment variables
        endpoint = os.environ.get("TEAM_COSMOS_BENCHMARK_ENDPOINT")
        region = os.environ.get("TEAM_COSMOS_BENCHMARK_REGION")
        secret_key = os.environ.get("XIANL_TEAM_COSMOS_BENCHMARK")
        
        if not all([endpoint, region, secret_key]):
            print(f"⚠️ Missing PBSS configuration. Endpoint: {endpoint}, Region: {region}, Secret: {'*' * 10 if secret_key else 'None'}")
            return False
        
        # Credentials should be set up beforehand using setup_pbss_credentials.py
        # Just use the existing credentials in /root/.aws/
        env = os.environ.copy()
        
        # Construct PBSS destination path
        timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
        pbss_dest = f"s3://evaluation_videos/debug/lepton_api/{model_name}/{timestamp}_{job_id}/"
        
        # Run s5cmd sync command with profile
        cmd = ["s5cmd", "--profile", "team-cosmos-benchmark", "cp", local_path, pbss_dest]
        
        print(f"🔄 Syncing to PBSS: {' '.join(cmd)}")
        print(f"🌐 Using endpoint: {endpoint}")
        print(f"🌍 Using region: {region}")
        print(f"👤 Using profile: team-cosmos-benchmark")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"✅ Successfully synced to PBSS: {pbss_dest}")
            return True
        else:
            print(f"❌ Failed to sync to PBSS: {result.stderr}")
            print(f"stdout: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Error during PBSS sync: {e}")
        return False


def run_generation_async(generation_params: dict, output_folder: str, job_id: str, model_name: str):
    """
    Run generation in a background thread and sync results to PBSS.
    
    Args:
        generation_params: Generation parameters
        output_folder: Output folder path
        job_id: Unique job identifier
        model_name: Name of the model
    """
    try:
        print(f"🔄 Starting async generation for job {job_id}...")
        
        # Get the Wan2.1 directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        wan2_1_dir = os.path.join(project_root, "Wan2.1")
        generate_py_path = os.path.join(wan2_1_dir, "generate.py")
        
        # Build command line arguments for generate.py
        cmd_args = [
            "python",
            generate_py_path,
            "--task",
            "t2v-1.3B",
            "--prompt",
            generation_params["prompt"],
            "--ckpt_dir",
            generation_params["checkpoint_dir"],
            "--save_file",
            os.path.join(output_folder, "output.mp4"),
            "--offload_model",
            "False",
            "--sample_shift",
            "5.0",
            "--sample_solver",
            "unipc",
        ]

        # Add additional parameters from JSON with correct mapping
        if "frame_num" in generation_params:
            cmd_args.extend(["--frame_num", str(generation_params["frame_num"])])
        elif "num_frames" in generation_params:
            cmd_args.extend(["--frame_num", str(generation_params["num_frames"])])

        if "sample_steps" in generation_params:
            cmd_args.extend(["--sample_steps", str(generation_params["sample_steps"])])
        elif "num_inference_steps" in generation_params:
            cmd_args.extend(["--sample_steps", str(generation_params["num_inference_steps"])])

        if "sample_guide_scale" in generation_params:
            cmd_args.extend(["--sample_guide_scale", str(generation_params["sample_guide_scale"])])
        elif "guidance_scale" in generation_params:
            cmd_args.extend(["--sample_guide_scale", str(generation_params["sample_guide_scale"])])

        if "base_seed" in generation_params:
            cmd_args.extend(["--base_seed", str(generation_params["base_seed"])])
        elif "seed" in generation_params:
            cmd_args.extend(["--base_seed", str(generation_params["seed"])])

        if "width" in generation_params and "height" in generation_params:
            cmd_args.extend(["--size", f"{generation_params['width']}*{generation_params['height']}"])

        if generation_params.get("use_prompt_extend", False):
            cmd_args.append("--use_prompt_extend")

        print(f"Running Wan2.1 generation with command: {' '.join(cmd_args)}")

        # Run the generation with H100 GPU optimization
        env = os.environ.copy()
        env["PYTHONPATH"] = wan2_1_dir
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
        env["CUDA_LAUNCH_BLOCKING"] = "0"
        env["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        env["NVIDIA_TF32_OVERRIDE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True"
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        print(f"generate.py return code: {result.returncode}")
        if result.stdout:
            print(f"generate.py stdout: {result.stdout}")
        if result.stderr:
            print(f"generate.py stderr: {result.stderr}")

        if result.returncode == 0:
            # Look for the generated video
            expected_video_path = os.path.join(output_folder, "output.mp4")
            
            if os.path.exists(expected_video_path):
                print(f"✅ Generation completed successfully for job {job_id}")
                
                # Sync to PBSS
                print(f"🔄 Syncing results to PBSS for job {job_id}...")
                sync_success = sync_to_pbss(expected_video_path, job_id, model_name)
                
                if sync_success:
                    print(f"✅ Job {job_id} completed and synced to PBSS successfully!")
                else:
                    print(f"⚠️ Job {job_id} completed but PBSS sync failed")
                
                # Write completion status
                status_file = os.path.join(output_folder, "generation_status.json")
                # Generate timestamp for PBSS path
                timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
                status_data = {
                    "job_id": job_id,
                    "status": "completed",
                    "completion_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                    "video_path": expected_video_path,
                    "pbss_sync": sync_success,
                    "pbss_path": f"s3://evaluation_videos/debug/lepton_api/{model_name}/{timestamp}_{job_id}/"
                }
                
                with open(status_file, "w") as f:
                    json.dump(status_data, f, indent=2)
                    
            else:
                print(f"❌ Generation failed for job {job_id}: No video file found")
                # Write failure status
                status_file = os.path.join(output_folder, "generation_status.json")
                status_data = {
                    "job_id": job_id,
                    "status": "failed",
                    "failure_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                    "error": "No video file generated"
                }
                
                with open(status_file, "w") as f:
                    json.dump(status_data, f, indent=2)
        else:
            print(f"❌ Generation failed for job {job_id} with return code {result.returncode}")
            # Write failure status
            status_file = os.path.join(output_folder, "generation_status.json")
            status_data = {
                "job_id": job_id,
                "status": "failed",
                "failure_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                "error": f"Generation failed with return code {result.returncode}",
                "stderr": result.stderr
            }
            
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)
                
    except Exception as e:
        print(f"❌ Error during async generation for job {job_id}: {e}")
        # Write error status
        status_file = os.path.join(output_folder, "generation_status.json")
        status_data = {
            "job_id": job_id,
            "status": "error",
            "error_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
            "error": str(e)
        }
        
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)


def create_gradio_interface(checkpoint_dir, output_dir):
    """
    Create Gradio interface for Wan2.1 text-to-video generation.

    Args:
        checkpoint_dir: Path to model checkpoints
        output_dir: Directory to save generated videos

    Returns:
        Gradio interface
    """
    
    # Define model name for output organization
    model_name = "wan2.1_t2v_1.3B_singleGPU"

    def _infer(prompt, negative_prompt, input_text: str, api_token: str = "") -> tuple[Optional[str], Optional[str]]:
        """
        Inference function which is called by the gradio app.

        - takes the inputs from the UI (prompt, negative prompt, and JSON config)
        - parses the JSON configuration
        - runs the Wan2.1 text-to-video generation using generate.py
        - returns the output video path and/or status message

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain elements
            input_text: JSON string containing generation parameters
            api_token: API token for authentication (optional)

        Returns:
            Tuple containing the path to the output video and the status message.
        """
        try:
            # Validate API token if provided
            if api_token:
                # You can add token validation logic here
                # For now, we'll just log that a token was provided
                print(f"API token provided: {api_token[:10]}...")

            if not prompt:
                raise ValueError("No prompt provided. Please enter a text prompt.")

            if not input_text:
                raise ValueError("No configuration provided. Please provide a valid JSON string.")

            # Parse input text as JSON
            input_json = json.loads(input_text)

            # Create unique output folder with model name and job ID
            timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
            random_generation_id = "".join(random.choices(string.ascii_lowercase, k=4))
            job_id = f"{timestamp}_{random_generation_id}"
            
            # Create model-specific output directory
            model_output_dir = os.path.join(output_dir, model_name)
            output_folder = os.path.join(model_output_dir, f"generation_{job_id}")
            os.makedirs(output_folder, exist_ok=True)

            # Set generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "checkpoint_dir": checkpoint_dir,
                "output_folder": output_folder,
                "video_save_name": "output",
                "model_name": model_name,
                **input_json,  # Merge additional parameters from JSON
            }

            # Write generation params to file for reference
            params_path = os.path.join(output_folder, "generation_params.json")
            with open(params_path, "w") as f:
                json.dump(generation_params, f, indent=2)

            # Write prompt and negative prompt to files
            with open(os.path.join(output_folder, "prompt.txt"), "w") as f:
                f.write(prompt)

            with open(os.path.join(output_folder, "negative_prompt.txt"), "w") as f:
                f.write(negative_prompt)

            # Write initial job status
            status_file = os.path.join(output_folder, "generation_status.json")
            initial_status = {
                "job_id": job_id,
                "status": "submitted",
                "submission_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "parameters": input_json
            }
            
            with open(status_file, "w") as f:
                json.dump(initial_status, f, indent=2)

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=run_generation_async,
                args=(generation_params, output_folder, job_id, model_name)
            )
            generation_thread.daemon = True
            generation_thread.start()

            # Return immediate response with job ID and PBSS location
            status_message = "✅ Generation job submitted successfully!\n\n"
            status_message += f"🆔 Job ID: {job_id}\n"
            status_message += f"📁 Output folder: {output_folder}\n"
            status_message += f"📝 Prompt: {prompt}\n"
            status_message += f"⏰ Submission time: {timestamp}\n\n"
            status_message += "🔄 Generation is running in the background.\n"
            status_message += "📊 Check generation_status.json for progress updates.\n"
            status_message += f"🌐 Results will be synced to PBSS at: s3://evaluation_videos/debug/lepton_api/{model_name}/{timestamp}_{job_id}/"

            return None, status_message

        except Exception:
            return None, f"Error during generation:\n\n{traceback.format_exc()}"

    # Define the gradio interface (UI/API) - using gr.Interface to expose API endpoints
    interface = gr.Interface(
        fn=_infer,
        inputs=[
            gr.Textbox(label="Prompt", lines=3, placeholder="A beautiful sunset over the ocean, cinematic lighting, 4K quality"),
            gr.Textbox(label="Negative Prompt", lines=2, placeholder="blurry, low quality, distorted, ugly"),
            gr.Textbox(
                label="Generation Parameters JSON",
                lines=10,
                placeholder=json.dumps(
                    {
                        "frame_num": 16,
                        "sample_steps": 50,
                        "sample_guide_scale": 7.5,
                        "base_seed": 42,
                        "width": 832,
                        "height": 480,
                        "use_prompt_extend": False,
                    },
                    indent=2,
                ),
            ),
            gr.Textbox(label="API Token (Optional)", type="password", placeholder="Enter your API token here..."),
        ],
        outputs=[gr.Video(label="Generated Video", height=500), gr.Textbox(label="Output", lines=10)],
        title="Wan2.1: Text-to-Video Generation",
        description="Generate videos from text prompts using the Wan2.1 1.3B model. GPU: H100 (80GB) - Optimized for fast generation",
        theme=gr.themes.Soft(),
    )

    # Configure queue settings
    concurrency_limit = int(os.environ.get("GRADIO_CONCURRENCY_LIMIT", 1))
    max_queue_size = int(os.environ.get("GRADIO_MAX_QUEUE_SIZE", 20))
    try:
        status_update_rate = float(os.environ.get("GRADIO_STATUS_UPDATE_RATE", 1.0))
    except ValueError:
        status_update_rate = "auto"

    print(f"Configuring queue with {concurrency_limit=} {max_queue_size=} {status_update_rate=}")

    return interface.queue(
        max_size=max_queue_size,
    )


if __name__ == "__main__":
    # Get current directory and construct paths relative to it
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    
    # Wan2.1 paths - use relative paths from project root
    wan2_1_dir = os.path.join(project_root, "Wan2.1")
    generate_py_path = os.path.join(wan2_1_dir, "generate.py")
    checkpoint_dir = os.path.join(wan2_1_dir, "Wan2.1-T2V-1.3B")
    
    # Environment variables with fallbacks
    save_dir = os.environ.get("GRADIO_SAVE_DIR", "/mnt/vfms/gradio_output")
    allowed_paths = os.environ.get("GRADIO_ALLOWED_PATHS", f"{project_root},{save_dir}").split(",")

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))

    print(f"Starting Gradio Wan2.1 App - {server_name=} {server_port=}")
    print(f"Project root: {project_root}")
    print(f"Wan2.1 directory: {wan2_1_dir}")
    print(f"Generate.py path: {generate_py_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Save directory: {save_dir}")

    # Check if Wan2.1 code exists
    if not os.path.exists(wan2_1_dir):
        print(f"Error: Wan2.1 code not found at {wan2_1_dir}")
        print("Please ensure the Wan2.1 repository is available in the project directory.")
        print("Expected structure:")
        print(f"  {project_root}/")
        print(f"  ├── Wan2.1/")
        print(f"  │   ├── generate.py")
        print(f"  │   └── Wan2.1-T2V-1.3B/")
        print(f"  └── gradio_host/")
        sys.exit(1)

    # Check if generate.py exists
    if not os.path.exists(generate_py_path):
        print(f"Error: generate.py not found at {generate_py_path}")
        print("Please ensure the Wan2.1 repository is properly set up.")
        sys.exit(1)

    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found at {checkpoint_dir}")
        print("You may need to download the model checkpoints.")
        print("Please refer to the Wan2.1 documentation for model setup.")

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    interface = create_gradio_interface(checkpoint_dir=checkpoint_dir, output_dir=save_dir)
    interface.launch(
        allowed_paths=allowed_paths,
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,  # Show detailed errors for debugging
    )
