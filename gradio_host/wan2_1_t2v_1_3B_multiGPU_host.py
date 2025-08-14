# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
Gradio host for Wan2.1 text-to-video generation endpoint (1.3B model with multi-GPU).

This script provides a Gradio web interface and API for:
1. Text-to-video generation using Wan2.1 1.3B model
2. Multi-GPU acceleration with FSDP (Fully Sharded Data Parallel)
3. Asynchronous job processing with PBSS integration
4. RESTful API endpoints for client integration

Usage:
    python wan2_1_t2v_1_3B_multiGPU_host.py
"""

import datetime
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import zoneinfo
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr


def sync_to_pbss(local_path: str, job_id: str, model_name: str) -> bool:
    """
    Sync generated video to PBSS (Persistent Blob Storage Service).

    Args:
        local_path: Local path to the video file
        job_id: Unique job identifier
        model_name: Name of the model

    Returns:
        True if sync successful, False otherwise
    """
    try:
        # PBSS configuration
        endpoint = os.environ.get("PBSS_ENDPOINT", "https://pbss.nvda-ai.com")
        region = os.environ.get("PBSS_REGION", "us-east-1")

        # Credentials should be set up beforehand using setup_pbss_credentials.py
        # Just use the existing credentials in /root/.aws/
        env = os.environ.copy()

        # Construct PBSS destination path
        timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
        pbss_dest = f"s3://evaluation_videos/lepton_api/{model_name}/{timestamp}_{job_id}/"

        # Run s5cmd sync command with profile and endpoint
        cmd = ["s5cmd", "--profile", "team-cosmos-benchmark", "--endpoint-url", endpoint, "cp", local_path, pbss_dest]

        print(f"🔄 Syncing to PBSS: {' '.join(cmd)}")
        print(f"🌐 Using endpoint: {endpoint}")
        print(f"🌍 Using region: {region}")
        print("👤 Using profile: team-cosmos-benchmark")

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
        # Use torchrun for multi-GPU inference with Wan2.1 1.3B parameters
        cmd_args = [
            "torchrun",
            "--nproc_per_node=8",  # Use 8 GPUs
            generate_py_path,
            "--task",
            "t2v-1.3b",  # Fixed: lowercase 'b' for Wan2.1 1.3B model
            "--prompt",
            generation_params["prompt"],
            "--ckpt_dir",
            os.path.join(wan2_1_dir, "Wan2.1-T2V-1.3B"),  # Updated for Wan2.1 1.3B model
            "--save_file",
            os.path.join(output_folder, "output.mp4"),
            "--dit_fsdp",  # Enable FSDP for DIT model
            "--t5_fsdp",  # Enable FSDP for T5 text encoder
            "--ulysses_size",
            "8",  # Ulysses communication group size
            "--sample_shift",
            "5.0",
            "--sample_solver",
            "unipc",
        ]

        # Add additional parameters from generation_params
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
            cmd_args.extend(["--sample_guide_scale", str(generation_params["guidance_scale"])])

        if "base_seed" in generation_params:
            cmd_args.extend(["--base_seed", str(generation_params["base_seed"])])
        elif "seed" in generation_params:
            cmd_args.extend(["--base_seed", str(generation_params["seed"])])

        if "size" in generation_params:
            # Use the size parameter directly (format: "WIDTH*HEIGHT")
            cmd_args.extend(["--size", str(generation_params["size"])])
        elif "width" in generation_params and "height" in generation_params:
            # Fallback: combine width and height into size parameter
            size_param = f"{generation_params['width']}*{generation_params['height']}"
            cmd_args.extend(["--size", size_param])

        # Run generation
        print(f"🚀 Starting generation for job {job_id}...")
        print(f"📝 Prompt: {generation_params['prompt']}")
        print(f"⚙️ Parameters: {generation_params}")

        result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode == 0:
            # Generation successful
            video_path = os.path.join(output_folder, "output.mp4")

            # Check if video file exists
            if os.path.exists(video_path):
                # Update status
                status_data = {
                    "job_id": job_id,
                    "status": "completed",
                    "completion_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                    "video_path": video_path,
                    "pbss_sync": False,
                    "pbss_path": None,
                }

                # Try to sync to PBSS
                sync_success = sync_to_pbss(video_path, job_id, model_name)
                if sync_success:
                    status_data["pbss_sync"] = True
                    # Generate timestamp for PBSS path (just the date, not the full timestamp)
                    date_only = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
                    status_data["pbss_path"] = f"s3://evaluation_videos/lepton_api/{model_name}/{date_only}_{job_id}/"

                # Save final status
                status_file = os.path.join(output_folder, "generation_status.json")
                with open(status_file, "w") as f:
                    json.dump(status_data, f, indent=2)

                print(f"✅ Generation completed successfully for job {job_id}")
                print(f"📁 Video saved to: {video_path}")

            else:
                raise FileNotFoundError(f"Expected video file not found: {video_path}")

        else:
            # Generation failed
            error_msg = f"Generation failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"

            status_data = {
                "job_id": job_id,
                "status": "error",
                "error_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                "error": error_msg,
            }

            status_file = os.path.join(output_folder, "generation_status.json")
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)

            print(f"❌ Generation failed for job {job_id}: {error_msg}")

    except Exception as e:
        print(f"❌ Error in async generation: {e}")
        traceback.print_exc()


def create_gradio_interface(checkpoint_dir, output_dir):
    """
    Create Gradio interface for Wan2.1 text-to-video generation with 1.3B model and multi-GPU support.

    Args:
        checkpoint_dir: Path to model checkpoints
        output_dir: Directory to save generated videos

    Returns:
        Gradio interface
    """

    # Define model name for output organization
    model_name = "wan2_1_t2v_1_3B_multiGPU"

    # Capture checkpoint_dir in closure for _infer function to access
    checkpoint_dir_closure = checkpoint_dir

    def _infer(prompt, negative_prompt, input_text: str, api_token: str = "") -> tuple[Optional[str], Optional[str]]:
        """
        Main inference function for text-to-video generation.

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain elements
            input_text: JSON string with generation parameters
            api_token: API token for authentication (optional)

        Returns:
            Tuple of (video_path, status_message)
        """
        try:
            print(f"🎬 Starting Wan2.1 1.3B Multi-GPU generation...")
            print(f"📝 Prompt: {prompt}")
            print(f"📝 Negative Prompt: {negative_prompt}")

            # Parse generation parameters
            try:
                generation_params = json.loads(input_text) if input_text.strip() else {}
            except json.JSONDecodeError as e:
                return None, f"❌ Invalid JSON in parameters: {e}"

            # Add prompt and negative_prompt to generation_params
            generation_params["prompt"] = prompt
            generation_params["negative_prompt"] = negative_prompt

            # Generate unique job ID
            import uuid
            job_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Create output directory structure
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            generation_dir = os.path.join(model_output_dir, f"generation_{timestamp}_{job_id}")
            os.makedirs(generation_dir, exist_ok=True)

            # Save initial job metadata
            job_metadata = {
                "job_id": job_id,
                "model": model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "parameters": generation_params,
                "submission_time": timestamp,
                "status": "submitted",
            }

            metadata_file = os.path.join(generation_dir, "job_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(job_metadata, f, indent=2)

            # Save initial status
            status_data = {
                "job_id": job_id,
                "status": "submitted",
                "submission_time": datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).isoformat(),
                "prompt": prompt,
                "parameters": generation_params,
            }

            status_file = os.path.join(generation_dir, "generation_status.json")
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)

            print(f"🆔 Job ID: {job_id}")
            print(f"📁 Output directory: {generation_dir}")

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=run_generation_async,
                args=(generation_params, generation_dir, job_id, model_name),
                daemon=True,
            )
            generation_thread.start()

            # Return immediate response
            status_message = f"✅ Job submitted successfully!\n🆔 Job ID: {job_id}\n📁 Output: {generation_dir}\n🔄 Generation in progress...\n💡 Check status or sync from PBSS later"

            return None, status_message

        except Exception as e:
            error_msg = f"❌ Error during generation: {e}"
            print(error_msg)
            traceback.print_exc()
            return None, error_msg

    # Define the gradio interface (UI/API) - using gr.Interface to expose API endpoints
    interface = gr.Interface(
        fn=_infer,
        inputs=[
            gr.Textbox(
                label="Prompt", lines=3, placeholder="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
            ),
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
                        "size": "1280*720",  # Updated for 1.3B multi-GPU default size
                        "use_prompt_extend": False,
                    },
                    indent=2,
                ),
            ),
            gr.Textbox(label="API Token (Optional)", type="password", placeholder="Enter your API token here..."),
        ],
        outputs=[gr.Video(label="Generated Video", height=500), gr.Textbox(label="Output", lines=10)],
        title="Wan2.1: Text-to-Video Generation (1.3B Model - Multi-GPU)",
        description="Generate videos from text prompts using the Wan2.1 1.3B model with 8-GPU FSDP acceleration. Fast and efficient generation.",
        theme=gr.themes.Soft(),
    )

    # Configure queue settings
    concurrency_limit = int(os.environ.get("GRADIO_CONCURRENCY_LIMIT", 1))
    max_queue_size = int(os.environ.get("GRADIO_MAX_QUEUE_SIZE", 20))
    try:
        status_update_rate = float(os.environ.get("GRADIO_STATUS_UPDATE_RATE", 1.0))
    except ValueError:
        status_update_rate = "auto"

    print(
        f"Configuring queue with concurrency_limit={concurrency_limit} max_queue_size={max_queue_size} status_update_rate={status_update_rate}"
    )

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
    checkpoint_dir = os.path.join(wan2_1_dir, "Wan2.1-T2V-1.3B")  # Updated for Wan2.1 1.3B model

    # Environment variables with fallbacks
    save_dir = os.environ.get("GRADIO_SAVE_DIR", "/mnt/vfms/gradio_output")
    allowed_paths = os.environ.get("GRADIO_ALLOWED_PATHS", f"{project_root},{save_dir}").split(",")

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))

    print(f"Starting Gradio Wan2.1 1.3B Multi-GPU App - server_name={server_name} server_port={server_port}")
    print(f"Project root: {project_root}")
    print(f"Wan2.1 directory: {wan2_1_dir}")
    print(f"Generate.py path: {generate_py_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Save directory: {save_dir}")
    print("Multi-GPU configuration: 8 GPUs with FSDP (DIT + T5)")

    # Check if Wan2.1 code exists
    if not os.path.exists(wan2_1_dir):
        print(f"Error: Wan2.1 code not found at {wan2_1_dir}")
        print("Please ensure the Wan2.1 repository is available in the project directory.")
        print("Expected structure:")
        print(f"  {project_root}/")
        print("  ├── Wan2.1/")
        print("  │   ├── generate.py")
        print("  │   └── Wan2.1-T2V-1.3B/")
        print("  └── gradio_host/")
        sys.exit(1)

    if not os.path.exists(generate_py_path):
        print(f"Error: generate.py not found at {generate_py_path}")
        print("Please ensure the Wan2.1 repository is properly set up.")
        sys.exit(1)

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        print("Please ensure the Wan2.1-T2V-1.3B checkpoints are available.")
        sys.exit(1)

    # Create the interface
    interface = create_gradio_interface(checkpoint_dir, save_dir)

    # Launch the app
    print(f"🚀 Launching Wan2.1 1.3B Multi-GPU Gradio app on {server_name}:{server_port}")
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        allowed_paths=allowed_paths,
    ) 