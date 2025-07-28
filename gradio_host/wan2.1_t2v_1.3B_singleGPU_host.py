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
from typing import Optional

import gradio as gr


def create_gradio_interface(checkpoint_dir, output_dir):
    """
    Create Gradio interface for Wan2.1 text-to-video generation.

    Args:
        checkpoint_dir: Path to model checkpoints
        output_dir: Directory to save generated videos

    Returns:
        Gradio interface
    """

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

            # Create unique output folder
            timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
            random_generation_id = "".join(random.choices(string.ascii_lowercase, k=4))
            output_folder = os.path.join(output_dir, f"generation_{timestamp}_{random_generation_id}")
            os.makedirs(output_folder, exist_ok=True)

            # Set generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "checkpoint_dir": checkpoint_dir,
                "output_folder": output_folder,
                "video_save_name": "output",
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

            # Build command line arguments for generate.py
            cmd_args = [
                "python",
                "/mnt/Wan2.1/generate.py",
                "--task",
                "t2v-1.3B",
                "--prompt",
                prompt,
                "--ckpt_dir",
                checkpoint_dir,
                "--save_file",
                os.path.join(output_folder, "output"),
            ]

            # Add negative prompt if provided
            if negative_prompt:
                cmd_args.extend(["--negative_prompt", negative_prompt])

            # Add additional parameters from JSON with correct mapping
            if "frame_num" in input_json:
                cmd_args.extend(["--frame_num", str(input_json["frame_num"])])
            elif "num_frames" in input_json:
                cmd_args.extend(["--frame_num", str(input_json["num_frames"])])

            if "sample_steps" in input_json:
                cmd_args.extend(["--sample_steps", str(input_json["sample_steps"])])
            elif "num_inference_steps" in input_json:
                cmd_args.extend(["--sample_steps", str(input_json["num_inference_steps"])])

            if "sample_guide_scale" in input_json:
                cmd_args.extend(["--sample_guide_scale", str(input_json["sample_guide_scale"])])
            elif "guidance_scale" in input_json:
                cmd_args.extend(["--sample_guide_scale", str(input_json["guidance_scale"])])

            if "base_seed" in input_json:
                cmd_args.extend(["--base_seed", str(input_json["base_seed"])])
            elif "seed" in input_json:
                cmd_args.extend(["--base_seed", str(input_json["seed"])])

            if "width" in input_json and "height" in input_json:
                cmd_args.extend(["--size", f"{input_json['width']}*{input_json['height']}"])

            # Keep model on GPU for fast generation (no offloading)
            # cmd_args.extend(["--offload_model", "True", "--t5_cpu"])  # Commented out to use GPU

            # Add GPU optimization flags for faster generation with H100
            # Note: generate.py doesn't support --device, --batch_size, --num_workers
            # GPU device is automatically detected by PyTorch

            # Add prompt extension if specified
            if input_json.get("use_prompt_extend", False):
                cmd_args.append("--use_prompt_extend")

            print(f"Running Wan2.1 generation with command: {' '.join(cmd_args)}")

            # Run the generation with H100 GPU optimization
            env = os.environ.copy()
            env["PYTHONPATH"] = "/mnt/Wan2.1"
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use primary H100 GPU
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"  # Larger chunks for H100
            env["CUDA_LAUNCH_BLOCKING"] = "0"  # Non-blocking CUDA operations
            env["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8 for H100
            env["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 for faster computation

            result = subprocess.run(cmd_args, capture_output=True, text=True, cwd="/mnt/Wan2.1", env=env)

            # Always log the output for debugging
            print(f"generate.py stdout: {result.stdout}")
            print(f"generate.py stderr: {result.stderr}")
            print(f"generate.py return code: {result.returncode}")

            if result.returncode != 0:
                error_msg = f"Generation failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                return None, error_msg

            # Look for the generated video
            expected_video_path = os.path.join(output_folder, "output.mp4")

            # Debug: List all files in output directory
            print(f"Files in output directory {output_folder}:")
            if os.path.exists(output_folder):
                for file in os.listdir(output_folder):
                    print(f"  - {file}")
            else:
                print(f"Output directory {output_folder} does not exist!")

            if os.path.exists(expected_video_path):
                return expected_video_path, f"Video generated successfully!\nOutput saved to: {output_folder}\nPrompt: {prompt}"
            else:
                # Check for other possible video files
                video_files = [f for f in os.listdir(output_folder) if f.endswith((".mp4", ".avi", ".mov"))]
                if video_files:
                    video_path = os.path.join(output_folder, video_files[0])
                    return video_path, f"Video generated successfully!\nOutput saved to: {output_folder}\nPrompt: {prompt}"
                else:
                    return None, f"Generation completed but no video file found in {output_folder}"

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
    allowed_paths = os.environ.get("GRADIO_ALLOWED_PATHS", "/mnt/vfm/gradio").split(",")
    save_dir = os.environ.get("GRADIO_SAVE_DIR", "/mnt/vfm/gradio/wan2.1_t2v_1.3B_singleGPU")

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))

    print(f"Starting Gradio Wan2.1 App - {server_name=} {server_port=} {allowed_paths=} {save_dir=}")

    # Check if Wan2.1 code exists
    if not os.path.exists("/mnt/Wan2.1"):
        print("Error: Wan2.1 code not found at /mnt/Wan2.1")
        print("Please ensure the Wan2.1 repository is available at the specified path.")
        sys.exit(1)

    # Check if generate.py exists
    if not os.path.exists("/mnt/Wan2.1/generate.py"):
        print("Error: generate.py not found in /mnt/Wan2.1")
        print("Please ensure the Wan2.1 repository is properly set up.")
        sys.exit(1)

    interface = create_gradio_interface(checkpoint_dir="/mnt/Wan2.1/Wan2.1-T2V-1.3B", output_dir=save_dir)
    interface.launch(
        allowed_paths=allowed_paths,
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,  # Show detailed errors for debugging
    )
