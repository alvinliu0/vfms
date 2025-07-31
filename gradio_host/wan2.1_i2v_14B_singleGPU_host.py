"""
Gradio app for Wan2.1 image-to-video generation with 14B model on single GPU.

This app provides a clean interface for image-to-video generation using the Wan2.1 model.
It follows the same patterns as the t2v models but adapted for image-to-video generation.

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
    Create Gradio interface for Wan2.1 image-to-video generation.

    Args:
        checkpoint_dir: Path to model checkpoints
        output_dir: Directory to save generated videos

    Returns:
        Gradio interface
    """
    
    # Define model name for output organization
    model_name = "wan2.1_i2v_14B_singleGPU"

    def _infer(prompt, negative_prompt, input_image, input_text: str, api_token: str = "") -> tuple[Optional[str], Optional[str]]:
        """
        Inference function which is called by the gradio app.

        - takes the inputs from the UI (prompt, negative prompt, image, and JSON config)
        - parses the JSON configuration
        - runs the Wan2.1 image-to-video generation using generate.py
        - returns the output video path and/or status message

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain elements
            input_image: Input image for video generation
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

            if not input_image:
                raise ValueError("No input image provided. Please upload an image.")

            if not input_text:
                raise ValueError("No configuration provided. Please provide a valid JSON string.")

            # Parse input text as JSON
            input_json = json.loads(input_text)

            # Create unique output folder with model name
            timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
            random_generation_id = "".join(random.choices(string.ascii_lowercase, k=4))
            
            # Create model-specific output directory
            model_output_dir = os.path.join(output_dir, model_name)
            output_folder = os.path.join(model_output_dir, f"generation_{timestamp}_{random_generation_id}")
            os.makedirs(output_folder, exist_ok=True)

            # Handle input image - it could be a PIL Image object or base64 data
            input_image_path = os.path.join(output_folder, "input_image.jpg")
            
            # Check if input_image is a base64 string
            if isinstance(input_image, str) and input_image.startswith("data:image"):
                # Convert base64 to PIL Image
                import base64
                import io
                from PIL import Image
                
                # Extract base64 data from data URL
                if input_image.startswith("data:image/jpeg;base64,"):
                    base64_data = input_image.replace("data:image/jpeg;base64,", "")
                elif input_image.startswith("data:image/png;base64,"):
                    base64_data = input_image.replace("data:image/png;base64,", "")
                else:
                    raise ValueError("Unsupported image format. Only JPEG and PNG are supported.")
                
                # Decode base64 and create PIL Image
                image_data = base64.b64decode(base64_data)
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image.save(input_image_path)
                
            else:
                # It's a PIL Image object, save it
                input_image.save(input_image_path)

            # Determine checkpoint directory based on checkpoint parameter
            checkpoint_choice = input_json.get("checkpoint", "480p")
            if checkpoint_choice == "480p":
                actual_checkpoint_dir = "./Wan2.1-I2V-14B-480P"
            elif checkpoint_choice == "720p":
                actual_checkpoint_dir = "./Wan2.1-I2V-14B-720P"
            else:
                # Fallback to default
                actual_checkpoint_dir = checkpoint_dir
                print(f"⚠️ Unknown checkpoint choice '{checkpoint_choice}', using default: {checkpoint_dir}")
            
            print(f"🔧 Using checkpoint directory: {actual_checkpoint_dir}")
            
            # Set generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "input_image": input_image_path,
                "checkpoint_dir": actual_checkpoint_dir,
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

            # Build command line arguments for generate.py
            # Get the Wan2.1 directory path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            wan2_1_dir = os.path.join(project_root, "Wan2.1")
            generate_py_path = os.path.join(wan2_1_dir, "generate.py")
            
            # Build command line arguments for generate.py (following official Wan example)
            cmd_args = [
                "python",
                generate_py_path,
                "--task",
                "i2v-14B",
                "--size",
                "1280*720" if checkpoint_choice == "720p" else "832*480",  # Match checkpoint resolution
                "--ckpt_dir",
                actual_checkpoint_dir,
                "--image",
                input_image_path,
                "--prompt",
                prompt,
                "--save_file",
                os.path.join(output_folder, "output.mp4"),
            ]

            # Note: Following the official Wan example, we use minimal parameters
            # The model will use its default values for other parameters
            # Additional parameters from JSON are logged but not passed to generate.py
            print(f"📝 Additional parameters (not used): {input_json}")

            print(f"Running Wan2.1 i2v generation with command: {' '.join(cmd_args)}")

            # Run the generation with H100 GPU optimization
            env = os.environ.copy()
            env["PYTHONPATH"] = wan2_1_dir
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use primary H100 GPU
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"  # Larger chunks for H100
            env["CUDA_LAUNCH_BLOCKING"] = "0"  # Non-blocking CUDA operations
            env["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8 for H100
            env["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 for faster computation
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True"  # Better memory management
            env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Reduce GPU memory fragmentation

            result = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,     
                stderr=subprocess.PIPE,
                cwd=wan2_1_dir,  # Set working directory to Wan2.1
                env=env,
                text=True,  # Return strings instead of bytes
            )

            # Always log the output for debugging
            print(f"generate.py stdout: {result.stdout}")
            print(f"generate.py stderr: {result.stderr}")
            print(f"generate.py return code: {result.returncode}")

            if result.returncode != 0:
                error_msg = f"Generation failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                return None, error_msg

            # Check if output file was created
            output_video_path = os.path.join(output_folder, "output.mp4")
            if not os.path.exists(output_video_path):
                error_msg = f"Expected output video not found at: {output_video_path}"
                return None, error_msg

            # Return the video path and success message
            success_msg = f"✅ Video generated successfully!\n"
            success_msg += f"📁 Output folder: {output_folder}\n"
            success_msg += f"🎥 Video saved to: {output_video_path}\n"
            success_msg += f"🤖 Model: {model_name}"

            return output_video_path, success_msg

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON configuration: {str(e)}"
            return None, error_msg
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}\n"
            error_msg += f"Traceback: {traceback.format_exc()}"
            return None, error_msg

    # Create Gradio interface
    with gr.Blocks(title="Wan2.1: Image-to-Video Generation (14B Model)") as demo:
        gr.Markdown("# 🎬 Wan2.1 Image-to-Video Generation (14B Model)")
        gr.Markdown("Generate videos from images using the Wan2.1 14B model.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                prompt_input = gr.Textbox(
                    label="📝 Text Prompt",
                    placeholder="Describe the video you want to generate from the image...",
                    lines=3,
                    max_lines=5
                )
                
                negative_prompt_input = gr.Textbox(
                    label="🚫 Negative Prompt",
                    placeholder="Describe what you don't want in the video...",
                    lines=2,
                    max_lines=3
                )
                
                image_input = gr.Image(
                    label="🖼️ Input Image",
                    type="pil",
                    height=300
                )
                
                api_token_input = gr.Textbox(
                    label="🔑 API Token (Optional)",
                    type="password",
                    placeholder="Enter your API token here..."
                )

            with gr.Column(scale=1):
                # Configuration
                config_input = gr.Textbox(
                    label="⚙️ Generation Parameters (JSON)",
                    placeholder='{"frame_num": 81, "sample_steps": 40, "sample_guide_scale": 5.0, "base_seed": 42}',
                    lines=8,
                    max_lines=10,
                    value=json.dumps({
                        "frame_num": 81,
                        "sample_steps": 40,
                        "sample_guide_scale": 5.0,
                        "base_seed": 42,
                        "checkpoint": "480p",
                        "use_prompt_extend": False
                    }, indent=2)
                )

        # Generate button
        generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

        # Output components
        with gr.Row():
            video_output = gr.Video(
                label="🎥 Generated Video",
                height=400
            )
            
            status_output = gr.Textbox(
                label="📊 Status",
                lines=5,
                max_lines=10,
                interactive=False
            )

        # Event handlers
        generate_btn.click(
            fn=_infer,
            inputs=[prompt_input, negative_prompt_input, image_input, config_input, api_token_input],
            outputs=[video_output, status_output]
        )

        # Add some helpful information
        gr.Markdown("""
        ### 📋 Usage Instructions:
        1. **Upload an image** that you want to animate
        2. **Enter a text prompt** describing the desired video
        3. **Optionally add a negative prompt** to avoid unwanted elements
        4. **Configure generation parameters** (or use defaults)
        5. **Click Generate** to create your video

        ### ⚙️ Default Parameters:
        - **Frames**: 81 (4n+1 format for optimal generation)
        - **Steps**: 40 (diffusion sampling steps)
        - **Guidance Scale**: 5.0 (prompt adherence vs creativity)
        - **Seed**: 42 (for reproducible results)

        ### 💡 Tips:
        - Higher frame counts create longer videos
        - More sampling steps improve quality but take longer
        - Higher guidance scale makes the video follow the prompt more closely
        - Use negative prompts to avoid unwanted elements
        """)

    return demo


if __name__ == "__main__":
    # Configuration
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./Wan2.1-I2V-14B")
    output_dir = os.environ.get("OUTPUT_DIR", "./gradio_output")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and launch the interface
    demo = create_gradio_interface(checkpoint_dir, output_dir)
    
    # Launch settings
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))
    
    # Security settings
    allowed_paths = os.environ.get("GRADIO_ALLOWED_PATHS", os.getcwd()).split(",")
    save_dir = os.environ.get("GRADIO_SAVE_DIR", os.path.join(os.getcwd(), "gradio_output"))
    
    print(f"🚀 Starting Wan2.1 i2v 14B server...")
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🌐 Server: {server_name}:{server_port}")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        allowed_paths=allowed_paths,
        show_error=True,
        quiet=False
    ) 