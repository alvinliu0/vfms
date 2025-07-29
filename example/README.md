# Example Scripts

This folder contains example scripts for using the VFMS video generation framework.

## 🚀 Quick Start Scripts

### `scripts/wan2.1_t2v_1.3B_singleGPU/generate.py`

A script that handles CLI arguments for basic inputs (API token, endpoint, generation parameters) and calls the client's `generate_video` or `batch_generate` method for video generation. Supports both single and batch generation modes.

#### Features

- **CLI Argument Handling**: Parses basic inputs (API token, endpoint, generation args)
- **Client Method Integration**: Calls the client's `generate_video` or `batch_generate` method directly
- **Single & Batch Generation**: Supports both single video and batch video generation
- **Secure CLI Arguments**: API token and endpoint passed via command-line (no hardcoded credentials)
- **Parameter Customization**: All generation parameters can be set via CLI arguments
- **Connection Testing**: Built-in connection testing with `--test-connection`
- **Code Separation**: Clean separation between argument parsing and generation logic

#### Usage

1. **Run the Script**:
   ```bash
   # Option 1: From the model-specific directory
   cd example/scripts/wan2.1_t2v_1.3B_singleGPU
   python generate.py --token "your_token" --endpoint "your_endpoint" --prompt "your prompt"
   
   # Option 2: From the vfms root directory
   python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --prompt "your prompt"
   ```

#### Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--token` | ✅ Yes | API token for authentication | None |
| `--endpoint` | ✅ Yes | Server endpoint URL | None |
| `--prompt` | ❌ No* | Text prompt for single video generation | None |
| `--batch-prompts` | ❌ No* | Comma-separated prompts for batch generation | None |
| `--batch-prompts-file` | ❌ No* | File containing prompts (one per line) for batch generation | None |
| `--negative-prompt` | ❌ No | Negative prompt for single generation | "blurry, low quality, distorted, ugly" |
| `--batch-negative-prompts` | ❌ No | Comma-separated negative prompts for batch generation | None |
| `--batch-negative-prompts-file` | ❌ No | File containing negative prompts (one per line) for batch generation | None |
| `--output-dir` | ❌ No | Output directory for results | "./results" |
| `--test-connection` | ❌ No | Test connection only | False |
| `--frame-num` | ❌ No | Number of frames (8-32) | 16 |
| `--sample-steps` | ❌ No | Inference steps (20-100) | 50 |
| `--sample-guide-scale` | ❌ No | Guidance scale (1.0-20.0) | 7.5 |
| `--base-seed` | ❌ No | Random seed for reproducibility | 42 |
| `--width` | ❌ No | Video width (must be divisible by 8) | 832 |
| `--height` | ❌ No | Video height (must be divisible by 8) | 480 |
| `--use-prompt-extend` | ❌ No | Use prompt extension | False |

*Either `--prompt`, `--batch-prompts`, or `--batch-prompts-file` is required (mutually exclusive)

#### Usage Examples

```bash
# Single generation with required arguments
python generate.py --token "your_token" --endpoint "your_endpoint" --prompt "A beautiful sunset"

# Batch generation with multiple prompts (comma-separated)
python generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts "A beautiful sunset,A stormy night,A peaceful morning"

# Batch generation with file-based prompts (handles commas in prompts)
python generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts-file "prompts_example.txt"

# Batch generation with custom negative prompts (comma-separated)
python generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts "prompt1,prompt2" --batch-negative-prompts "neg1,neg2"

# Batch generation with file-based negative prompts
python generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts-file "prompts_example.txt" --batch-negative-prompts-file "negative_prompts_example.txt"

# Test connection only
python generate.py --token "your_token" --endpoint "your_endpoint" --test-connection

# With custom parameters
python generate.py --token "your_token" --endpoint "your_endpoint" --prompt "A beautiful sunset" --frame-num 24 --sample-steps 75 --width 1024 --height 576

# Batch generation with custom parameters
python generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts "prompt1,prompt2,prompt3" --frame-num 24 --sample-steps 75

# With custom output directory
python generate.py --token "your_token" --endpoint "your_endpoint" --prompt "A beautiful sunset" --output-dir "./my_results"
```

#### Security Benefits

- **No Hardcoded Credentials**: API tokens and endpoints are never stored in the script
- **Command History Safe**: Credentials don't persist in shell history (when using quotes)
- **Environment Variable Compatible**: Can use environment variables for sensitive data
- **Git Safe**: No risk of accidentally committing credentials to version control

#### Tips

1. **Use Environment Variables**: For extra security, use environment variables:
   ```bash
   export API_TOKEN="your_token"
   export ENDPOINT_URL="your_endpoint"
   python generate.py --token "$API_TOKEN" --endpoint "$ENDPOINT_URL" --prompt "A beautiful sunset"
   ```

2. **Test Connection First**: Always test the connection before running generation:
   ```bash
   python generate.py --token "your_token" --endpoint "your_endpoint" --test-connection
   ```

3. **Start Simple**: Use the default parameters first, then experiment with custom values
4. **Be Patient**: Generation takes 1-2 hours
5. **Check Logs**: Monitor the output for progress and errors
6. **Use Seeds**: Set `--base-seed` for reproducible results

#### Example Output

**Single Generation:**
```
🎬 Wan2.1 Video Generation Script
==================================================
🔗 Connecting to endpoint: https://your-endpoint.com
🧪 Testing connection...
✅ Connection successful!

📋 Single Generation Parameters:
   Prompt: A beautiful sunset over the ocean, cinematic lighting, 4K quality
   Negative Prompt: blurry, low quality, distorted, ugly
   Frames: 16
   Steps: 50
   Guidance Scale: 7.5
   Resolution: 832x480
   Seed: 42

🚀 Starting single video generation...
⏳ This may take 1-2 hours. Please be patient...

✅ Generation completed successfully!
🎥 Video saved to: ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143022/generated_video.mp4
📁 Full output directory: ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143022
📄 Generated files:
   - generated_video.mp4
   - generation_params.json
   - prompt.txt
   - negative_prompt.txt
   - status.txt
   - request_metadata.json

🎉 Script completed successfully!
```

**Batch Generation:**
```
🎬 Wan2.1 Video Generation Script
==================================================
🔗 Connecting to endpoint: https://your-endpoint.com
🧪 Testing connection...
✅ Connection successful!

📋 Batch Generation Parameters:
   Number of videos: 3
   Frames: 16
   Steps: 50
   Guidance Scale: 7.5
   Resolution: 832x480
   Seed: 42

📝 Prompts:
   1. A beautiful sunset over the ocean
   2. A stormy night with lightning
   3. A peaceful morning in the forest

🚫 Negative Prompts:
   1. blurry, low quality, distorted, ugly
   2. blurry, low quality, distorted, ugly
   3. blurry, low quality, distorted, ugly

🚀 Starting batch generation of 3 videos...
⏳ This may take several hours. Please be patient...

--- Generating video 1/3 ---
🎬 Generating video with prompt: 'A beautiful sunset over the ocean'
📁 Output folder: ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143022
🤖 Model: wan2.1_t2v_1.3B_singleGPU
✅ Generation completed!

--- Generating video 2/3 ---
🎬 Generating video with prompt: 'A stormy night with lightning'
📁 Output folder: ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143045
🤖 Model: wan2.1_t2v_1.3B_singleGPU
✅ Generation completed!

--- Generating video 3/3 ---
🎬 Generating video with prompt: 'A peaceful morning in the forest'
📁 Output folder: ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143108
🤖 Model: wan2.1_t2v_1.3B_singleGPU
✅ Generation completed!

✅ Batch generation completed!
🎥 Successfully generated: 3/3 videos

📁 Generated videos:
   1. ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143022/generated_video.mp4
   2. ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143045/generated_video.mp4
   3. ./results/wan2.1_t2v_1.3B_singleGPU/generation_20250729_143108/generated_video.mp4

🎉 Script completed successfully!
```

#### Troubleshooting

- **Connection Failed**: Check your API token and endpoint URL
- **Generation Failed**: Check server logs and ensure the model is properly loaded
- **Timeout**: Generation can take 1-2 hours, be patient
- **Memory Issues**: Reduce `frame_num` or `sample_steps` if needed

## 🔮 Future Scripts

This folder will contain additional generation scripts for:
- Different model variants (e.g., different model sizes)
- Different generation tasks (e.g., image-to-video, video-to-video)
- Different optimization settings
- Batch processing workflows

Each script will follow the same pattern: simple wrapper around the main client functionality.

---

For more advanced usage, see the main documentation in the project root. 