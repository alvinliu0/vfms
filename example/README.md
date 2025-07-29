# Example Scripts

This folder contains example scripts for using the VFMS video generation framework.

## 🚀 Quick Start Scripts

### `scripts/wan2.1_t2v_1.3B_singleGPU/generate.py`

A script that handles CLI arguments for basic inputs (API token, endpoint, generation parameters) and calls the client's `generate_video` or `batch_generate` method for video generation. Supports both single and batch generation modes.

### `scripts/wan2.1_t2v_14B_singleGPU/generate.py`

A script that handles CLI arguments for basic inputs (API token, endpoint, generation parameters) and calls the client's `generate_video` or `batch_generate` method for video generation using the 14B model. Supports both single and batch generation modes.

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
# Single generation with required arguments (1.3B model)
python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --prompt "A beautiful sunset"

# Single generation with required arguments (14B model)
python example/scripts/wan2.1_t2v_14B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --prompt "A beautiful sunset"

# Batch generation with multiple prompts (comma-separated) - 1.3B model
python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts "A beautiful sunset,A stormy night,A peaceful morning"

# Batch generation with file-based negative prompts - 1.3B model
python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --batch-prompts-file "example\scripts\wan2.1_t2v_1.3B_singleGPU\prompts_example.txt" --batch-negative-prompts-file "example\scripts\wan2.1_t2v_1.3B_singleGPU\negative_prompts_example.txt"

# Test connection only (1.3B model)
python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --test-connection

# With custom parameters (1.3B model)
python example/scripts/wan2.1_t2v_1.3B_singleGPU/generate.py --token "your_token" --endpoint "your_endpoint" --prompt "A beautiful sunset" --frame-num 24 --sample-steps 75 --width 1024 --height 576
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