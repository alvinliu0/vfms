# VFMS - Video Generation Framework

A command-line video generation framework built around the Wan2.1 text-to-video model, featuring a client-server architecture with secure API token handling.

## 🚀 Features

- **Text-to-Video Generation**: Generate high-quality videos from text prompts using Wan2.1 1.3B model
- **Command-Line Interface**: Full CLI support for automation and scripting
- **Client-Server Architecture**: Separate host and client applications for flexible deployment
- **Secure API Token Handling**: Command-line based token management to prevent credential leaks
- **Batch Processing**: Support for generating multiple videos in sequence
- **Parameter Customization**: Fine-tune generation parameters via JSON configuration

## 📁 Project Structure

```
vfms/
├── gradio_host/          # Gradio server application
│   └── wan2.1_t2v_1.3B_singleGPU_host.py
├── gradio_client/        # CLI client application for API calls
│   └── wan2.1_t2v_1.3B_singleGPU_client.py
├── gradio_output/        # Output directory for generated videos
├── Wan2.1/              # Wan2.1 model repository (submodule)
└── README.md            # This file
```

## 🔐 Security Features

### API Token Management

This repository implements secure API token handling to prevent credential leaks:

- **Command-line Priority**: API tokens are primarily provided via command-line arguments
- **File Fallback**: Optional fallback to `api_token.txt` file for convenience
- **Gitignore Protection**: Sensitive files are automatically excluded from version control
- **No Hardcoded Credentials**: No API keys or tokens are stored in the codebase

### Recommended Usage

```bash
# Preferred: Use command-line argument (most secure)
python gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py \
  --token "your_api_token_here" \
  --prompt "A beautiful sunset over the ocean"

# Alternative: Use token file (less secure but convenient)
echo "your_api_token_here" > api_token.txt
python gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py \
  --prompt "A beautiful sunset over the ocean"
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: H100 80GB)
- Wan2.1 model checkpoints

### Setup

1. **Clone the repository with submodules**:
   ```bash
   git clone --recursive <repository-url>
   cd vfms
   ```

2. **Update existing repository** (if already cloned):
   ```bash
   # If you have local changes, stash them first
   git stash
   git pull origin main
   git submodule update --init --recursive --remote
   git submodule sync --recursive
   git stash pop  # Restore your changes
   ```

3. **Install dependencies**:
   ```bash
   pip install -r Wan2.1/requirements.txt
   pip install gradio gradio-client requests
   ```

4. **Download Wan2.1 model** (if not already present):
   ```bash
   # Follow Wan2.1 installation instructions
   # Ensure model checkpoints are available in Wan2.1/Wan2.1-T2V-1.3B/
   ```

## 🚀 Usage

### Starting the Host Server

```bash
# Set environment variables
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=8080
export GRADIO_SAVE_DIR="/path/to/output/directory"

# Start the Gradio host
python gradio_host/wan2.1_t2v_1.3B_singleGPU_host.py
```

The server will be available at `http://localhost:8080`

### Using the CLI Client

#### Basic Usage

```bash
# Generate a video with command-line token
python gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py \
  --token "your_api_token" \
  --prompt "A beautiful sunset over the ocean" \
  --endpoint "http://localhost:8080"
```

#### Advanced Usage

```bash
# With custom parameters
python gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py \
  --token "your_api_token" \
  --prompt "A beautiful sunset over the ocean" \
  --negative-prompt "blurry, low quality" \
  --parameters '{"frame_num": 16, "sample_steps": 50, "sample_guide_scale": 7.5}' \
  --output-dir "./my_results"
```

#### Test Connection

```bash
# Test connection without generating video
python gradio_client/wan2.1_t2v_1.3B_singleGPU_client.py \
  --token "your_api_token" \
  --test-connection
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--token` | API token for authentication (recommended) | None |
| `--endpoint` | Server endpoint URL | `http://localhost:8080` |
| `--prompt` | Text prompt for video generation | Required |
| `--negative-prompt` | Negative prompt to avoid elements | `""` |
| `--parameters` | JSON string of generation parameters | None |
| `--output-dir` | Output directory for results | `./results` |
| `--test-connection` | Test connection only | False |

### Generation Parameters

Common parameters that can be passed via `--parameters`:

```json
{
  "frame_num": 16,
  "sample_steps": 50,
  "sample_guide_scale": 7.5,
  "base_seed": 42,
  "width": 832,
  "height": 480,
  "use_prompt_extend": false
}
```

## 🔧 Configuration

### Environment Variables

The host server can be configured via environment variables:

- `GRADIO_SERVER_NAME`: Server hostname (default: "0.0.0.0")
- `GRADIO_SERVER_PORT`: Server port (default: 8080)
- `GRADIO_SAVE_DIR`: Output directory for generated videos
- `GRADIO_ALLOWED_PATHS`: Comma-separated list of allowed file paths
- `GRADIO_CONCURRENCY_LIMIT`: Maximum concurrent requests (default: 1)
- `GRADIO_MAX_QUEUE_SIZE`: Maximum queue size (default: 20)

### Model Configuration

The system expects the Wan2.1 model to be available in the project structure:
- Model code: `Wan2.1/`
- Checkpoints: `Wan2.1/Wan2.1-T2V-1.3B/`

## 📊 Output

Generated videos are saved with the following structure:

```
output_directory/
├── wan2.1_t2v_1.3B_singleGPU/          # Model-specific directory
│   ├── generation_YYYY-MM-DD_HH-MM-SS_xxxx/
│   │   ├── generated_video.mp4          # Generated video
│   │   ├── generation_params.json       # Generation parameters
│   │   ├── prompt.txt                   # Input prompt
│   │   ├── negative_prompt.txt          # Negative prompt
│   │   ├── status.txt                   # Generation status
│   │   └── request_metadata.json        # Client request metadata
│   └── generation_YYYY-MM-DD_HH-MM-SS_yyyy/
│       └── ...
└── other_model_name/                     # Other models (if any)
    └── ...
```

This structure ensures:
- **Model Organization**: Each model has its own directory
- **Timestamp-based Folders**: Each generation gets a unique timestamp folder
- **Complete Metadata**: All generation parameters and status are preserved
- **Easy Navigation**: Clear hierarchy for multiple generations

## 🔒 Security Best Practices

1. **Never commit API tokens**: Use command-line arguments or environment variables
2. **Use secure tokens**: Generate strong, unique tokens for each deployment
3. **Rotate tokens regularly**: Update API tokens periodically
4. **Monitor access**: Log and monitor API usage
5. **Use HTTPS**: Always use HTTPS in production environments

## 🐛 Troubleshooting

### Common Issues

1. **Connection refused**: Ensure the host server is running
2. **Model not found**: Verify Wan2.1 model is properly installed
3. **CUDA out of memory**: Reduce batch size or use smaller model
4. **Invalid token**: Check API token format and permissions
5. **Generation timeout**: Video generation can take 1-2 hours, be patient

### Debug Mode

Enable detailed logging by setting environment variables:

```bash
export GRADIO_SHOW_ERROR=true
export GRADIO_DEBUG=true
```

### Performance Notes

- **Generation Time**: Video generation typically takes 1-2 hours
- **GPU Memory**: Requires significant GPU memory (H100 80GB recommended)
- **Network Timeout**: Client is configured with 2-hour timeout for long generations

## 📝 License

This project is licensed under the same license as the Wan2.1 model. Please refer to the Wan2.1 repository for license details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Wan2.1 documentation
3. Open an issue on GitHub

---

**Note**: This framework is designed for research and development purposes. Please ensure compliance with all applicable laws and regulations when using AI-generated content.