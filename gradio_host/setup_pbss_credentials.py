#!/usr/bin/env python3
"""
Setup PBSS Credentials Script

This script sets up the AWS credentials and config files needed for PBSS synchronization.
Run this on the server host BEFORE starting any generation jobs.

Usage:
    python setup_pbss_credentials.py
"""

import os
from pathlib import Path

def setup_pbss_credentials():
    """Set up PBSS credentials for s5cmd."""
    
    print("🔐 Setting up PBSS Credentials for s5cmd")
    print("=" * 50)
    
    # Check environment variables
    endpoint = os.environ.get("TEAM_COSMOS_BENCHMARK_ENDPOINT")
    region = os.environ.get("TEAM_COSMOS_BENCHMARK_REGION")
    secret_key = os.environ.get("XIANL_TEAM_COSMOS_BENCHMARK")
    
    print("🔍 Environment Variables:")
    print(f"   Endpoint: {endpoint}")
    print(f"   Region: {region}")
    print(f"   Secret Key: {'*' * 10 if secret_key else 'None'}")
    
    if not all([endpoint, region, secret_key]):
        print("❌ Missing required environment variables!")
        print("💡 Please set:")
        print("   export TEAM_COSMOS_BENCHMARK_ENDPOINT='your_pbss_endpoint'")
        print("   export TEAM_COSMOS_BENCHMARK_REGION='your_pbss_region'")
        print("   export XIANL_TEAM_COSMOS_BENCHMARK='your_secret_key'")
        return False
    
    # Create AWS credentials file (matching your exact format)
    aws_credentials = f"""[team-cosmos-benchmark]
region={region}
aws_access_key_id=team-cosmos-benchmark
aws_secret_access_key={secret_key}
s3=
    endpoint_url={endpoint}
s3api=
    endpoint_url={endpoint}
    payload_signing_enabled=true
"""
    
    # Create AWS config file (matching your exact format)
    aws_config = f"""[profile team-cosmos-benchmark]
aws_access_key_id = team-cosmos-benchmark
aws_secret_access_key = {secret_key}
region = {region}
"""
    
    # Create AWS directory
    aws_dir = Path("/root/.aws")
    try:
        aws_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created AWS directory: {aws_dir}")
    except PermissionError:
        print(f"❌ Permission denied creating {aws_dir}")
        print("💡 Try running with sudo or as root")
        return False
    except Exception as e:
        print(f"❌ Error creating AWS directory: {e}")
        return False
    
    # Write credentials file
    credentials_path = aws_dir / "credentials"
    try:
        with open(credentials_path, "w") as f:
            f.write(aws_credentials)
        print(f"✅ Created credentials file: {credentials_path}")
    except Exception as e:
        print(f"❌ Error writing credentials file: {e}")
        return False
    
    # Write config file
    config_path = aws_dir / "config"
    try:
        with open(config_path, "w") as f:
            f.write(aws_config)
        print(f"✅ Created config file: {config_path}")
    except Exception as e:
        print(f"❌ Error writing config file: {e}")
        return False
    
    # Set proper permissions
    try:
        os.chmod(credentials_path, 0o600)  # Owner read/write only
        os.chmod(config_path, 0o600)       # Owner read/write only
        print("✅ Set proper file permissions (600)")
    except Exception as e:
        print(f"⚠️ Warning: Could not set file permissions: {e}")
    
    # Display created files
    print(f"\n📁 Created files:")
    print(f"   Credentials: {credentials_path}")
    print(f"   Config: {config_path}")
    
    # Test the credentials
    print(f"\n🧪 Testing credentials...")
    test_result = test_credentials()
    
    if test_result:
        print("✅ Credentials setup completed successfully!")
        print("💡 You can now run generation jobs - PBSS sync should work!")
        return True
    else:
        print("⚠️ Credentials created but test failed - check the error above")
        return False

def test_credentials():
    """Test if the credentials work with s5cmd."""
    
    try:
        import subprocess
        
        print("🔄 Testing s5cmd ls command...")
        
        # Test listing the bucket
        cmd = ["s5cmd", "--profile", "team-cosmos-benchmark", "--endpoint-url", "https://pdx.s8k.io", "ls", "s3://evaluation_videos/"]
        print(f"📋 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print(f"   stdout: {result.stdout}")
        if result.stderr:
            print(f"   stderr: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ s5cmd test successful! Credentials are working.")
            return True
        else:
            print("❌ s5cmd test failed. Check the error above.")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ s5cmd test timed out. Check network connectivity.")
        return False
    except FileNotFoundError:
        print("❌ s5cmd not found. Please install s5cmd first.")
        return False
    except Exception as e:
        print(f"❌ Error testing credentials: {e}")
        return False

def main():
    """Main function."""
    print("🚀 PBSS Credentials Setup")
    print("=" * 60)
    
    success = setup_pbss_credentials()
    
    if success:
        print("\n" + "="*60)
        print("✅ Setup completed successfully!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Setup failed!")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit(main()) 