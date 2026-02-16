#!/bin/bash
# Setup script for enhanced audio transcription

echo "=================================================="
echo "Audio Transcription Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✓ pip3 found"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi
echo ""

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg is not installed"
    echo ""
    echo "Please install ffmpeg:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Linux:   sudo apt-get install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
    echo ""
    exit 1
fi

echo "✓ ffmpeg found: $(ffmpeg -version | head -n1)"
echo ""

# Check for API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY environment variable not set"
    echo ""
    echo "To set your API key:"
    echo "  export GEMINI_API_KEY='your_api_key_here'"
    echo ""
    echo "Or add to your ~/.zshrc or ~/.bashrc:"
    echo "  echo 'export GEMINI_API_KEY=\"your_key\"' >> ~/.zshrc"
    echo ""
    echo "Get your API key from: https://aistudio.google.com/app/apikey"
    echo ""
else
    echo "✓ GEMINI_API_KEY environment variable is set"
    echo ""
fi

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "You can now run:"
echo "  python3 transcribe_audio.py"
echo ""
echo "For more information, see README.md"
