#!/bin/bash

# Food Health Analyzer - Setup Script
# This script helps set up the application environment

set -e

echo "🍎 Food Health Analyzer - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo "❌ Error: Python 3.8 or higher is required. You have Python $python_version"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Check USDA API Key
echo "🔑 Checking USDA API Key..."
if [ -z "$USDA_API_KEY" ]; then
    echo "⚠️  USDA_API_KEY not set. Using DEMO_KEY (limited to 1000 requests/hour)"
    echo ""
    echo "To get better rate limits:"
    echo "1. Sign up at: https://fdc.nal.usda.gov/api-key-signup.html"
    echo "2. Set your key: export USDA_API_KEY=your_key_here"
    echo ""
else
    echo "✓ USDA_API_KEY is set"
fi
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p examples models
echo "✓ Directories created"
echo ""

# Final message
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     venv\\Scripts\\activate"
else
    echo "     source venv/bin/activate"
fi
echo "  2. Run the app:"
echo "     python app.py"
echo ""
echo "The app will open at: http://127.0.0.1:7860"
echo ""
echo "Happy analyzing! 🎉"
