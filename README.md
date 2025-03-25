# AI-Poker-Bot

A Texas Hold'em AI bot optimized for macOS with M1/M2 support.

## Features
- Monte Carlo hand evaluation
- Neural network decision model
- Thermal throttling protection
- MPS GPU acceleration

## Requirements
- macOS 13.0+ (Ventura or newer)
- Python 3.11
- Apple Silicon (M1/M2) recommended

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/poker-ai-mac.git
cd poker-ai-mac

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Nvidia ML frameworks (if using external GPU)
# brew install --cask cuda