#!/bin/bash
# Build documentation locally for testing

set -e

echo "Building PyRadtran documentation..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e .
pip install -r docs/requirements.txt

# Build documentation
echo "Building documentation..."
cd docs
make clean
make html

echo "Documentation built successfully!"
echo "Open docs/build/html/index.html in your browser to view the documentation."

# Optionally open in browser (uncomment if desired)
# python -m webbrowser docs/build/html/index.html
