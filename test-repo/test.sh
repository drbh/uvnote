#!/bin/bash
# Simple integration test for uvnote

set -e

echo "=== uvnote Integration Test ==="

# Save current directory
TEST_DIR=$(pwd)
UVNOTE_DIR=$(dirname "$TEST_DIR")

# Create and activate virtual environment with uv
echo "Setting up test environment..."
rm -rf .venv
uv venv
source .venv/bin/activate

# Install uvnote in development mode
echo "Installing uvnote..."
uv pip install -e "$UVNOTE_DIR"

# Clean any existing outputs
rm -rf .uvnote site/

echo "Building report.md..."
uvnote build report.md

echo "Checking outputs..."

# Check that HTML was generated
if [ ! -f "site/report.html" ]; then
    echo "ERROR: site/report.html not generated"
    exit 1
fi

# Check that cache was created
if [ ! -d ".uvnote/cache" ]; then
    echo "ERROR: .uvnote/cache not created"
    exit 1
fi

# Check that artifacts were generated
if [ ! -d "site/artifacts" ]; then
    echo "ERROR: site/artifacts not created"
    exit 1
fi

# Check specific artifacts exist
expected_artifacts=(
    "site/artifacts/plot/scatter_plot.png"
    "site/artifacts/plot/histogram.png"
    "site/artifacts/save_data/data.csv"
    "site/artifacts/save_data/summary.txt"
)

for artifact in "${expected_artifacts[@]}"; do
    if [ ! -f "$artifact" ]; then
        echo "WARNING: Expected artifact not found: $artifact"
    else
        echo "✓ Found: $artifact"
    fi
done

# Test single cell execution
echo "Testing single cell execution..."
uvnote run report.md --cell generate_data

# Test cache hit
echo "Testing cache (should be fast)..."
time uvnote build report.md

# # Test clean
# echo "Testing clean..."
# uvnote clean --all

# if [ -d ".uvnote" ] || [ -d "site" ]; then
#     echo "ERROR: Clean did not remove directories"
#     exit 1
# fi

echo "✅ All tests passed!"
echo "Integration test completed successfully."

# Cleanup
echo "Cleaning up test environment..."
deactivate
rm -rf .venv