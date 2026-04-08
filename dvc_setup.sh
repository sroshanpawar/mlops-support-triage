#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DVC Setup Script
# Initializes DVC tracking for the training dataset
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  🔧 DVC (Data Version Control) Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Step 1: Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC is not installed. Installing..."
    pip install dvc
    echo "✅ DVC installed successfully"
else
    echo "✅ DVC is already installed ($(dvc --version))"
fi

# Step 2: Check if Git is initialized
if [ ! -d ".git" ]; then
    echo "📂 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository found"
fi

# Step 3: Initialize DVC
if [ ! -d ".dvc" ]; then
    echo "📂 Initializing DVC..."
    dvc init
    echo "✅ DVC initialized"
else
    echo "✅ DVC already initialized"
fi

# Step 4: Track the training dataset with DVC
echo ""
echo "📊 Adding training data to DVC tracking..."
dvc add training/data/training_data.json
echo "✅ training/data/training_data.json is now tracked by DVC"

# Step 5: Configure local remote storage (for demo)
echo ""
echo "📦 Setting up local DVC remote storage..."
mkdir -p /tmp/dvc-storage
dvc remote add -d local-storage /tmp/dvc-storage 2>/dev/null || \
    dvc remote modify local-storage url /tmp/dvc-storage
echo "✅ Local DVC remote configured at /tmp/dvc-storage"

# Step 6: Push data to remote
echo ""
echo "🚀 Pushing data to DVC remote..."
dvc push
echo "✅ Data pushed to remote storage"

# Step 7: Add DVC files to Git
echo ""
echo "📝 Adding DVC metadata to Git..."
git add training/data/training_data.json.dvc training/data/.gitignore .dvc/ .dvcignore
echo "✅ DVC files staged for Git commit"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✨ DVC Setup Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Common DVC Commands:"
echo "    dvc status        — Check data status"
echo "    dvc push          — Push data to remote"
echo "    dvc pull          — Pull data from remote"
echo "    dvc diff          — Show changes in data"
echo "    dvc repro         — Reproduce pipeline"
echo ""
echo "  To track new data files:"
echo "    dvc add <filepath>"
echo ""
