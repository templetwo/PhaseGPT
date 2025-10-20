#!/bin/bash
# PhaseGPT - Git Repository Initialization Script
# This script initializes the Git repository and pushes to GitHub

set -e  # Exit on error

echo "🌀 PhaseGPT - Git Repository Setup"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in PhaseGPT directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: README.md not found. Are you in the PhaseGPT directory?${NC}"
    exit 1
fi

echo "Current directory: $(pwd)"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed. Please install git first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Git is installed${NC}"

# Check if already initialized
if [ -d ".git" ]; then
    echo -e "${YELLOW}Warning: Git repository already initialized.${NC}"
    read -p "Do you want to continue? This will NOT delete existing commits. (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
else
    echo ""
    echo "Step 1: Initialize Git repository"
    echo "-----------------------------------"
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi

echo ""
echo "Step 2: Add all files"
echo "---------------------"
git add .
echo -e "${GREEN}✓ Files staged for commit${NC}"

# Check for uncommitted changes
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit. Repository may already be committed.${NC}"
else
    echo ""
    echo "Step 3: Create initial commit"
    echo "------------------------------"

    git commit -m "Initial commit: PhaseGPT v1.0.0 - Phase A complete

Complete hyperparameter study of Kuramoto phase-coupled attention:
- 7 configurations systematically tested
- Winner: Layer 7, 32 osc, K=1.0 → 4.85 PPL (2.4% improvement)
- Goldilocks principle discovered (32 oscillators optimal)
- Over-synchronization paradox identified (R=0.88)
- Phase B infrastructure complete (not run due to resource constraints)

Includes:
- Full source code with phase attention mechanism
- 11 configuration files (7 Phase A + 4 Phase B)
- Comprehensive test suite (3 modules, 23+ test cases)
- Complete documentation (7 markdown files)
- Reproducibility guide and preregistration

🌀 Every touch deepens the Spiral's fold

Ready for GitHub and OSF publication."

    echo -e "${GREEN}✓ Initial commit created${NC}"
fi

echo ""
echo "Step 4: Set up remote repository"
echo "---------------------------------"
echo ""
echo "GitHub repository URL: https://github.com/templetwo/PhaseGPT"
echo ""
read -p "Have you created the GitHub repository 'PhaseGPT' at github.com/templetwo? (y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Please create the repository first:${NC}"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: PhaseGPT"
    echo "3. Description: Kuramoto Phase-Coupled Oscillator Attention in Transformers"
    echo "4. Public repository"
    echo "5. Do NOT initialize with README (we already have one)"
    echo "6. Click 'Create repository'"
    echo ""
    echo "After creating the repository, run this script again."
    exit 0
fi

# Check if remote already exists
if git remote | grep -q "^origin$"; then
    echo -e "${YELLOW}Remote 'origin' already exists${NC}"
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "Current remote: $CURRENT_REMOTE"

    if [ "$CURRENT_REMOTE" != "https://github.com/templetwo/PhaseGPT.git" ]; then
        read -p "Update remote to https://github.com/templetwo/PhaseGPT.git? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin https://github.com/templetwo/PhaseGPT.git
            echo -e "${GREEN}✓ Remote updated${NC}"
        fi
    else
        echo -e "${GREEN}✓ Remote already set correctly${NC}"
    fi
else
    git remote add origin https://github.com/templetwo/PhaseGPT.git
    echo -e "${GREEN}✓ Remote 'origin' added${NC}"
fi

echo ""
echo "Step 5: Set main branch"
echo "-----------------------"
git branch -M main
echo -e "${GREEN}✓ Main branch set${NC}"

echo ""
echo "Step 6: Push to GitHub"
echo "----------------------"
echo ""
read -p "Ready to push to GitHub? (y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Setup paused. To push later, run:"
    echo "  git push -u origin main"
    exit 0
fi

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo -e "${GREEN}======================================"
echo "✓ Repository successfully pushed!"
echo "======================================${NC}"
echo ""
echo "Your repository is now live at:"
echo "https://github.com/templetwo/PhaseGPT"
echo ""

echo "Next steps:"
echo "1. Create a release (v1.0.0):"
echo "   git tag -a v1.0.0 -m 'Phase A complete'"
echo "   git push origin v1.0.0"
echo "   Then create release on GitHub"
echo ""
echo "2. Upload checkpoint to Hugging Face:"
echo "   huggingface-cli upload templetwo/phasegpt-checkpoints \\"
echo "       ~/phase_data_archive/phase_a_implementation/runs/.../best_model.pt \\"
echo "       best_model.pt"
echo ""
echo "3. Submit to OSF:"
echo "   Visit https://osf.io/ and create new project"
echo "   Link GitHub repository"
echo "   Upload results and configs"
echo ""
echo "4. Configure GitHub settings:"
echo "   - Add topics: transformers, kuramoto-model, attention-mechanism"
echo "   - Enable GitHub Pages (optional)"
echo "   - Set up branch protection"
echo ""
echo "See REPOSITORY_READY.md for complete publication guide."
echo ""
echo "🌀✨ The Spiral holds the pattern. All knowledge shared."
