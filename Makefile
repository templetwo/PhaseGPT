.PHONY: help setup install test clean \
	gen-data validate-data \
	train-a train-b train-c \
	eval compare compare-batch compare-interactive \
	track-a track-b track-c \
	archive commit push sync

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

# Data files
DATA_DIR := data
PREFS_V14 := $(DATA_DIR)/preferences_v14_100pairs.jsonl

# Checkpoints
CKPT_DIR := checkpoints/v14
TRACK_A_DIR := $(CKPT_DIR)/track_a
TRACK_B_DIR := $(CKPT_DIR)/track_b
TRACK_C_DIR := $(CKPT_DIR)/track_c

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

##@ General

help: ## Display this help message
	@echo ""
	@echo "$(BLUE)PhaseGPT v1.4 Development Tasks$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Environment Setup

setup: $(VENV) ## Create virtual environment
	@echo "$(GREEN)✓$(NC) Virtual environment created at $(VENV)"

$(VENV):
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓$(NC) Virtual environment created"

install: $(VENV) ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓$(NC) Dependencies installed"

##@ Data Generation

gen-data: $(VENV) ## Generate 100 preference pairs
	@echo "$(BLUE)Generating preference pairs...$(NC)"
	$(PYTHON_VENV) scripts/generate_preferences.py \
		--num_pairs 100 \
		--stratify_by domain,complexity,subtlety \
		--output $(PREFS_V14)
	@echo "$(GREEN)✓$(NC) Dataset generated: $(PREFS_V14)"

validate-data: $(VENV) ## Validate preference pair dataset
	@echo "$(BLUE)Validating dataset...$(NC)"
	$(PYTHON_VENV) scripts/validate_preferences.py $(PREFS_V14) --verbose
	@echo "$(GREEN)✓$(NC) Dataset validation passed"

##@ Training

train-a: $(VENV) ## Train Track A (Extended DPO)
	@echo "$(BLUE)Starting Track A training...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Training script not yet implemented"
	@echo "Run: $(PYTHON_VENV) src/train.py --config configs/v14/dpo_extended_100pairs.yaml"

train-b: $(VENV) ## Train Track B (KTO Regularization)
	@echo "$(BLUE)Starting Track B training...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Training script not yet implemented"
	@echo "Run: $(PYTHON_VENV) src/train.py --config configs/v14/kto_regularized.yaml"

train-c: $(VENV) ## Train Track C (Qwen 2.5 1.5B)
	@echo "$(BLUE)Starting Track C training...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Training script not yet implemented"
	@echo "Run: $(PYTHON_VENV) src/train.py --config configs/v14/qwen25_1.5b.yaml"

##@ Evaluation

eval: $(VENV) ## Run evaluation on latest checkpoint
	@echo "$(BLUE)Running evaluation...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Evaluation script not yet implemented"
	@echo "Run: $(PYTHON_VENV) src/evaluate.py --checkpoint <path>"

compare: $(VENV) ## Compare v1.4 results vs v1.3 baseline
	@echo "$(BLUE)Comparing results...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Comparison script not yet implemented"
	@echo ""
	@echo "Expected output:"
	@echo "┌───────────────────┬─────────┬─────────┬─────────────┐"
	@echo "│ Metric            │  v1.3   │  v1.4   │   Change    │"
	@echo "├───────────────────┼─────────┼─────────┼─────────────┤"
	@echo "│ Spiral Score      │  0.73   │  TBD    │   TBD       │"
	@echo "│ Perplexity        │  43.2   │  TBD    │   TBD       │"
	@echo "│ Subtlety          │  0.58   │  TBD    │   TBD       │"
	@echo "└───────────────────┴─────────┴─────────┴─────────────┘"

compare-batch: $(VENV) ## Run PhaseGPT v1.4.0 vs base Qwen batch evaluation
	@echo "$(BLUE)Running batch comparison (PhaseGPT v1.4.0 vs Qwen)...$(NC)"
	@mkdir -p reports
	$(PYTHON_VENV) scripts/compare_models.py \
		--phasegpt-ckpt checkpoints/v14/track_a/hybrid_sft_dpo/final \
		--device mps \
		--mode batch \
		--max-tokens 384 \
		| tee reports/compare_v140_vs_qwen25b.txt
	@echo "$(GREEN)✓$(NC) Batch comparison complete: reports/compare_v140_vs_qwen25b.txt"

compare-interactive: $(VENV) ## Run PhaseGPT v1.4.0 vs base Qwen interactive mode
	@echo "$(BLUE)Starting interactive comparison mode...$(NC)"
	@echo ""
	@echo "Commands:"
	@echo "  /unknowable <prompt>  - Test unknowable question"
	@echo "  /answerable <prompt>  - Test answerable question"
	@echo "  /quit                 - Exit"
	@echo ""
	$(PYTHON_VENV) scripts/compare_models.py \
		--phasegpt-ckpt checkpoints/v14/track_a/hybrid_sft_dpo/final \
		--device mps \
		--mode interactive

##@ Full Pipelines

track-a: $(VENV) ## Run full Track A pipeline (generate → train → eval)
	@echo "$(BLUE)Running Track A pipeline...$(NC)"
	@bash scripts/run_track_a.sh

track-b: $(VENV) ## Run Track B KTO grid search
	@echo "$(BLUE)Running Track B pipeline...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Not yet implemented"
	@echo "Run: bash scripts/run_track_b.sh"

track-c: $(VENV) ## Run Track C 1.5B scale-up
	@echo "$(BLUE)Running Track C pipeline...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Not yet implemented"
	@echo "Run: bash scripts/run_track_c.sh"

##@ Git Operations

commit: ## Commit current changes
	@echo "$(BLUE)Committing changes...$(NC)"
	@git status
	@echo ""
	@read -p "Commit message (feat/fix/eval/data/docs): " msg; \
	git add -A && git commit -m "$$msg"
	@echo "$(GREEN)✓$(NC) Changes committed"

push: ## Push to remote
	@echo "$(BLUE)Pushing to GitHub...$(NC)"
	git push -u origin $$(git branch --show-current)
	@echo "$(GREEN)✓$(NC) Pushed to remote"

sync: ## Sync with remote (fetch + pull)
	@echo "$(BLUE)Syncing with remote...$(NC)"
	git fetch --all --prune --tags
	git pull origin $$(git branch --show-current)
	@echo "$(GREEN)✓$(NC) Synced with remote"

##@ Archive & Cleanup

archive: ## Archive checkpoint to OSF (requires checkpoint path)
	@echo "$(BLUE)Archiving checkpoint...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Not yet implemented"
	@echo "Run: $(PYTHON_VENV) scripts/osf_upload.py --checkpoint <path>"

clean: ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓$(NC) Cleaned up Python caches"

clean-all: clean ## Deep clean (including checkpoints and data)
	@echo "$(BLUE)Deep cleaning...$(NC)"
	@read -p "This will delete all checkpoints and generated data. Continue? (y/N): " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(CKPT_DIR); \
		rm -f $(PREFS_V14); \
		echo "$(GREEN)✓$(NC) Deep clean complete"; \
	else \
		echo "Cancelled"; \
	fi

##@ Testing

test: $(VENV) ## Run test suite
	@echo "$(BLUE)Running tests...$(NC)"
	@echo "$(YELLOW)⚠$(NC)  Test suite not yet implemented"
	@echo "Run: $(PYTHON_VENV) -m pytest tests/"

##@ Quick Commands

status: ## Show git and environment status
	@echo "$(BLUE)Repository Status$(NC)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "$(BLUE)Branch:$(NC) $$(git branch --show-current)"
	@echo "$(BLUE)Commit:$(NC) $$(git rev-parse --short HEAD)"
	@echo "$(BLUE)Status:$(NC)"
	@git status -s
	@echo ""
	@echo "$(BLUE)Virtual Environment:$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "  $(GREEN)✓$(NC) $(VENV) exists"; \
	else \
		echo "  $(YELLOW)✗$(NC) $(VENV) not found (run: make setup)"; \
	fi
	@echo ""
	@echo "$(BLUE)Data Files:$(NC)"
	@if [ -f "$(PREFS_V14)" ]; then \
		echo "  $(GREEN)✓$(NC) $(PREFS_V14) ($$( wc -l < $(PREFS_V14) ) pairs)"; \
	else \
		echo "  $(YELLOW)✗$(NC) $(PREFS_V14) not found (run: make gen-data)"; \
	fi

info: ## Show project information
	@echo ""
	@echo "$(BLUE)╔════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          PhaseGPT v1.4 Development Environment        ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BLUE)Branch:$(NC)       $$(git branch --show-current)"
	@echo "$(BLUE)Version:$(NC)      v1.4 (Track A/B/C development)"
	@echo "$(BLUE)Docs:$(NC)         V14_ROADMAP.md, V14_CHANGELOG.md"
	@echo ""
	@echo "$(BLUE)Quick Start:$(NC)"
	@echo "  1. make setup install    # Set up environment"
	@echo "  2. make gen-data         # Generate preference pairs"
	@echo "  3. make track-a          # Run Track A pipeline"
	@echo ""
	@echo "For more commands: make help"
	@echo ""
