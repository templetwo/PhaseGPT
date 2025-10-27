# PhaseGPT Development Workflow

This document describes the git workflow and branching strategy for PhaseGPT development.

## Branch Structure

```
main (stable, production-ready)
├── v1.0.0 (tag) - Phase A complete
├── v1.3 (latest stable) - DPO training complete
│
claude/session-* (ephemeral, session-based)
├── claude/session-011CUYNJfTX4Ame2nnowx7FL (completed)
│
claude/v14-dev-* (active research)
├── claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL (current)
│   ├── Track A: Extended DPO experiments
│   ├── Track B: KTO regularization experiments
│   └── Track C: Qwen 2.5 1.5B scale-up
```

## Branch Naming Convention

Due to repository security settings, all pushable branches must follow:

**Pattern:** `claude/<descriptive-name>-<session-id>`

**Examples:**
- `claude/session-011CUYNJfTX4Ame2nnowx7FL` - General development
- `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL` - v1.4 research track
- `claude/bugfix-eval-011CUYNJfTX4Ame2nnowx7FL` - Bug fix branch

**Session ID:** `011CUYNJfTX4Ame2nnowx7FL` (current session)

---

## Workflow Stages

### 1. Active Development (Current: v1.4)

**Branch:** `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL`
**Status:** Active experimentation
**Purpose:** Test new ideas, run experiments, iterate rapidly

**Commands:**
```bash
# Switch to v1.4 dev branch
git checkout claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL

# Make changes, run experiments
# ... edit files ...

# Commit your work
git add .
git commit -m "feat: Add Track A extended DPO training run"

# Push to GitHub
git push -u origin claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
```

### 2. Experimentation Guidelines

**Commit Early, Commit Often:**
- Use semantic commit prefixes:
  - `feat:` - New features or experiments
  - `fix:` - Bug fixes
  - `eval:` - Evaluation results
  - `data:` - Dataset changes
  - `docs:` - Documentation updates
  - `refactor:` - Code refactoring
  - `perf:` - Performance improvements

**Example Commits:**
```bash
git commit -m "feat: Implement curriculum learning for Track A"
git commit -m "eval: Track B run 3, lambda=0.1, perplexity=39.8"
git commit -m "data: Add 50 new preference pairs, stratified by domain"
git commit -m "fix: Correct KTO loss calculation in training loop"
```

### 3. Experiment Tracking

All experiments should be logged in `V14_CHANGELOG.md`:

```bash
# After completing an experiment, document it
# Edit V14_CHANGELOG.md with results

git add V14_CHANGELOG.md
git commit -m "eval: Document Track A run 5 results"
git push
```

### 4. Checkpointing Best Models

When you achieve a significant milestone:

```bash
# Save checkpoint to OSF
python scripts/osf_upload.py --checkpoint checkpoints/v14/track_a/run_5.pt

# Document in changelog
# Update V14_CHANGELOG.md with checkpoint details

git add V14_CHANGELOG.md
git commit -m "checkpoint: Archive Track A run 5 (Spiral Score 0.87)"
git push
```

### 5. Creating Pull Requests

When a track is complete and ready to merge to main:

**Option A: GitHub UI**
1. Visit: https://github.com/templetwo/PhaseGPT/pull/new/claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
2. Title: `Merge v1.4 Track A: Extended DPO → main`
3. Description:
   ```markdown
   ## Summary
   Integrates Track A extended DPO training (100 preference pairs)

   ## Results
   - Spiral Score: 0.87 (+19% vs. v1.3)
   - Perplexity: 44.1 (maintained fluency)
   - Subtlety: 0.68 (+17% vs. v1.3)

   ## Artifacts
   - Checkpoint: OSF DOI:10.XXXX/XXXXX
   - Logs: experiments/v14/track_a/
   - Config: configs/v14/dpo_extended_100pairs.yaml

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   ```
4. Create PR and request review

**Option B: GitHub CLI**
```bash
gh pr create \
  --title "Merge v1.4 Track A: Extended DPO → main" \
  --body "$(cat V14_CHANGELOG.md)" \
  --base main \
  --head claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
```

### 6. Multi-Computer Sync

**Switching between Linux and MacBook Pro:**

```bash
# On MacBook Pro: Pull latest v1.4 work
cd ~/Projects/PhaseGPT
git fetch --all --prune --tags
git checkout claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
git pull origin claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL

# Make changes, commit, push
git add .
git commit -m "feat: Run Track B experiment on Mac GPU"
git push

# On Linux: Pull Mac's work
git fetch --all
git pull origin claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
```

**See [SYNC_INSTRUCTIONS.md](SYNC_INSTRUCTIONS.md) for detailed sync guide.**

---

## Workflow Examples

### Example 1: Running Track A Experiment

```bash
# 1. Ensure you're on v1.4 dev branch
git checkout claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL

# 2. Run training
python train.py --config configs/v14/dpo_extended_100pairs.yaml

# 3. Document results in changelog
nano V14_CHANGELOG.md  # Add experiment entry

# 4. Commit and push
git add V14_CHANGELOG.md
git commit -m "eval: Track A run 1 - Spiral Score 0.82, PPL 44.5"
git push

# 5. Archive checkpoint if significant
python scripts/osf_upload.py --checkpoint checkpoints/v14/track_a/run_1.pt
```

### Example 2: Implementing New Feature

```bash
# 1. Create feature branch (if major change)
git checkout -b claude/feature-kto-annealing-011CUYNJfTX4Ame2nnowx7FL

# 2. Implement feature
nano src/training/kto_loss.py  # Add lambda annealing

# 3. Test it
python tests/test_kto_annealing.py

# 4. Commit and push
git add src/training/kto_loss.py tests/test_kto_annealing.py
git commit -m "feat: Add KTO lambda annealing schedule"
git push -u origin claude/feature-kto-annealing-011CUYNJfTX4Ame2nnowx7FL

# 5. Merge back to v1.4 dev when ready
git checkout claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
git merge claude/feature-kto-annealing-011CUYNJfTX4Ame2nnowx7FL
git push
```

### Example 3: Preparing v1.4 Release

```bash
# 1. Finalize documentation
nano V14_REPORT.md  # Write final report
nano V14_CHANGELOG.md  # Clean up changelog

# 2. Archive all significant checkpoints
python scripts/osf_upload.py --batch checkpoints/v14/best_models/

# 3. Commit final state
git add V14_REPORT.md V14_CHANGELOG.md
git commit -m "docs: Finalize v1.4 documentation for release"
git push

# 4. Create pull request to main
gh pr create \
  --title "Release v1.4: Extended DPO, KTO, and Scale-up" \
  --body-file V14_REPORT.md \
  --base main \
  --head claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL

# 5. After PR merge, tag the release
git checkout main
git pull
git tag -a v1.4.0 -m "PhaseGPT v1.4: Extended DPO + KTO + Scale-up"
git push origin v1.4.0
```

---

## Branch Lifecycle

### Active Branches
- `main` - Stable, production-ready code
- `claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL` - Current v1.4 research

### Archived Branches (on GitHub)
- `claude/session-011CUYNJfTX4Ame2nnowx7FL` - Initial setup work
  - Status: Can be deleted after sync instructions merge to main
  - Contains: SYNC_INSTRUCTIONS.md (already merged)

### Future Branches
- `claude/v15-dev-<session-id>` - Next major version
- `claude/bugfix-*-<session-id>` - Critical bug fixes
- `claude/hotfix-*-<session-id>` - Emergency hotfixes for main

---

## Best Practices

### DO:
✅ Commit frequently with descriptive messages
✅ Push to GitHub daily (enables cross-computer sync)
✅ Document experiments in V14_CHANGELOG.md
✅ Archive significant checkpoints to OSF
✅ Use semantic commit prefixes (feat, fix, eval, etc.)
✅ Keep main branch stable (only merge completed work)

### DON'T:
❌ Push directly to main (use PRs instead)
❌ Commit large binary files (use .gitignore, OSF for checkpoints)
❌ Force push (`git push --force`) without extreme caution
❌ Merge without documenting results in changelog
❌ Delete remote branches without backing up checkpoints

---

## Troubleshooting

### Issue: Cannot push to branch
**Error:** `403 Forbidden`

**Solution:** Ensure branch name matches `claude/*-011CUYNJfTX4Ame2nnowx7FL` pattern

```bash
# Check current branch
git branch --show-current

# If incorrect, rename it
git branch -m old-name claude/new-name-011CUYNJfTX4Ame2nnowx7FL
git push -u origin claude/new-name-011CUYNJfTX4Ame2nnowx7FL
```

### Issue: Diverged branches
**Error:** `Your branch and 'origin/branch' have diverged`

**Solution:** Pull and rebase
```bash
git fetch origin
git rebase origin/claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
git push
```

### Issue: Merge conflicts
**Solution:** Resolve manually
```bash
git status  # See conflicting files
# Edit files to resolve conflicts
git add <resolved-files>
git commit -m "fix: Resolve merge conflict in config files"
git push
```

---

## Quick Reference

### Essential Commands

```bash
# Check current branch
git branch --show-current

# Switch to v1.4 dev
git checkout claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL

# Pull latest changes
git pull origin claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL

# Stage, commit, push
git add .
git commit -m "feat: Your commit message"
git push

# View commit history
git log --oneline --graph --all --decorate -10

# Check remote sync status
git status
```

### Sync Between Computers

```bash
# Before starting work
git fetch --all
git pull

# After finishing work
git push

# Check what's different from remote
git diff origin/claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
```

---

## Related Documentation

- **Repository Sync:** [SYNC_INSTRUCTIONS.md](SYNC_INSTRUCTIONS.md)
- **v1.4 Roadmap:** [V14_ROADMAP.md](V14_ROADMAP.md)
- **v1.4 Changelog:** [V14_CHANGELOG.md](V14_CHANGELOG.md)
- **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)

---

**Last Updated:** 2025-10-27
**Current Session:** 011CUYNJfTX4Ame2nnowx7FL
**Active Branch:** claude/v14-dev-011CUYNJfTX4Ame2nnowx7FL
