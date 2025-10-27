# PhaseGPT Repository Sync Instructions

## Current Status (This Computer - Linux)

✅ **Fully Synced with GitHub**

- **Local branches:** `main` and `claude/session-011CUYNJfTX4Ame2nnowx7FL`
- **Remote branches:** Both synced at commit `db32a05`
- **Tags:** `v1.0.0` (at commit `bcd73a4`)
- **Working directory:** Clean, no uncommitted changes
- **Status:** 100% aligned with GitHub

### Branch Structure
```
main (db32a05)
  └── Same commit as claude/session-011CUYNJfTX4Ame2nnowx7FL
  └── Tagged: v1.0.0 at bcd73a4
```

---

## MacBook Pro Sync Instructions

### Option 1: Fresh Clone (Recommended if you don't have the repo yet)

```bash
# Navigate to your projects directory
cd ~/Projects  # or wherever you keep repositories

# Clone the repository
git clone https://github.com/templetwo/PhaseGPT.git
cd PhaseGPT

# Fetch all branches and tags
git fetch --all --tags

# View available branches
git branch -a

# Checkout main branch (default)
git checkout main

# Optional: Create local tracking branch for session branch
git checkout -b claude/session-011CUYNJfTX4Ame2nnowx7FL origin/claude/session-011CUYNJfTX4Ame2nnowx7FL
```

### Option 2: Update Existing Repository

If you already have PhaseGPT cloned on your MacBook Pro:

```bash
# Navigate to your PhaseGPT directory
cd ~/Projects/PhaseGPT  # adjust path as needed

# Check current status
git status

# If you have uncommitted changes, stash them
git stash save "Temporary stash before sync"

# Fetch all updates from GitHub
git fetch --all --prune --tags

# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Optional: Update session branch if you're working on it
git checkout claude/session-011CUYNJfTX4Ame2nnowx7FL
git pull origin claude/session-011CUYNJfTX4Ame2nnowx7FL

# If you stashed changes earlier, apply them back
git stash pop
```

### Option 3: Verify Sync Status

After syncing, verify everything is aligned:

```bash
# Check git status
git status

# View all branches
git branch -vv

# Check remote branches
git branch -r

# View commit history
git log --oneline --graph --all --decorate -10

# Verify tags
git tag -l

# Check if local matches remote
git diff origin/main  # Should show no differences
```

---

## Expected State After Sync

Both computers should show:

### Commits
```
db32a05 - feat: Add Phase-GPT Oracle v1.0 - Goldilocks LoRA training complete
bfefd2a - Update checkpoints README with OSF DOI
671fbf6 - Add OSF DOI to documentation
bcd73a4 - Initial commit: PhaseGPT v1.0.0 - Phase A complete (tagged v1.0.0)
```

### Branches
- `main` → tracking `origin/main` at `db32a05`
- `claude/session-011CUYNJfTX4Ame2nnowx7FL` → tracking `origin/claude/session-011CUYNJfTX4Ame2nnowx7FL` at `db32a05`

### Tags
- `v1.0.0` → at commit `bcd73a4`

---

## Authentication Setup for MacBook Pro

If you encounter authentication issues when pulling/pushing:

### Method 1: macOS Keychain (Recommended)
```bash
# Configure Git to use macOS Keychain
git config --global credential.helper osxkeychain

# On next push/pull, enter your credentials:
# Username: templetwo
# Password: [Your GitHub Personal Access Token]
```

### Method 2: SSH Keys (Alternative)
```bash
# Check for existing SSH keys
ls -al ~/.ssh

# If none exist, generate new SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add SSH key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
cat ~/.ssh/id_ed25519.pub | pbcopy

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Then update remote URL:
git remote set-url origin git@github.com:templetwo/PhaseGPT.git
```

---

## Troubleshooting

### Issue: "Your branch is behind 'origin/main'"
```bash
git pull origin main
```

### Issue: "You have uncommitted changes"
```bash
# Option 1: Commit them
git add .
git commit -m "WIP: describe changes"

# Option 2: Stash them
git stash save "Work in progress"
```

### Issue: "Merge conflict"
```bash
# View conflicting files
git status

# Resolve conflicts manually, then:
git add <resolved-files>
git commit -m "Resolve merge conflicts"
```

### Issue: "Authentication failed"
- Regenerate your GitHub Personal Access Token
- Ensure token has `repo` scope permissions
- Use token as password, NOT your GitHub password

---

## Quick Verification Commands

Run these on both computers to ensure they're identical:

```bash
# Should show same commit hash
git rev-parse HEAD

# Should show same branch
git branch --show-current

# Should show "nothing to commit, working tree clean"
git status

# Should show no differences
git diff origin/main
```

---

## Next Steps After Sync

Once both computers are synced, you can:

1. **Continue Development:**
   ```bash
   git checkout -b v1.4-dev
   # Make changes, commit, push
   ```

2. **Create Pull Request:**
   - Visit: https://github.com/templetwo/PhaseGPT/compare
   - Select base: `main`, compare: `claude/session-011CUYNJfTX4Ame2nnowx7FL`

3. **Tag New Versions:**
   ```bash
   git tag -a v1.3.0 -m "Release v1.3 with DPO training"
   git push origin v1.3.0
   ```

---

## Repository URLs

- **GitHub:** https://github.com/templetwo/PhaseGPT
- **Clone (HTTPS):** https://github.com/templetwo/PhaseGPT.git
- **Clone (SSH):** git@github.com:templetwo/PhaseGPT.git

---

**Last Updated:** 2025-10-27
**Current Version:** v1.0.0 (tagged), working on v1.3 features
**Active Branches:** `main`, `claude/session-011CUYNJfTX4Ame2nnowx7FL`
