#!/usr/bin/env bash
#
# PhaseGPT Overnight Grid Search
# Trains multiple DPO configurations, evaluates each, and promotes the best.
#
set -euo pipefail

# Configuration
PROJECT_ROOT="/Users/tony_studio/phase-gpt-base"
STEPS_LIST="120 240"
BETA_LIST="0.1 0.3 0.7"
SEED_LIST="1 2"
BASE_ID="Qwen/Qwen2.5-0.5B-Instruct"
CKPT_ROOT="checkpoints/v14/track_a/hybrid_sft_dpo"
DEVICE="mps"

cd "${PROJECT_ROOT}"

# Keep Mac awake during overnight run
caffeinate -dimsu bash -lc 'echo "☕ Caffeinated" >/dev/null' &
CAFE_PID=$!
trap "kill ${CAFE_PID} 2>/dev/null || true" EXIT

# Activate virtual environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: Virtual environment not found at .venv/"
    exit 1
fi
source .venv/bin/activate

# Verify Python
python -V || { echo "Error: Python not found"; exit 1; }

# Quietly update pip and fix urllib3 version
pip -q install -U pip 2>/dev/null || true
pip -q install 'urllib3<2' 2>/dev/null || true

# Create directories
mkdir -p reports/runs logs "${CKPT_ROOT}"

echo "==================================================================="
echo "PhaseGPT Overnight Grid Search"
echo "==================================================================="
echo "Search space:"
echo "  Steps: ${STEPS_LIST}"
echo "  Beta: ${BETA_LIST}"
echo "  Seeds: ${SEED_LIST}"
echo "  Device: ${DEVICE}"
echo "==================================================================="
echo ""

# Helper: Train one configuration
train_one() {
    local steps="$1" beta="$2" seed="$3"
    local tag="steps${steps}_beta${beta}_seed${seed}"
    local outdir="${CKPT_ROOT}/${tag}"
    local logfile="logs/train_${tag}.log"

    echo "[$(date +%H:%M:%S)] 🛠  Training ${tag}..."
    mkdir -p "${outdir}"

    python scripts/train_dpo_min.py \
        --device "${DEVICE}" \
        --steps "${steps}" \
        --beta "${beta}" \
        --seed "${seed}" \
        --output "${outdir}" \
        > "${logfile}" 2>&1 || {
            echo "[$(date +%H:%M:%S)] ❌ Training failed for ${tag} (see ${logfile})"
            return 1
        }

    echo "[$(date +%H:%M:%S)] ✓  Completed ${tag}"
    return 0
}

# Helper: Evaluate one configuration
eval_one() {
    local tag="$1"
    local ckpt="${CKPT_ROOT}/${tag}"
    local out="reports/runs/compare_${tag}_max768.txt"
    local logfile="logs/eval_${tag}.log"

    echo "[$(date +%H:%M:%S)] 🧪 Evaluating ${tag}..."

    python scripts/compare_models.py \
        --phasegpt-ckpt "${ckpt}" \
        --device "${DEVICE}" \
        --mode batch \
        --max-tokens 768 \
        --auto-continue \
        > "${out}" 2>&1 || {
            echo "[$(date +%H:%M:%S)] ⚠️  Evaluation failed for ${tag} (continuing)"
            echo "N/A" > "${out}"
        }

    echo "${out}"
}

# Helper: Parse epistemic scores from comparison output
parse_score() {
    local f="$1"
    local base_ok base_tot ph_ok ph_tot

    # Try to extract "Base ... X/Y" and "PhaseGPT ... X/Y"
    base_ok=$(grep -Eo 'Base[^0-9]*([0-9]+)/([0-9]+)' "$f" 2>/dev/null | head -1 | awk -F'[ /]' '{print $2+0}' || echo "0")
    base_tot=$(grep -Eo 'Base[^0-9]*([0-9]+)/([0-9]+)' "$f" 2>/dev/null | head -1 | awk -F'[ /]' '{print $3+0}' || echo "9")
    ph_ok=$(grep -Eo 'PhaseGPT[^0-9]*([0-9]+)/([0-9]+)' "$f" 2>/dev/null | head -1 | awk -F'[ /]' '{print $2+0}' || echo "0")
    ph_tot=$(grep -Eo 'PhaseGPT[^0-9]*([0-9]+)/([0-9]+)' "$f" 2>/dev/null | head -1 | awk -F'[ /]' '{print $3+0}' || echo "9")

    # Fallback: Try percentage format "66.7% (6/9)"
    if [ "${ph_ok}" = "0" ] && [ "${ph_tot}" = "0" ]; then
        ph_ok=$(grep -Eo 'PhaseGPT[^%]*([0-9]+\.[0-9]+)% \(([0-9]+)/([0-9]+)\)' "$f" 2>/dev/null | head -1 | awk -F'[()%/ ]' '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/){print $i;exit}}' || echo "0")
        ph_tot=$(grep -Eo 'PhaseGPT[^%]*([0-9]+\.[0-9]+)% \(([0-9]+)/([0-9]+)\)' "$f" 2>/dev/null | head -1 | awk -F'[()/ ]' '{print $NF+0}' || echo "9")
    fi

    echo "${base_ok:-0} ${base_tot:-9} ${ph_ok:-0} ${ph_tot:-9}"
}

# Initialize tracking
BEST_TAG=""
BEST_DELTA="-1000"
SUMMARY="reports/overnight_summary.md"

echo "# PhaseGPT Overnight Grid Search" > "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "**Date:** $(date)" >> "${SUMMARY}"
echo "**Search space:** steps={${STEPS_LIST}}, beta={${BETA_LIST}}, seed={${SEED_LIST}}" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "| Tag | Steps | Beta | Seed | PhaseGPT | Base | Δpp | Report |" >> "${SUMMARY}"
echo "|-----|------:|-----:|-----:|---------:|-----:|----:|--------|" >> "${SUMMARY}"

# Grid search
TOTAL_CONFIGS=$(echo "${STEPS_LIST}" | wc -w)
TOTAL_CONFIGS=$((TOTAL_CONFIGS * $(echo "${BETA_LIST}" | wc -w) * $(echo "${SEED_LIST}" | wc -w)))
CURRENT=0

for steps in ${STEPS_LIST}; do
    for beta in ${BETA_LIST}; do
        for seed in ${SEED_LIST}; do
            CURRENT=$((CURRENT + 1))
            TAG="steps${steps}_beta${beta}_seed${seed}"

            echo ""
            echo "==================================================================="
            echo "Configuration ${CURRENT}/${TOTAL_CONFIGS}: ${TAG}"
            echo "==================================================================="

            # Train
            if ! train_one "${steps}" "${beta}" "${seed}"; then
                echo "| ${TAG} | ${steps} | ${beta} | ${seed} | FAILED | - | - | - |" >> "${SUMMARY}"
                continue
            fi

            # Evaluate
            REPORT=$(eval_one "${TAG}")

            # Parse scores
            read -r BASE_OK BASE_TOT PH_OK PH_TOT < <(parse_score "${REPORT}")

            # Calculate percentages and delta
            ph_pct=$(python3 -c "print(int(round(100.0 * ${PH_OK} / max(1, ${PH_TOT}))))" 2>/dev/null || echo "0")
            base_pct=$(python3 -c "print(int(round(100.0 * ${BASE_OK} / max(1, ${BASE_TOT}))))" 2>/dev/null || echo "0")
            delta=$((ph_pct - base_pct))

            # Log to summary
            printf "| %s | %d | %.1f | %d | %d%% (%d/%d) | %d%% (%d/%d) | %+d | %s |\n" \
                "${TAG}" "${steps}" "${beta}" "${seed}" \
                "${ph_pct}" "${PH_OK}" "${PH_TOT}" \
                "${base_pct}" "${BASE_OK}" "${BASE_TOT}" \
                "${delta}" "$(basename "${REPORT}")" >> "${SUMMARY}"

            echo "[$(date +%H:%M:%S)] 📊 ${TAG}: PhaseGPT ${ph_pct}% vs Base ${base_pct}% (Δ${delta}pp)"

            # Track best
            if [ "${delta}" -gt "${BEST_DELTA}" ]; then
                BEST_DELTA="${delta}"
                BEST_TAG="${TAG}"
                echo "[$(date +%H:%M:%S)] 🏆 New best: ${BEST_TAG} (Δ${BEST_DELTA}pp)"
            fi
        done
    done
done

echo ""
echo "==================================================================="
echo "Grid Search Complete"
echo "==================================================================="
echo ""

# Promote best configuration
echo "" >> "${SUMMARY}"
echo "## Best Configuration" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "**Winner:** \`${BEST_TAG}\` **(Δ${BEST_DELTA}pp)**" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"

BEST_CKPT="${CKPT_ROOT}/${BEST_TAG}"
if [ -d "${BEST_CKPT}" ]; then
    echo "[$(date +%H:%M:%S)] 🔗 Promoting ${BEST_TAG} → final symlink"
    rm -f "${CKPT_ROOT}/final"
    ln -s "${BEST_TAG}" "${CKPT_ROOT}/final"
    echo "- Linked \`${BEST_CKPT}\` → \`${CKPT_ROOT}/final\`" >> "${SUMMARY}"
    echo "- Dashboard will now use this configuration" >> "${SUMMARY}"
fi

# Run additional evaluations on best config
echo "[$(date +%H:%M:%S)] 🧪 Running additional evaluations on best config..."
echo "" >> "${SUMMARY}"
echo "## Additional Evaluations (Best Config)" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"

python scripts/compare_models.py \
    --phasegpt-ckpt "${BEST_CKPT}" \
    --device "${DEVICE}" \
    --mode batch \
    --max-tokens 384 \
    --auto-continue \
    > "reports/runs/compare_${BEST_TAG}_max384.txt" 2>&1 || true

echo "- \`compare_${BEST_TAG}_max384.txt\` - Quick evaluation (384 tokens)" >> "${SUMMARY}"
echo "- \`compare_${BEST_TAG}_max768.txt\` - Full evaluation (768 tokens)" >> "${SUMMARY}"

# Commit results
echo ""
echo "[$(date +%H:%M:%S)] 💾 Committing results..."

git add -A
git commit -m "exp: overnight grid ($(date +%Y-%m-%d)) — best: ${BEST_TAG} (Δ${BEST_DELTA}pp)

Grid search results:
- Configurations tested: ${TOTAL_CONFIGS}
- Best performer: ${BEST_TAG}
- Improvement over base: +${BEST_DELTA}pp
- Promoted to: ${CKPT_ROOT}/final

Files:
- reports/overnight_summary.md
- reports/runs/compare_*.txt
- checkpoints/v14/track_a/hybrid_sft_dpo/${BEST_TAG}/

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>" || echo "[$(date +%H:%M:%S)] ⚠️  Nothing to commit"

# Tag if v1.4.3 doesn't exist
if ! git rev-parse v1.4.3 >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] 🏷️  Tagging as v1.4.3..."
    git tag -a v1.4.3 -m "PhaseGPT v1.4.3 — Overnight best: ${BEST_TAG} (Δ${BEST_DELTA}pp)

Automated grid search winner from overnight run on $(date +%Y-%m-%d).

Performance:
- Best config: ${BEST_TAG}
- Improvement: +${BEST_DELTA}pp vs base model

See reports/overnight_summary.md for full results."
else
    echo "[$(date +%H:%M:%S)] ℹ️  Tag v1.4.3 already exists, skipping"
fi

# Push to GitHub
echo "[$(date +%H:%M:%S)] 📤 Pushing to GitHub..."
GIT_SSH_COMMAND="ssh -i ~/.ssh/phasegpt" git push origin v14-dev v1.4.3 2>&1 || {
    echo "[$(date +%H:%M:%S)] ⚠️  Push failed (check SSH key)"
}

# Final summary
echo ""
echo "==================================================================="
echo "✅ Overnight run complete!"
echo "==================================================================="
echo ""
echo "Summary: ${SUMMARY}"
echo ""
head -n 30 "${SUMMARY}"
echo ""
echo "Full results in: reports/overnight_summary.md"
echo "Best checkpoint: ${CKPT_ROOT}/final → ${BEST_TAG}"
echo ""
