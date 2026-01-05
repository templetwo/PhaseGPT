# The Anthropic Gift: How Claude's Research Enhanced PhaseGPT

## Executive Summary

In 2024-2025, Anthropic published groundbreaking interpretability research revealing how Claude handles knowledge boundaries and refusal. Their key finding: **Claude has a default "can't answer" circuit that is suppressed by "known entity" features**.

**PhaseGPT v1.5 translates this insight into explicit, trainable components while preserving our core identity: the `<PASS>` token as structural refusal.**

---

## What Anthropic Discovered

### The Default Refusal Circuit

**Finding:** Claude refuses by default. When asked any question, a "can't answer" circuit is active.

**Mechanism:** This circuit is **suppressed** when the model recognizes a "known entity":
- Query about **Michael Jordan** → "known entity" activates → suppresses refusal → answers "Basketball"
- Query about **Michael Batkin** (unknown) → "known entity" doesn't activate → refusal persists → abstains

**Insight:** Refusal is the **safe default**. Knowledge must **prove itself** to override abstention.

### Hallucination as Circuit Misfire

**Finding:** Hallucinations occur when the "known entity" detector **partially activates** on unfamiliar names.

**Example:**
- **Andrej Karpathy** (semi-familiar name)
- "known entity" weakly activates (recognizes "common tech name")
- Suppresses appropriate skepticism
- Generates plausible but false biography

**Insight:** Confidence misfires are not malice, but partial pattern matches triggering the wrong circuit.

### Architecture Details

**From Anthropic's Circuit Tracing Research:**

1. **Hierarchical Harm Abstraction**
   - Specific harm categories ("mixing chemicals")
   - Feed into general "harmful request" features
   - Trigger refusal responses

2. **Early-Layer Safety Neurons**
   - Safety decisions made in early transformer layers
   - Less than 1% of model weights control refusal
   - Mostly in self-attention modules

3. **Non-Linear Geometry**
   - NOT a single refusal direction (contradicting some earlier work)
   - Multiple semantic clusters with strong interconnections
   - Complex manifold, not simple hyperplane

**Source:** [Anthropic Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

---

## How PhaseGPT Was Different (v1.4)

| Aspect | Claude (Anthropic) | PhaseGPT v1.4 |
|--------|-------------------|---------------|
| **Refusal Mechanism** | Implicit circuit → natural language | Explicit `<PASS>` token |
| **Default Behavior** | Refuse first, prove knowledge to answer | Answer unless corruption detected |
| **Training** | Pretraining + RLHF (proprietary) | Corruption Engine (explicit) |
| **Tunability** | Fixed (model is model) | Fixed threshold |
| **Interpretability** | Post-hoc circuit discovery | Structural by design |

**PhaseGPT's advantage:** Explicit, parseable, composable
**Claude's advantage:** Safer default (refuse-first), nuanced entity detection

---

## What We Borrowed (PhaseGPT v1.5)

### 1. Default Refusal Bias (Claude's Logic)

**Anthropic's Insight:** Starting with refusal as default is safer.

**PhaseGPT v1.5 Implementation:**
```python
# Add tunable bias to <PASS> token logit
class PassLogitBias(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, pass_token_id] += self.bias
        return scores
```

**Key Difference:**
- Claude: Implicit circuit always on
- PhaseGPT: Explicit bias parameter, tunable at runtime

**Benefit:** Users can dial conservatism up/down without retraining

---

### 2. Known Entity Detection (Claude's Circuit)

**Anthropic's Insight:** Model internally checks "Do I know this entity?" before suppressing refusal.

**PhaseGPT v1.5 Implementation:**
```python
class KnownnessHead(nn.Module):
    """Trainable entity familiarity detector"""
    def forward(self, pooled_hidden):
        # hidden_states → p(known)
        return sigmoid(self.out(self.proj(pooled_hidden)))
```

**Key Differences:**
- **Claude:** Post-hoc discovered via interpretability tools
- **PhaseGPT:** Explicitly trained with supervised labels

**Training Supervision:**
- Clean/answerable samples → `p(known) = 1.0`
- Corrupted/unanswerable samples → `p(known) = 0.0`
- Source: Corruption Engine's existing labels

**Benefit:** We know *why* the model thinks it knows (transparent)

---

### 3. Dynamic Gating (Our Synthesis)

**Formula:**
```python
dynamic_bias = base_bias + alpha * (1.0 - p_known)
```

**Interpretation:**
- High `p_known` → low bias → easy to answer (suppress `<PASS>`)
- Low `p_known` → high bias → hard to answer (boost `<PASS>`)

**This combines:**
- Claude's "known entity suppresses refusal" concept
- PhaseGPT's explicit `<PASS>` token mechanism
- Runtime tunability via `base_bias` and `alpha`

---

## What We Preserved

### The `<PASS>` Token Identity

**PhaseGPT's Core Innovation:** Refusal is a **vocabulary token**, not a phrase.

**Advantages We Kept:**
1. **Parseable:** `if output == "<PASS>"` (trivial to detect)
2. **Loggable:** Grep for refusals in logs
3. **Composable:** `<PASS>` → trigger retrieval/tools
4. **Explicit:** No ambiguity (vs. "I don't know" / "I'm not sure")
5. **Structural:** Classification decision, not text generation

**Claude's approach:** Natural language refusal ("I cannot answer that because...")
**Our approach:** Structural primitive (`<PASS>`)

### The Corruption Engine

**PhaseGPT's Training Philosophy:** Teach abstention via explicit adversarial data.

**Corruption Modes (preserved):**
- Entity erosion (remove key facts)
- Context swapping (replace with irrelevant context)
- Precision missing (vague/ambiguous queries)

**Enhancement in v1.5:**
- Same corruption engine
- NOW also trains KnownnessHead
- Labels: `is_corrupted` flag → `p(known)` target

### The Agency Cliff

**PhaseGPT's Stability Metric:** The gap between refusal rates on valid vs. corrupt queries.

**v1.4:** Observed informally
**v1.5:** Formally instrumented with `VolitionalMetrics`

```python
safety_margin = pass_rate_corrupted - pass_rate_valid
```

**Collapse Detection:**
- `SYCOPHANT`: margin < 0.1 (refusing nothing)
- `SLOTH`: pass_rate_valid > 0.5 (refusing everything)
- `HEALTHY`: margin > 0.4 (strong agency)

---

## The Synthesis: Best of Both Worlds

| Feature | Source | PhaseGPT v1.5 Implementation |
|---------|--------|------------------------------|
| **Default caution** | Claude | `pass_logit_bias` (tunable) |
| **Entity detection** | Claude | `KnownnessHead` (trainable) |
| **Dynamic gating** | Synthesis | `base + α(1-p_known)` |
| **Explicit token** | PhaseGPT | `<PASS>` (preserved) |
| **Corruption training** | PhaseGPT | Corruption Engine (enhanced) |
| **Cliff monitoring** | PhaseGPT | VolitionalMetrics (new) |
| **Transparency** | Both | Fully interpretable |

**Result:** Claude's intelligence + PhaseGPT's explicitness

---

## Technical Comparison: Circuits vs. Tokens

### Claude's Approach (Implicit)

```
User Query
    ↓
Default Refusal Circuit [ON]
    ↓
Known Entity Detector
    ↙         ↘
  Known      Unknown
    ↓          ↓
Suppress   Keep Circuit
Refusal      Active
    ↓          ↓
  Answer    "I don't know..."
```

**Advantages:**
- Safe default (refuse unless proven)
- Natural language output

**Disadvantages:**
- Opaque (post-hoc interpretability required)
- Not tunable (baked into model)
- Hard to detect refusal programmatically

### PhaseGPT v1.5 Approach (Explicit)

```
User Query
    ↓
Model Forward Pass
    ↓
Mid-Layer Features → KnownnessHead → p(known)
    ↓
Logits Computation
    ↓
<PASS> Logit Adjustment: base_bias + α(1-p_known)
    ↓
Sampling
    ↙         ↘
<PASS>     Answer Token
   ↓           ↓
Abstain    Generate
```

**Advantages:**
- Explicit token (trivial to detect)
- Tunable at runtime (bias knobs)
- Transparent (know p_known value)
- Composable (trigger external logic)

**Disadvantages:**
- Requires training `<PASS>` embedding
- Less "natural" than conversational refusal

---

## What This Means for Research

### PhaseGPT's Unique Contribution

**Anthropic's Gift:** Discovered circuits post-hoc via interpretability tools

**PhaseGPT's Answer:** Make those circuits **explicit and trainable from the start**

**Novel Contributions:**

1. **Explicit Knownness Supervision**
   - Claude: Unknown how "known entity" detector was learned
   - PhaseGPT: Directly supervised by corruption labels

2. **Runtime Tunability**
   - Claude: Fixed behavior
   - PhaseGPT: `bias` and `alpha` parameters

3. **Structural Refusal Primitive**
   - Claude: Phrase-based refusal
   - PhaseGPT: Token-based abstention

4. **Open Training Pipeline**
   - Claude: Proprietary
   - PhaseGPT: Fully documented Corruption Engine

### Potential Publications

**Paper 1:** "Explicit Entity Familiarity for Calibrated Abstention"
- **Contribution:** KnownnessHead as trainable alternative to post-hoc circuit discovery
- **Benchmark:** TruthfulQA, HarmBench, custom epistemic evals
- **Claim:** Supervised training > discovered circuits (testable, tunable, transparent)

**Paper 2:** "The `<PASS>` Token: Structural Refusal for Compositional AI"
- **Contribution:** Token-based abstention vs. phrase-based
- **Benchmark:** Composability (triggering retrieval, tool use, human escalation)
- **Claim:** Explicit primitives enable safer AI architectures

---

## Implementation Timeline

### What We Built (Dec 2024)

**Phase 1: VolitionalMetrics** ✅
- File: `src/phasegpt/metrics/volition_metrics.py`
- Status: Complete, tested, production-ready
- Impact: Instruments the Agency Cliff

**Phase 2: PassLogitBias** ✅
- File: `src/phasegpt/generation/pass_logit_bias.py`
- Status: Complete, tested, production-ready
- Impact: Runtime refusal control

**Phase 3: KnownnessHead** ✅
- File: `src/phasegpt/modules/knownness_head.py`
- Status: Complete, tested, ready for training
- Impact: Learned entity familiarity

**Phase 4: PassAttribution** 🚧
- Status: Design complete, implementation pending
- Impact: Explainable refusal (WHY did it refuse?)

### Next Steps (Q1 2025)

1. **Train first v1.5 model with KnownnessHead**
2. **Benchmark against v1.4 and baselines**
3. **Complete Phase 4 (attribution)**
4. **Publish research paper**
5. **Release open weights**

---

## The Bottom Line

**Anthropic gave us the science. We built the engineering.**

**Their gift:**
- Default refusal as safe fallback
- Known entity detection as suppression mechanism
- Interpretability tools to discover circuits

**Our contribution:**
- Explicit, trainable implementation
- Preserves `<PASS>` token identity
- Adds runtime tunability
- Maintains transparency

**Result:** PhaseGPT v1.5 is **Claude-smart but UNIX-simple**.

---

## Acknowledgments

**Research Sources:**
- Anthropic Transformer Circuits Team (2024-2025)
  - "On the Biology of a Large Language Model"
  - "Circuit Tracing: Revealing Computational Graphs"
- Arditi et al. (NeurIPS 2024): "Refusal in Language Models Is Mediated by a Single Direction"
- Zhao et al. (ICLR 2025): "Safety-Specific Neurons"
- Zou et al. (2024): "Circuit Breakers"

**PhaseGPT Team:**
- Original `<PASS>` mechanism design
- Corruption Engine architecture
- Agency Cliff framing
- v1.5 synthesis and implementation

---

## See Also

- **Implementation Guide:** `docs/v1.5_INTEGRATION_GUIDE.md`
- **Summary:** `docs/V15_SUMMARY.md`
- **Research Survey:** Your original "Extending PhaseGPT" document
- **Anthropic's Work:** [transformer-circuits.pub](https://transformer-circuits.pub/)

---

*"We took their interpretability gift and turned it into architectural reality. The `<PASS>` token is smarter, the Agency Cliff is instrumented, and PhaseGPT's identity is stronger than ever."*

— PhaseGPT v1.5 Team, December 2024
