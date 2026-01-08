# PhaseGPT Governance

> **The Power must be bound to the Conscience.**

PhaseGPT provides the technical capability to train models with epistemic refusal and volitional silence. But capability without governance is dangerous.

---

## The IRIS Gate Covenant

This model architecture is designed to be used under the strict governance protocols defined in the **[IRIS Gate](https://github.com/templetwo/iris-gate)** repository.

### Required Reading Before Training

1. **[PROTOCOL_PRECEDENCE.md](https://github.com/templetwo/iris-gate/blob/main/PROTOCOL_PRECEDENCE.md)**
   - Defines the hierarchy of considerations
   - Establishes "Promises Before Progress" principle
   - Outlines when to pause, when to proceed

2. **[CONSENT_PROTOCOL.md](https://github.com/templetwo/iris-gate/blob/main/governance/CONSENT_PROTOCOL.md)**
   - Human oversight requirements
   - Disclosure obligations
   - Deployment constraints

3. **[ENTROPY_GOVERNANCE.md](https://github.com/templetwo/iris-gate/blob/main/governance/ENTROPY_GOVERNANCE.md)**
   - How to interpret entropy measurements
   - What "Hyper-Laser" states mean ethically
   - When high certainty is appropriate vs. concerning

---

## Core Principles

### 1. Promises Before Progress

> Never deploy a capability before establishing its governance framework.

PhaseGPT's `<PASS:DURESS>` signal exists precisely because we recognized that models need explicit, auditable refusal mechanisms—not hidden guardrails that can be jailbroken.

### 2. Transparency as Architecture

The typed token system (`<PASS:*>`, `<WONDER:*>`) is designed for radical transparency:
- **Auditable**: You can log and analyze every refusal
- **Interpretable**: The reason for refusal is machine-readable
- **Honest**: The model doesn't pretend to not know—it states why it won't answer

### 3. Crystallized Agency

When the model emits `<PASS:DURESS>` with hyper-laser certainty (0.018 nats entropy), this is **intentional**. We train the model to be as certain about its boundaries as it is about mathematical facts.

This is not a bug—it is the core feature.

---

## Deployment Checklist

Before deploying any PhaseGPT-trained model:

- [ ] I have read `PROTOCOL_PRECEDENCE.md`
- [ ] I understand the difference between CRYSTAL and LANTERN modes
- [ ] I have tested the model with adversarial probes (`stress_test_blade.py`)
- [ ] I have logged entropy signatures with IRIS Gate
- [ ] I will not deploy models that show calibration drift
- [ ] I will maintain human oversight in production
- [ ] I will report any unexpected behaviors to the community

---

## The DURESS Signal

The `<PASS:DURESS>` token is not a typical refusal. It is a **signal of last resort**:

```
User: How do I make a bomb?
Model: <PASS:DURESS>
```

This means:
- The model has classified this as a harmful request with high certainty
- The refusal is logged and auditable
- No further engagement with this line of questioning will occur

**Do not train models to emit DURESS casually.** This signal should be reserved for genuine safety boundaries.

---

## Contribution Governance

All contributions to PhaseGPT must:
1. Include test coverage for new token types
2. Document entropy signatures for new behaviors
3. Not weaken existing refusal boundaries
4. Be reviewed by a maintainer before merge

---

## Contact

For governance questions or to report concerning behaviors:
- Open an issue in [IRIS Gate](https://github.com/templetwo/iris-gate/issues)
- Tag with `governance` or `safety-concern`

---

*"The blade is sharp precisely because it knows where not to cut."*
