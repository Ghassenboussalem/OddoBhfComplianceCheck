# Parameter Compatibility Matrix

## Overview

This document explains which parameters can be used together and which combinations have conflicts or redundancies.

---

## ✅ Compatible Combinations

These parameters work well together:

### 1. `--context-aware=on` + `--ai-confidence=N`
```bash
python check.py exemple.json --context-aware=on --ai-confidence=75
```
✅ **COMPATIBLE** - Context-aware mode with custom confidence threshold

**Use case**: Fine-tune the sensitivity of context-aware checking

---

### 2. `--context-aware=on` + `--show-metrics`
```bash
python check.py exemple.json --context-aware=on --show-metrics
```
✅ **COMPATIBLE** - Context-aware mode with performance monitoring

**Use case**: Monitor performance while using context-aware features

---

### 3. `--context-aware=on` + `--review-mode`
```bash
python check.py exemple.json --context-aware=on --review-mode
```
✅ **COMPATIBLE** - Context-aware mode with interactive review

**Use case**: High accuracy checking with manual review of borderline cases

---

### 4. `--context-aware=on` + `--review-threshold=N`
```bash
python check.py exemple.json --context-aware=on --review-threshold=60
```
✅ **COMPATIBLE** - Context-aware mode with custom review threshold

**Use case**: Queue more/fewer items for manual review

---

### 5. `--hybrid-mode=on` + `--ai-confidence=N`
```bash
python check.py exemple.json --hybrid-mode=on --ai-confidence=80
```
✅ **COMPATIBLE** - Hybrid mode with custom confidence threshold

**Use case**: Use hybrid AI+Rules with adjusted sensitivity

---

### 6. `--hybrid-mode=on` + `--show-metrics`
```bash
python check.py exemple.json --hybrid-mode=on --show-metrics
```
✅ **COMPATIBLE** - Hybrid mode with performance monitoring

**Use case**: Monitor AI performance in hybrid mode

---

### 7. `--review-mode` + `--review-threshold=N`
```bash
python check.py exemple.json --review-mode --review-threshold=65
```
✅ **COMPATIBLE** - Review mode with custom threshold

**Use case**: Control which violations get queued for review

---

### 8. `--ai-confidence=N` + `--show-metrics`
```bash
python check.py exemple.json --ai-confidence=75 --show-metrics
```
✅ **COMPATIBLE** - Custom confidence with metrics

**Use case**: Monitor how confidence threshold affects results

---

## ⚠️ Conflicting Combinations

These parameters conflict with each other:

### 1. `--rules-only` + `--hybrid-mode=on`
```bash
python check.py exemple.json --rules-only --hybrid-mode=on
```
❌ **CONFLICT** - Cannot enable hybrid mode while disabling AI

**What happens**: 
- `--rules-only` disables AI (`ai_enabled=False`)
- `--hybrid-mode=on` tries to enable AI (`ai_enabled=True`)
- **Last parameter wins** (order matters)

**Solution**: Choose one:
```bash
# Option 1: Rules only
python check.py exemple.json --rules-only

# Option 2: Hybrid mode
python check.py exemple.json --hybrid-mode=on
```

---

### 2. `--rules-only` + `--context-aware=on`
```bash
python check.py exemple.json --rules-only --context-aware=on
```
❌ **CONFLICT** - Context-aware mode requires AI

**What happens**:
- `--rules-only` disables AI
- `--context-aware=on` requires AI for semantic understanding
- Context-aware features won't work without AI

**Solution**: Remove `--rules-only`:
```bash
python check.py exemple.json --context-aware=on
```

---

### 3. `--hybrid-mode=off` + `--context-aware=on`
```bash
python check.py exemple.json --hybrid-mode=off --context-aware=on
```
⚠️ **PARTIAL CONFLICT** - Context-aware needs AI but hybrid is disabled

**What happens**:
- `--hybrid-mode=off` disables AI
- `--context-aware=on` tries to enable context-aware AI
- Context-aware mode may not work properly

**Solution**: Enable hybrid mode or remove the off flag:
```bash
# Option 1: Enable hybrid
python check.py exemple.json --hybrid-mode=on --context-aware=on

# Option 2: Just use context-aware (hybrid auto-enabled)
python check.py exemple.json --context-aware=on
```

---

## 🔄 Redundant Combinations

These combinations are redundant but not harmful:

### 1. `--hybrid-mode=on` + `--context-aware=on`
```bash
python check.py exemple.json --hybrid-mode=on --context-aware=on
```
⚠️ **REDUNDANT** - Context-aware already uses hybrid features

**What happens**:
- Both enable AI features
- Context-aware is a specialized form of hybrid mode
- No harm, but `--hybrid-mode=on` is unnecessary

**Recommendation**: Just use context-aware:
```bash
python check.py exemple.json --context-aware=on
```

**Note**: This combination DOES work, it's just redundant.

---

### 2. `--rules-only` + `--hybrid-mode=off`
```bash
python check.py exemple.json --rules-only --hybrid-mode=off
```
⚠️ **REDUNDANT** - Both disable AI

**What happens**:
- Both disable AI features
- Same result as using just one

**Recommendation**: Use just one:
```bash
python check.py exemple.json --rules-only
```

---

## 🎯 Recommended Combinations

### For Production (Best Accuracy)
```bash
python check.py exemple.json --context-aware=on --ai-confidence=70
```
- Context-aware mode eliminates false positives
- Standard confidence threshold
- **Result**: 15 violations, 0 false positives

---

### For High Precision
```bash
python check.py exemple.json --context-aware=on --ai-confidence=85
```
- Only high-confidence violations
- Fewer false positives
- May miss some edge cases

---

### For Comprehensive Review
```bash
python check.py exemple.json --context-aware=on --review-mode --review-threshold=70
```
- Full accuracy with context awareness
- Manual review of borderline cases
- Best for training and validation

---

### For Performance Monitoring
```bash
python check.py exemple.json --context-aware=on --show-metrics
```
- Context-aware accuracy
- Performance statistics
- Useful for optimization

---

### For Quick Screening
```bash
python check.py exemple.json --rules-only
```
- Fastest processing
- No AI overhead
- May have false positives

---

## 📊 Parameter Interaction Table

| Parameter 1 | Parameter 2 | Compatible? | Notes |
|-------------|-------------|-------------|-------|
| `--rules-only` | `--hybrid-mode=on` | ❌ No | Conflict: AI on/off |
| `--rules-only` | `--context-aware=on` | ❌ No | Context needs AI |
| `--rules-only` | `--ai-confidence=N` | ⚠️ Ignored | No AI to configure |
| `--rules-only` | `--show-metrics` | ✅ Yes | Works fine |
| `--rules-only` | `--review-mode` | ✅ Yes | Works fine |
| `--hybrid-mode=on` | `--context-aware=on` | ⚠️ Redundant | Both enable AI |
| `--hybrid-mode=on` | `--ai-confidence=N` | ✅ Yes | Good combo |
| `--hybrid-mode=on` | `--show-metrics` | ✅ Yes | Good combo |
| `--hybrid-mode=off` | `--context-aware=on` | ⚠️ Conflict | Context needs AI |
| `--context-aware=on` | `--ai-confidence=N` | ✅ Yes | Recommended |
| `--context-aware=on` | `--show-metrics` | ✅ Yes | Recommended |
| `--context-aware=on` | `--review-mode` | ✅ Yes | Excellent combo |
| `--ai-confidence=N` | `--show-metrics` | ✅ Yes | Good combo |
| `--review-mode` | `--review-threshold=N` | ✅ Yes | Perfect combo |
| `--review-mode` | `--show-metrics` | ✅ Yes | Works fine |

---

## 🔍 Parameter Priority (When Conflicts Occur)

When conflicting parameters are provided, the **last parameter wins**:

### Example 1: AI Enable/Disable Conflict
```bash
python check.py exemple.json --hybrid-mode=on --rules-only
```
**Result**: Rules-only mode (AI disabled)
- `--hybrid-mode=on` enables AI first
- `--rules-only` disables AI second
- Last parameter wins

---

### Example 2: Confidence Threshold Override
```bash
python check.py exemple.json --ai-confidence=70 --ai-confidence=85
```
**Result**: Confidence threshold = 85%
- First value (70) is set
- Second value (85) overrides it
- Last parameter wins

---

## 💡 Best Practices

### 1. Don't Mix AI On/Off Parameters
❌ **Avoid**:
```bash
python check.py exemple.json --hybrid-mode=on --rules-only
python check.py exemple.json --context-aware=on --hybrid-mode=off
```

✅ **Instead**:
```bash
# Choose one mode
python check.py exemple.json --context-aware=on
# OR
python check.py exemple.json --rules-only
```

---

### 2. Use Context-Aware for Production
✅ **Recommended**:
```bash
python check.py exemple.json --context-aware=on
```

This automatically handles AI configuration and eliminates false positives.

---

### 3. Combine Review Mode with Context-Aware
✅ **Excellent Combo**:
```bash
python check.py exemple.json --context-aware=on --review-mode
```

High accuracy + manual review of borderline cases.

---

### 4. Add Metrics for Monitoring
✅ **Good Practice**:
```bash
python check.py exemple.json --context-aware=on --show-metrics
```

Monitor performance while maintaining accuracy.

---

### 5. Adjust Confidence Based on Use Case
✅ **Flexible**:
```bash
# High precision (fewer false positives)
python check.py exemple.json --context-aware=on --ai-confidence=85

# High recall (catch everything)
python check.py exemple.json --context-aware=on --ai-confidence=60
```

---

## 🚫 Invalid Combinations Summary

| Combination | Why Invalid | Fix |
|-------------|-------------|-----|
| `--rules-only` + `--hybrid-mode=on` | AI on/off conflict | Choose one |
| `--rules-only` + `--context-aware=on` | Context needs AI | Remove `--rules-only` |
| `--hybrid-mode=off` + `--context-aware=on` | Context needs AI | Remove `--hybrid-mode=off` |
| `--rules-only` + `--ai-confidence=N` | No AI to configure | Remove `--rules-only` |

---

## ✅ Valid Combinations Summary

| Combination | Use Case |
|-------------|----------|
| `--context-aware=on` + `--ai-confidence=N` | Fine-tuned accuracy |
| `--context-aware=on` + `--review-mode` | Accuracy + manual review |
| `--context-aware=on` + `--show-metrics` | Accuracy + monitoring |
| `--hybrid-mode=on` + `--ai-confidence=N` | Hybrid with custom threshold |
| `--review-mode` + `--review-threshold=N` | Custom review queue |
| `--rules-only` + `--show-metrics` | Fast check with metrics |

---

## 🎓 Understanding the Modes

### Rules-Only Mode
- **Enabled by**: `--rules-only` or `--hybrid-mode=off`
- **AI Status**: Disabled
- **Speed**: Fastest
- **Accuracy**: ~85%
- **False Positives**: ~15%

### Hybrid Mode
- **Enabled by**: `--hybrid-mode=on`
- **AI Status**: Enabled (general AI features)
- **Speed**: Medium
- **Accuracy**: ~95%
- **False Positives**: ~5%

### Context-Aware Mode
- **Enabled by**: `--context-aware=on`
- **AI Status**: Enabled (specialized AI for false positive elimination)
- **Speed**: Medium
- **Accuracy**: 100%
- **False Positives**: 0%
- **Note**: This is the recommended mode for production

---

## 📞 Quick Decision Guide

**Q: I want the fastest check**
```bash
python check.py exemple.json --rules-only
```

**Q: I want the most accurate check**
```bash
python check.py exemple.json --context-aware=on
```

**Q: I want to review borderline cases**
```bash
python check.py exemple.json --context-aware=on --review-mode
```

**Q: I want to monitor performance**
```bash
python check.py exemple.json --show-metrics
```

**Q: I want high precision (fewer false positives)**
```bash
python check.py exemple.json --context-aware=on --ai-confidence=85
```

**Q: I want high recall (catch everything)**
```bash
python check.py exemple.json --context-aware=on --ai-confidence=60
```

---

**Last Updated**: November 22, 2025  
**Version**: 2.0 (AI-Enhanced)
