# Quick Reference Card

## 🚀 Common Commands

```bash
# Basic check (rules only)
python check.py exemple.json

# Hybrid AI+Rules (recommended)
python check.py exemple.json --hybrid-mode=on

# Rules only (no AI)
python check.py exemple.json --rules-only

# Custom confidence threshold
python check.py exemple.json --ai-confidence=80

# Show metrics
python check.py exemple.json --show-metrics

# Clean up project
python cleanup_project.py
```

---

## ⚙️ Configuration Quick Edit

Edit `hybrid_config.json`:

```json
{
  "ai_enabled": true,
  "enhancement_level": "full",
  "confidence": {
    "threshold": 70
  },
  "cache": {
    "enabled": true
  }
}
```

**Enhancement Levels:**
- `disabled` - Rules only
- `minimal` - Critical checks only
- `standard` - Most checks (recommended)
- `full` - All checks (default)
- `aggressive` - AI-first

---

## 📊 Understanding Output

### Confidence Scores
- **85-100%** = High (verified by both)
- **60-84%** = Medium (AI variation)
- **0-59%** = Low (needs review)

### Severity
- **CRITICAL** = Must fix
- **MAJOR** = Should fix
- **WARNING** = Review

### Status
- `VERIFIED_BY_BOTH` = Both agree
- `AI_DETECTED_VARIATION` = AI found extra
- `NEEDS_REVIEW` = Low confidence

---

## 🔧 Quick Fixes

### AI Not Working?
```bash
# Check keys
cat .env

# Use rules only
python check.py exemple.json --rules-only
```

### Too Slow?
```json
{
  "cache": {"enabled": true},
  "enhancement_level": "minimal"
}
```

### Low Confidence?
```json
{
  "confidence": {"threshold": 60}
}
```

---

## 📁 Key Files

**Core:**
- `check.py` - Main entry
- `hybrid_config.json` - Configuration
- `.env` - API keys

**Output:**
- `exemple_violations.json` - Results

**Docs:**
- `README.md` - Overview
- `PROJECT_REPORT.md` - Full report
- `QUICK_START.md` - Setup guide

---

## 🎯 Your Test Results

**File:** `exemple.json`  
**Violations:** 38 total
- 22 CRITICAL (58%)
- 15 MAJOR (39%)
- 1 WARNING (3%)

**Categories:**
- STRUCTURE: 3
- GENERAL: 3
- SECURITIES/VALUES: 24
- PERFORMANCE: 3
- PROSPECTUS: 5

---

## 📚 Documentation

1. **README.md** - Start here
2. **PROJECT_REPORT.md** - Complete report
3. **QUICK_START.md** - 5-min setup
4. **API_DOCUMENTATION.md** - API ref
5. **TROUBLESHOOTING_GUIDE.md** - Issues

---

## ✅ Status

**Production Ready** ✅

- Hybrid AI+Rules working
- 38 violations detected
- Full documentation
- Cleanup script ready

**Next:** Run `python cleanup_project.py`

---

**Updated**: 2025-01-18
