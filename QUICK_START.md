# Quick Start Guide - Hybrid Compliance Checker

## 5-Minute Setup

### 1. Verify Installation

```bash
# Run integration test
python test_integration.py
```

Expected output: "✅ ALL INTEGRATION TESTS PASSED"

### 2. Test Backward Compatibility

```bash
# Run with existing workflow (no changes)
python check.py exemple.json
```

This should work exactly as before.

### 3. Enable AI Mode

```bash
# Enable AI+Rules hybrid mode
python check.py exemple.json --hybrid-mode=on
```

### 4. Review Results

Check the output file: `exemple_violations.json`

Look for AI-enhanced fields:
- `ai_reasoning` - Why AI detected the violation
- `confidence` - Confidence score (0-100)
- `status` - Detection status

### 5. Adjust Configuration (Optional)

Edit `hybrid_config.json`:

```json
{
  "enhancement_level": "full",
  "confidence": {
    "threshold": 70
  }
}
```

## Common Commands

### Basic Usage

```bash
# Standard check (backward compatible)
python check.py exemple.json

# Enable AI mode
python check.py exemple.json --hybrid-mode=on

# Rules only (no AI)
python check.py exemple.json --rules-only

# Set confidence threshold
python check.py exemple.json --ai-confidence=80

# Show performance metrics
python check.py exemple.json --show-metrics
```

### Configuration

```bash
# View current configuration
python -c "from config_manager import get_config_manager; get_config_manager().print_summary()"

# Test integration
python test_integration.py
```

## Enhancement Levels

Choose your AI enhancement level in `hybrid_config.json`:

| Level | Description | Use Case |
|-------|-------------|----------|
| `disabled` | Rules only | Testing, fallback |
| `minimal` | Critical checks only | Conservative start |
| `standard` | Most checks | Recommended |
| `full` | All checks | Maximum accuracy |
| `aggressive` | AI-first | Experimental |

## Configuration Quick Reference

### Enable/Disable AI

```json
{
  "ai_enabled": true,
  "enhancement_level": "full"
}
```

### Adjust Confidence

```json
{
  "confidence": {
    "threshold": 70,
    "high_confidence": 85,
    "review_threshold": 60
  }
}
```

### Enable Caching

```json
{
  "cache": {
    "enabled": true,
    "max_size": 1000
  }
}
```

### Feature Flags

```json
{
  "features": {
    "enable_promotional_ai": true,
    "enable_performance_ai": true,
    "enable_prospectus_ai": true
  }
}
```

## Troubleshooting

### AI Not Working

1. Check API keys in `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   TOKENFACTORY_API_KEY=your_key_here
   ```

2. Test connection:
   ```bash
   python test_integration.py
   ```

3. Check configuration:
   ```bash
   python -c "from check_hybrid import get_hybrid_integration; print(get_hybrid_integration().is_hybrid_enabled())"
   ```

### Low Confidence Scores

Edit `hybrid_config.json`:

```json
{
  "confidence": {
    "threshold": 60
  }
}
```

### Slow Performance

Enable caching and reduce batch size:

```json
{
  "cache": {
    "enabled": true,
    "max_size": 1000
  },
  "batch_size": 3
}
```

## Next Steps

1. **Read the full guide**: See `INTEGRATION_GUIDE.md`
2. **Follow migration checklist**: See `MIGRATION_CHECKLIST.md`
3. **Review configuration**: Edit `hybrid_config.json`
4. **Test thoroughly**: Use `test_integration.py`
5. **Monitor performance**: Use `--show-metrics` flag

## Support

- **Integration Guide**: `INTEGRATION_GUIDE.md`
- **Migration Checklist**: `MIGRATION_CHECKLIST.md`
- **Implementation Summary**: `TASK_6_IMPLEMENTATION_SUMMARY.md`
- **Configuration Reference**: See `config_manager.py` docstrings

## Key Points

✅ **Backward Compatible**: Works without any changes
✅ **Gradual Adoption**: Enable AI features at your pace
✅ **Easy Rollback**: Use `--rules-only` to disable AI
✅ **Flexible Configuration**: Control every aspect
✅ **Well Documented**: Comprehensive guides included

---

**Ready to go!** Start with `python check.py exemple.json --hybrid-mode=on`
