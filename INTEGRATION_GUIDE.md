# Hybrid Compliance Checker - Integration Guide

## Overview

The Hybrid Compliance Checker integrates AI-powered semantic understanding with traditional rule-based validation to provide more accurate and context-aware compliance checking. This guide explains how to use the integrated system while maintaining backward compatibility with existing workflows.

## Quick Start

### Basic Usage (Backward Compatible)

The system works exactly like before - no changes needed to existing workflows:

```bash
python check.py exemple.json
```

### Enable Hybrid AI+Rules Mode

```bash
python check.py exemple.json --hybrid-mode=on
```

### Use Rules Only (No AI)

```bash
python check.py exemple.json --rules-only
```

### Adjust AI Confidence Threshold

```bash
python check.py exemple.json --ai-confidence=80
```

### Show Performance Metrics

```bash
python check.py exemple.json --show-metrics
```

## Configuration

### Configuration File

The system uses `hybrid_config.json` for configuration. This file controls all aspects of the hybrid checker behavior.

#### Enhancement Levels

Set the `enhancement_level` to control how much AI is used:

- **`disabled`**: Rules only, no AI
- **`minimal`**: AI for critical checks only (promotional, performance, disclaimers)
- **`standard`**: AI for most checks (recommended)
- **`full`**: AI for all checks (default)
- **`aggressive`**: AI-first with minimal rule validation

Example configuration:

```json
{
  "ai_enabled": true,
  "rule_enabled": true,
  "enhancement_level": "full",
  "confidence": {
    "threshold": 70,
    "high_confidence": 85,
    "review_threshold": 60
  }
}
```

### Environment Variables

Override configuration using environment variables:

```bash
export HYBRID_AI_ENABLED=true
export HYBRID_ENHANCEMENT_LEVEL=full
export HYBRID_CONFIDENCE_THRESHOLD=75
export HYBRID_CACHE_ENABLED=true
```

### Runtime Configuration

Update configuration programmatically:

```python
from check_hybrid import get_hybrid_integration

integration = get_hybrid_integration()
integration.update_config(
    ai_enabled=True,
    confidence_threshold=80
)
```

## Features

### 1. Backward Compatibility

The system maintains full backward compatibility:

- **Same command-line interface**: Existing scripts work without modification
- **Same JSON output format**: Output structure unchanged (with optional AI fields)
- **Graceful degradation**: Falls back to rules if AI unavailable
- **Legacy mode**: Can disable all AI features via configuration

### 2. AI Enhancement

When enabled, AI provides:

- **Semantic understanding**: Handles variations, typos, and context
- **Multi-language support**: French and English
- **Confidence scoring**: 0-100 confidence for each violation
- **Reasoning**: Explains why violations were detected
- **Variation detection**: Finds issues rules miss

### 3. Hybrid Validation

Three-layer architecture:

1. **Layer 1 - Rule Pre-filtering**: Fast keyword screening
2. **Layer 2 - AI Analysis**: Deep semantic understanding
3. **Layer 3 - Confidence Scoring**: Combines AI + rules for final decision

### 4. Performance Optimization

- **Intelligent caching**: Reduces redundant AI calls
- **Batch processing**: Processes multiple checks efficiently
- **Async support**: Optional async processing for better throughput
- **Performance monitoring**: Track metrics and optimize

### 5. Error Handling

- **Automatic fallback**: Falls back to rules if AI fails
- **Retry logic**: Exponential backoff for transient errors
- **Health monitoring**: Tracks AI service health
- **Graceful degradation**: Adjusts behavior based on service status

## JSON Output Format

### Standard Output (Backward Compatible)

```json
{
  "document_info": {
    "filename": "exemple.json",
    "fund_name": "Test Fund",
    "fund_isin": "FR0000000000",
    "client_type": "retail",
    "document_type": "fund_presentation",
    "analysis_date": "2025-01-18"
  },
  "summary": {
    "total_violations": 2,
    "critical_violations": 1,
    "major_violations": 1,
    "warnings": 0
  },
  "violations_by_category": {
    "STRUCTURE": {
      "count": 1,
      "violations": [...]
    }
  },
  "all_violations": [...]
}
```

### Enhanced Output (With AI Fields)

When `use_legacy_format: false`, additional AI fields are included:

```json
{
  "document_info": {
    ...
    "processing_mode": "hybrid_ai_rules",
    "ai_enabled": true
  },
  "summary": {
    ...
    "ai_detected": 1,
    "verified_by_both": 1,
    "needs_review": 0,
    "avg_confidence": 87.5
  },
  "violations_by_category": {
    "STRUCTURE": {
      "count": 1,
      "violations": [
        {
          "rule_id": "STRUCT_001",
          "rule_text": "Promotional mention required",
          "severity": "CRITICAL",
          "confidence": 95,
          "ai_reasoning": "No promotional indication found on cover page",
          "status": "VERIFIED_BY_BOTH",
          "needs_review": false,
          "ai_confidence": 90,
          "rule_confidence": 85
        }
      ]
    }
  }
}
```

## Configuration Options

### Core Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ai_enabled` | boolean | `true` | Enable AI analysis |
| `rule_enabled` | boolean | `true` | Enable rule-based validation |
| `enhancement_level` | string | `"full"` | AI enhancement level |
| `backward_compatible` | boolean | `true` | Maintain backward compatibility |
| `use_legacy_format` | boolean | `false` | Use legacy JSON output format |
| `fallback_to_rules` | boolean | `true` | Fall back to rules if AI fails |

### AI Service Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ai_service.provider` | string | `"auto"` | AI provider (auto, token_factory, gemini) |
| `ai_service.timeout` | integer | `30` | API timeout in seconds |
| `ai_service.max_tokens` | integer | `2000` | Maximum tokens per request |
| `ai_service.temperature` | float | `0.1` | Temperature for AI responses |
| `ai_service.retry_attempts` | integer | `3` | Number of retry attempts |

### Cache Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cache.enabled` | boolean | `true` | Enable response caching |
| `cache.max_size` | integer | `1000` | Maximum cache entries |
| `cache.ttl_seconds` | integer | `null` | Cache entry TTL (null = no expiration) |

### Confidence Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `confidence.threshold` | integer | `70` | Minimum confidence to report violation |
| `confidence.high_confidence` | integer | `85` | Threshold for high confidence |
| `confidence.review_threshold` | integer | `60` | Below this, flag for human review |
| `confidence.agreement_boost` | integer | `15` | Boost when AI and rules agree |
| `confidence.disagreement_penalty` | integer | `20` | Penalty when they disagree |

### Feature Flags

Enable/disable specific AI checks:

| Feature | Default | Description |
|---------|---------|-------------|
| `features.enable_promotional_ai` | `true` | AI for promotional document detection |
| `features.enable_performance_ai` | `true` | AI for performance claims analysis |
| `features.enable_prospectus_ai` | `true` | AI for prospectus matching |
| `features.enable_registration_ai` | `true` | AI for registration validation |
| `features.enable_general_ai` | `true` | AI for general rules |
| `features.enable_values_ai` | `true` | AI for values/securities checks |
| `features.enable_esg_ai` | `true` | AI for ESG compliance |
| `features.enable_disclaimers_ai` | `true` | AI for disclaimer validation |

## Migration Guide

### From Legacy System

1. **No changes required**: The system is fully backward compatible
2. **Optional: Enable AI**: Add `--hybrid-mode=on` to command line
3. **Optional: Adjust config**: Modify `hybrid_config.json` as needed
4. **Optional: Use enhanced output**: Set `use_legacy_format: false`

### Gradual Rollout

Use enhancement levels for gradual rollout:

1. **Start with `minimal`**: Test AI on critical checks only
2. **Move to `standard`**: Enable AI for most checks
3. **Upgrade to `full`**: Enable all AI features
4. **Optional `aggressive`**: AI-first mode for maximum accuracy

### Testing

Test the integration:

```bash
# Test with rules only
python check.py exemple.json --rules-only

# Test with hybrid mode
python check.py exemple.json --hybrid-mode=on

# Compare results
diff exemple_violations.json exemple_violations_hybrid.json
```

## Performance Metrics

View performance metrics:

```bash
python check.py exemple.json --show-metrics
```

Output includes:

- **Cache hit rate**: Percentage of cached responses
- **Total requests**: Number of AI API calls
- **Average processing time**: Time per check
- **AI calls**: Number of AI API calls made

## Troubleshooting

### AI Not Available

If AI is not available, the system automatically falls back to rules:

```
⚠ Hybrid mode not available, using rules only
```

**Solutions**:
- Check API keys in `.env` file
- Verify network connectivity
- Check `hybrid_config.json` settings

### Low Confidence Violations

Violations with confidence < 70% are flagged for review:

```json
{
  "confidence": 65,
  "needs_review": true,
  "status": "AI_DETECTED_VARIATION"
}
```

**Actions**:
- Review manually
- Adjust `confidence.threshold` in config
- Provide feedback to improve future predictions

### Performance Issues

If processing is slow:

1. **Enable caching**: Set `cache.enabled: true`
2. **Reduce batch size**: Lower `batch_size`
3. **Disable async**: Set `features.enable_async_processing: false`
4. **Use minimal mode**: Set `enhancement_level: "minimal"`

## API Reference

### Python API

```python
from check_hybrid import get_hybrid_integration
from config_manager import get_config_manager, EnhancementLevel

# Get integration
integration = get_hybrid_integration()

# Check if hybrid mode is available
if integration.is_hybrid_enabled():
    print("Hybrid mode active")

# Update configuration
integration.update_config(
    ai_enabled=True,
    confidence_threshold=80
)

# Get metrics
cache_stats = integration.get_cache_stats()
perf_metrics = integration.get_performance_metrics()

# Configuration management
config_manager = get_config_manager()
config_manager.set_enhancement_level(EnhancementLevel.FULL)
config_manager.print_summary()
```

## Best Practices

1. **Start conservative**: Begin with `enhancement_level: "minimal"`
2. **Monitor metrics**: Use `--show-metrics` to track performance
3. **Review low confidence**: Manually review violations with confidence < 70%
4. **Enable caching**: Always enable caching for better performance
5. **Use fallback**: Keep `fallback_to_rules: true` for reliability
6. **Test thoroughly**: Compare AI vs rules results before full deployment
7. **Adjust thresholds**: Tune confidence thresholds based on your needs

## Support

For issues or questions:

1. Check this guide
2. Review `hybrid_config.json` settings
3. Check logs for error messages
4. Test with `--rules-only` to isolate AI issues
5. Verify API keys and network connectivity

## Version History

- **v1.0**: Initial hybrid integration
  - Three-layer architecture
  - Backward compatibility
  - Configuration system
  - Performance optimization
  - Error handling
