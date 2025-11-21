# Configuration Guide - AI-Enhanced Compliance Checker

## Overview

This guide covers all configuration options for the AI-enhanced compliance checking system, including AI service setup, performance tuning, and feature customization.

## Quick Start Configuration

### Minimum Required Configuration

```python
# .env file
AI_API_KEY=your_gemini_api_key_here
AI_MODEL=gemini-pro
```

```python
# Python code
from hybrid_compliance_checker import HybridComplianceChecker
from ai_engine import AIEngine

# Minimal setup - uses defaults
checker = HybridComplianceChecker(
    ai_engine=AIEngine(),  # Reads API key from environment
    rule_engine=RuleEngine(),
    confidence_scorer=ConfidenceScorer()
)
```

## AI Service Configuration

### Supported AI Services

#### Google Gemini (Recommended)

```python
# .env
AI_SERVICE=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-pro
GEMINI_TIMEOUT=30
GEMINI_MAX_RETRIES=3
```

```python
# config.json
{
  "ai_service": {
    "provider": "gemini",
    "api_key": "${GEMINI_API_KEY}",
    "model": "gemini-pro",
    "timeout": 30,
    "max_retries": 3,
    "temperature": 0.1,
    "max_tokens": 2048
  }
}
```

#### OpenAI (Alternative)

```python
# .env
AI_SERVICE=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4
OPENAI_TIMEOUT=30
```

```python
# config.json
{
  "ai_service": {
    "provider": "openai",
    "api_key": "${OPENAI_API_KEY}",
    "model": "gpt-4",
    "timeout": 30,
    "max_retries": 3,
    "temperature": 0.1
  }
}
```

### AI Engine Parameters

```python
ai_config = {
    "provider": "gemini",           # AI service provider
    "api_key": "your_key",          # API authentication key
    "model": "gemini-pro",          # Model identifier
    "timeout": 30,                  # Request timeout (seconds)
    "max_retries": 3,               # Retry attempts on failure
    "temperature": 0.1,             # Response randomness (0-1)
    "max_tokens": 2048,             # Maximum response length
    "cache_enabled": True,          # Enable response caching
    "cache_ttl": 3600,              # Cache time-to-live (seconds)
    "batch_enabled": True,          # Enable batch processing
    "batch_size": 10,               # Documents per batch
    "async_enabled": True,          # Enable async processing
    "max_concurrent": 5             # Max concurrent requests
}

ai_engine = AIEngine(config=ai_config)
```

## Confidence Scoring Configuration

### Threshold Settings

```python
confidence_config = {
    "thresholds": {
        "min_confidence": 70,       # Minimum acceptable confidence
        "review_threshold": 85,     # Flag for human review below this
        "high_confidence": 95,      # High confidence threshold
        "critical_threshold": 90    # Critical violation threshold
    },
    "calibration": {
        "enabled": True,            # Enable automatic calibration
        "min_samples": 100,         # Minimum samples before calibration
        "adjustment_rate": 0.1,     # How quickly to adjust (0-1)
        "recalibration_interval": 1000  # Recalibrate every N checks
    },
    "scoring": {
        "ai_weight": 0.7,           # Weight for AI results (0-1)
        "rule_weight": 0.3,         # Weight for rule results (0-1)
        "agreement_boost": 15,      # Confidence boost when both agree
        "disagreement_penalty": 10  # Confidence penalty on disagreement
    }
}

scorer = ConfidenceScorer(config=confidence_config)
```

### Status Classification Rules

```python
status_config = {
    "verified_both_min": 90,        # Min confidence for VERIFIED_BY_BOTH
    "ai_variation_max": 84,         # Max confidence for AI_DETECTED_VARIATION
    "needs_review_max": 70,         # Max confidence for NEEDS_REVIEW
    "false_positive_conditions": {
        "rule_found": True,
        "ai_violation": False,
        "min_ai_confidence": 80
    }
}
```

## Rule Engine Configuration

### Rule Strictness Levels

```python
rule_config = {
    "mode": "balanced",             # strict, balanced, or lenient
    "strict_mode": {
        "exact_matching": True,
        "case_sensitive": True,
        "no_fuzzy_matching": True
    },
    "balanced_mode": {
        "exact_matching": False,
        "case_sensitive": False,
        "fuzzy_threshold": 0.8
    },
    "lenient_mode": {
        "exact_matching": False,
        "case_sensitive": False,
        "fuzzy_threshold": 0.6
    }
}

rule_engine = RuleEngine(config=rule_config)
```

### Custom Rule Sets

```python
# Load custom rules from JSON files
custom_rules = {
    "promotional": "rules/promotional_rules.json",
    "performance": "rules/performance_rules.json",
    "disclaimers": "rules/disclaimer_rules.json"
}

rule_engine = RuleEngine(custom_rules=custom_rules)
```

## Performance Configuration

### Caching Settings

```python
cache_config = {
    "enabled": True,
    "backend": "memory",            # memory, redis, or file
    "max_size": 1000,               # Maximum cache entries
    "ttl": 3600,                    # Time-to-live in seconds
    "eviction_policy": "lru",       # lru, lfu, or fifo
    "key_strategy": "content_hash"  # content_hash or prompt_hash
}

# Redis cache (for distributed systems)
cache_config_redis = {
    "enabled": True,
    "backend": "redis",
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0,
    "redis_password": None,
    "ttl": 3600
}
```

### Batch Processing

```python
batch_config = {
    "enabled": True,
    "batch_size": 10,               # Documents per batch
    "max_wait_time": 5,             # Max seconds to wait for batch
    "parallel_batches": 3,          # Number of parallel batches
    "retry_failed": True,           # Retry failed items individually
    "preserve_order": False         # Maintain input order (slower)
}
```

### Async Processing

```python
async_config = {
    "enabled": True,
    "max_concurrent": 5,            # Max concurrent AI requests
    "timeout": 30,                  # Per-request timeout
    "queue_size": 100,              # Max queued requests
    "worker_threads": 4             # Background worker threads
}
```

## Feature Flags

### Gradual Rollout Configuration

```python
feature_flags = {
    "ai_enhanced_checks": {
        "promotional_mention": True,
        "performance_claims": True,
        "fund_name_match": True,
        "disclaimer_validation": True,
        "registration_compliance": False,  # Not yet enabled
        "structure_validation": False,
        "general_rules": False,
        "values_securities": False
    },
    "experimental": {
        "pattern_detection": True,
        "auto_rule_generation": False,
        "multi_model_ensemble": False
    }
}

checker = HybridComplianceChecker(
    ai_engine=ai_engine,
    rule_engine=rule_engine,
    confidence_scorer=scorer,
    feature_flags=feature_flags
)
```

## Environment-Specific Configurations

### Development Environment

```python
# config/development.json
{
  "ai_service": {
    "provider": "gemini",
    "model": "gemini-pro",
    "timeout": 60,
    "cache_enabled": True,
    "mock_responses": True          # Use mock responses for testing
  },
  "confidence": {
    "min_confidence": 60,           # Lower threshold for testing
    "calibration_enabled": False
  },
  "logging": {
    "level": "DEBUG",
    "detailed_prompts": True,
    "save_responses": True
  }
}
```

### Production Environment

```python
# config/production.json
{
  "ai_service": {
    "provider": "gemini",
    "model": "gemini-pro",
    "timeout": 30,
    "max_retries": 3,
    "cache_enabled": True,
    "cache_backend": "redis"
  },
  "confidence": {
    "min_confidence": 70,
    "review_threshold": 85,
    "calibration_enabled": True
  },
  "performance": {
    "batch_enabled": True,
    "async_enabled": True,
    "max_concurrent": 10
  },
  "monitoring": {
    "enabled": True,
    "metrics_endpoint": "http://metrics.internal/api",
    "alert_on_errors": True
  },
  "logging": {
    "level": "INFO",
    "detailed_prompts": False,
    "save_responses": False
  }
}
```

### Testing Environment

```python
# config/testing.json
{
  "ai_service": {
    "provider": "mock",             # Use mock AI service
    "mock_responses_file": "tests/mock_responses.json"
  },
  "confidence": {
    "min_confidence": 50,
    "calibration_enabled": False
  },
  "performance": {
    "cache_enabled": False,         # Disable cache for testing
    "batch_enabled": False,
    "async_enabled": False
  }
}
```

## Loading Configuration

### From File

```python
import json
from hybrid_compliance_checker import HybridComplianceChecker

# Load configuration from JSON file
with open('config/production.json', 'r') as f:
    config = json.load(f)

checker = HybridComplianceChecker.from_config(config)
```

### From Environment Variables

```python
import os
from hybrid_compliance_checker import HybridComplianceChecker

# Configuration from environment
config = {
    "ai_service": {
        "provider": os.getenv("AI_SERVICE", "gemini"),
        "api_key": os.getenv("AI_API_KEY"),
        "model": os.getenv("AI_MODEL", "gemini-pro"),
        "timeout": int(os.getenv("AI_TIMEOUT", "30"))
    },
    "confidence": {
        "min_confidence": int(os.getenv("MIN_CONFIDENCE", "70"))
    }
}

checker = HybridComplianceChecker.from_config(config)
```

### Hybrid Approach (Recommended)

```python
# config_manager.py
import os
import json

class ConfigManager:
    def __init__(self, config_file=None, env_prefix="COMPLIANCE_"):
        self.config = {}
        
        # Load from file if provided
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        
        # Override with environment variables
        self._load_from_env(env_prefix)
    
    def _load_from_env(self, prefix):
        # Override config with environment variables
        if os.getenv(f"{prefix}AI_API_KEY"):
            self.config.setdefault("ai_service", {})
            self.config["ai_service"]["api_key"] = os.getenv(f"{prefix}AI_API_KEY")
        
        if os.getenv(f"{prefix}MIN_CONFIDENCE"):
            self.config.setdefault("confidence", {})
            self.config["confidence"]["min_confidence"] = int(
                os.getenv(f"{prefix}MIN_CONFIDENCE")
            )
    
    def get_config(self):
        return self.config

# Usage
config_manager = ConfigManager(
    config_file="config/production.json",
    env_prefix="COMPLIANCE_"
)
checker = HybridComplianceChecker.from_config(config_manager.get_config())
```

## Monitoring and Logging Configuration

### Logging Setup

```python
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "compliance_checker.log",
            "level": "DEBUG",
            "formatter": "detailed"
        }
    },
    "loggers": {
        "hybrid_compliance_checker": {
            "level": "DEBUG",
            "handlers": ["console", "file"]
        },
        "ai_engine": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
}

import logging.config
logging.config.dictConfig(logging_config)
```

### Performance Monitoring

```python
monitoring_config = {
    "enabled": True,
    "metrics": {
        "processing_time": True,
        "api_calls": True,
        "cache_hit_rate": True,
        "confidence_distribution": True,
        "error_rate": True
    },
    "alerting": {
        "enabled": True,
        "error_threshold": 0.05,        # Alert if error rate > 5%
        "latency_threshold": 5000,      # Alert if avg latency > 5s
        "confidence_threshold": 0.7     # Alert if avg confidence < 70%
    },
    "export": {
        "format": "prometheus",         # prometheus, json, or csv
        "endpoint": "/metrics",
        "interval": 60                  # Export every 60 seconds
    }
}

from performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor(config=monitoring_config)
```

## Security Configuration

### API Key Management

```python
# Use environment variables (recommended)
AI_API_KEY=your_key_here

# Or use secrets management service
from secrets_manager import SecretsManager

secrets = SecretsManager(provider="aws_secrets_manager")
api_key = secrets.get_secret("compliance_checker/ai_api_key")

ai_engine = AIEngine(api_key=api_key)
```

### Rate Limiting

```python
rate_limit_config = {
    "enabled": True,
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "burst_size": 10,
    "strategy": "token_bucket"      # token_bucket or sliding_window
}
```

## Complete Configuration Example

```python
# complete_config.json
{
  "ai_service": {
    "provider": "gemini",
    "api_key": "${GEMINI_API_KEY}",
    "model": "gemini-pro",
    "timeout": 30,
    "max_retries": 3,
    "temperature": 0.1,
    "cache_enabled": true,
    "cache_ttl": 3600,
    "batch_enabled": true,
    "batch_size": 10
  },
  "rule_engine": {
    "mode": "balanced",
    "custom_rules": {
      "promotional": "rules/promotional_rules.json",
      "performance": "rules/performance_rules.json"
    }
  },
  "confidence": {
    "thresholds": {
      "min_confidence": 70,
      "review_threshold": 85,
      "high_confidence": 95
    },
    "calibration": {
      "enabled": true,
      "min_samples": 100
    },
    "scoring": {
      "ai_weight": 0.7,
      "rule_weight": 0.3,
      "agreement_boost": 15
    }
  },
  "performance": {
    "cache": {
      "enabled": true,
      "backend": "redis",
      "redis_host": "localhost",
      "max_size": 1000
    },
    "async": {
      "enabled": true,
      "max_concurrent": 5
    }
  },
  "feature_flags": {
    "ai_enhanced_checks": {
      "promotional_mention": true,
      "performance_claims": true,
      "fund_name_match": true
    }
  },
  "monitoring": {
    "enabled": true,
    "metrics_endpoint": "/metrics",
    "alerting_enabled": true
  },
  "logging": {
    "level": "INFO",
    "file": "compliance_checker.log"
  }
}
```

## Configuration Validation

```python
from config_validator import ConfigValidator

validator = ConfigValidator()

try:
    validator.validate_config(config)
    print("Configuration is valid")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Suggestions: {e.suggestions}")
```

## Best Practices

1. **Use Environment Variables for Secrets**: Never commit API keys to version control
2. **Start with Defaults**: Use default configuration and customize only what's needed
3. **Test Configuration Changes**: Validate in development before deploying to production
4. **Monitor Performance**: Track metrics to optimize configuration over time
5. **Enable Caching**: Significantly reduces API costs and improves performance
6. **Use Feature Flags**: Enable new features gradually to minimize risk
7. **Configure Logging Appropriately**: DEBUG in development, INFO in production
8. **Set Appropriate Timeouts**: Balance between reliability and performance
9. **Enable Calibration**: Improves accuracy over time based on actual usage
10. **Document Custom Settings**: Keep track of why specific values were chosen

For troubleshooting configuration issues, see TROUBLESHOOTING_GUIDE.md.
