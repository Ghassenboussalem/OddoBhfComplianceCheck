#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Manager - AI/Rule Balance Configuration System
Manages runtime configuration with validation and feature flags
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancementLevel(Enum):
    """AI enhancement levels"""
    DISABLED = "disabled"  # Rules only, no AI
    MINIMAL = "minimal"    # AI for critical checks only
    STANDARD = "standard"  # AI for most checks
    FULL = "full"          # AI for all checks
    AGGRESSIVE = "aggressive"  # AI-first with minimal rule validation


@dataclass
class AIServiceConfig:
    """Configuration for AI service"""
    provider: str = "auto"  # auto, token_factory, gemini
    model_name: str = "auto"
    api_key_env_var: str = ""
    timeout: int = 30
    max_tokens: int = 2000
    temperature: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class CacheConfig:
    """Configuration for caching"""
    enabled: bool = True
    max_size: int = 1000
    ttl_seconds: Optional[int] = None
    invalidate_on_rule_update: bool = True


@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring"""
    threshold: int = 70
    high_confidence: int = 85
    medium_confidence: int = 70
    review_threshold: int = 60
    agreement_boost: int = 15
    disagreement_penalty: int = 20


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout"""
    enable_promotional_ai: bool = True
    enable_performance_ai: bool = True
    enable_prospectus_ai: bool = True
    enable_registration_ai: bool = True
    enable_general_ai: bool = True
    enable_values_ai: bool = True
    enable_esg_ai: bool = True
    enable_disclaimers_ai: bool = True
    enable_caching: bool = True
    enable_batch_processing: bool = True
    enable_async_processing: bool = False
    enable_performance_monitoring: bool = True
    enable_error_handling: bool = True
    enable_feedback_loop: bool = False


@dataclass
class ComplianceConfig:
    """Complete compliance checker configuration"""
    # Core settings
    ai_enabled: bool = True
    rule_enabled: bool = True
    enhancement_level: str = "full"
    
    # AI service
    ai_service: AIServiceConfig = None
    
    # Caching
    cache: CacheConfig = None
    
    # Confidence scoring
    confidence: ConfidenceConfig = None
    
    # Feature flags
    features: FeatureFlags = None
    
    # Backward compatibility
    backward_compatible: bool = True
    use_legacy_format: bool = False
    
    # Processing
    batch_size: int = 5
    fallback_to_rules: bool = True
    
    def __post_init__(self):
        """Initialize nested configs if not provided"""
        if self.ai_service is None:
            self.ai_service = AIServiceConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.confidence is None:
            self.confidence = ConfidenceConfig()
        if self.features is None:
            self.features = FeatureFlags()


class ConfigManager:
    """
    Manages configuration for hybrid compliance checker
    Supports loading from file, environment variables, and runtime updates
    """
    
    def __init__(self, config_path: str = "hybrid_config.json"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        logger.info("ConfigManager initialized")
    
    def _load_config(self) -> ComplianceConfig:
        """Load configuration from file and environment"""
        # Start with defaults
        config_dict = self._get_default_config_dict()
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                config_dict.update(file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        else:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
        
        # Override with environment variables
        self._apply_env_overrides(config_dict)
        
        # Convert to ComplianceConfig
        return self._dict_to_config(config_dict)
    
    def _get_default_config_dict(self) -> Dict:
        """Get default configuration as dict"""
        default_config = ComplianceConfig()
        return self._config_to_dict(default_config)
    
    def _config_to_dict(self, config: ComplianceConfig) -> Dict:
        """Convert ComplianceConfig to dict"""
        return {
            'ai_enabled': config.ai_enabled,
            'rule_enabled': config.rule_enabled,
            'enhancement_level': config.enhancement_level,
            'ai_service': asdict(config.ai_service),
            'cache': asdict(config.cache),
            'confidence': asdict(config.confidence),
            'features': asdict(config.features),
            'backward_compatible': config.backward_compatible,
            'use_legacy_format': config.use_legacy_format,
            'batch_size': config.batch_size,
            'fallback_to_rules': config.fallback_to_rules
        }
    
    def _dict_to_config(self, config_dict: Dict) -> ComplianceConfig:
        """Convert dict to ComplianceConfig"""
        # Extract nested configs
        ai_service_dict = config_dict.get('ai_service', {})
        cache_dict = config_dict.get('cache', {})
        confidence_dict = config_dict.get('confidence', {})
        features_dict = config_dict.get('features', {})
        
        return ComplianceConfig(
            ai_enabled=config_dict.get('ai_enabled', True),
            rule_enabled=config_dict.get('rule_enabled', True),
            enhancement_level=config_dict.get('enhancement_level', 'full'),
            ai_service=AIServiceConfig(**ai_service_dict) if ai_service_dict else AIServiceConfig(),
            cache=CacheConfig(**cache_dict) if cache_dict else CacheConfig(),
            confidence=ConfidenceConfig(**confidence_dict) if confidence_dict else ConfidenceConfig(),
            features=FeatureFlags(**features_dict) if features_dict else FeatureFlags(),
            backward_compatible=config_dict.get('backward_compatible', True),
            use_legacy_format=config_dict.get('use_legacy_format', False),
            batch_size=config_dict.get('batch_size', 5),
            fallback_to_rules=config_dict.get('fallback_to_rules', True)
        )
    
    def _apply_env_overrides(self, config_dict: Dict):
        """Apply environment variable overrides"""
        # AI enabled
        if os.getenv('HYBRID_AI_ENABLED'):
            config_dict['ai_enabled'] = os.getenv('HYBRID_AI_ENABLED').lower() == 'true'
        
        # Enhancement level
        if os.getenv('HYBRID_ENHANCEMENT_LEVEL'):
            config_dict['enhancement_level'] = os.getenv('HYBRID_ENHANCEMENT_LEVEL')
        
        # Confidence threshold
        if os.getenv('HYBRID_CONFIDENCE_THRESHOLD'):
            if 'confidence' not in config_dict:
                config_dict['confidence'] = {}
            config_dict['confidence']['threshold'] = int(os.getenv('HYBRID_CONFIDENCE_THRESHOLD'))
        
        # Cache enabled
        if os.getenv('HYBRID_CACHE_ENABLED'):
            if 'cache' not in config_dict:
                config_dict['cache'] = {}
            config_dict['cache']['enabled'] = os.getenv('HYBRID_CACHE_ENABLED').lower() == 'true'
        
        logger.info("Applied environment variable overrides")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate enhancement level
        valid_levels = [level.value for level in EnhancementLevel]
        if self.config.enhancement_level not in valid_levels:
            errors.append(f"Invalid enhancement_level: {self.config.enhancement_level}. Must be one of {valid_levels}")
        
        # Validate confidence thresholds
        if not 0 <= self.config.confidence.threshold <= 100:
            errors.append(f"confidence.threshold must be 0-100, got {self.config.confidence.threshold}")
        
        if not 0 <= self.config.confidence.high_confidence <= 100:
            errors.append(f"confidence.high_confidence must be 0-100, got {self.config.confidence.high_confidence}")
        
        # Validate cache size
        if self.config.cache.max_size < 0:
            errors.append(f"cache.max_size must be >= 0, got {self.config.cache.max_size}")
        
        # Validate batch size
        if self.config.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.config.batch_size}")
        
        # Validate AI service timeout
        if self.config.ai_service.timeout < 1:
            errors.append(f"ai_service.timeout must be >= 1, got {self.config.ai_service.timeout}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get_config(self) -> ComplianceConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """
        Update configuration at runtime
        
        Args:
            **kwargs: Configuration values to update
        """
        config_dict = self._config_to_dict(self.config)
        
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys (e.g., 'confidence.threshold')
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
            
            logger.info(f"Updated config: {key} = {value}")
        
        # Recreate config object
        self.config = self._dict_to_config(config_dict)
        
        # Validate updated config
        self._validate_config()
    
    def save_config(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: Path to save to (uses self.config_path if None)
        """
        save_path = path or self.config_path
        
        try:
            config_dict = self._config_to_dict(self.config)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def set_enhancement_level(self, level: EnhancementLevel):
        """
        Set AI enhancement level
        
        Args:
            level: Enhancement level enum
        """
        self.update_config(enhancement_level=level.value)
        
        # Update feature flags based on level
        if level == EnhancementLevel.DISABLED:
            self.update_config(ai_enabled=False)
        elif level == EnhancementLevel.MINIMAL:
            self.update_config(ai_enabled=True)
            # Enable only critical checks
            self.config.features.enable_promotional_ai = True
            self.config.features.enable_performance_ai = True
            self.config.features.enable_prospectus_ai = False
            self.config.features.enable_registration_ai = False
            self.config.features.enable_general_ai = False
            self.config.features.enable_values_ai = False
            self.config.features.enable_esg_ai = False
            self.config.features.enable_disclaimers_ai = True
        elif level == EnhancementLevel.STANDARD:
            self.update_config(ai_enabled=True)
            # Enable most checks
            self.config.features.enable_promotional_ai = True
            self.config.features.enable_performance_ai = True
            self.config.features.enable_prospectus_ai = True
            self.config.features.enable_registration_ai = True
            self.config.features.enable_general_ai = True
            self.config.features.enable_values_ai = False
            self.config.features.enable_esg_ai = False
            self.config.features.enable_disclaimers_ai = True
        elif level == EnhancementLevel.FULL:
            self.update_config(ai_enabled=True)
            # Enable all checks
            for attr in dir(self.config.features):
                if attr.startswith('enable_') and attr.endswith('_ai'):
                    setattr(self.config.features, attr, True)
        elif level == EnhancementLevel.AGGRESSIVE:
            self.update_config(ai_enabled=True, rule_enabled=False)
            # Enable all AI checks, disable rule validation
            for attr in dir(self.config.features):
                if attr.startswith('enable_') and attr.endswith('_ai'):
                    setattr(self.config.features, attr, True)
        
        logger.info(f"Enhancement level set to {level.value}")
    
    def is_check_enabled(self, check_type: str) -> bool:
        """
        Check if AI is enabled for a specific check type
        
        Args:
            check_type: Check type (e.g., 'promotional', 'performance')
            
        Returns:
            True if AI is enabled for this check
        """
        if not self.config.ai_enabled:
            return False
        
        feature_name = f"enable_{check_type.lower()}_ai"
        return getattr(self.config.features, feature_name, False)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'ai_enabled': self.config.ai_enabled,
            'rule_enabled': self.config.rule_enabled,
            'enhancement_level': self.config.enhancement_level,
            'confidence_threshold': self.config.confidence.threshold,
            'cache_enabled': self.config.cache.enabled,
            'cache_size': self.config.cache.max_size,
            'fallback_to_rules': self.config.fallback_to_rules,
            'backward_compatible': self.config.backward_compatible,
            'ai_checks_enabled': sum(1 for attr in dir(self.config.features) 
                                    if attr.startswith('enable_') and attr.endswith('_ai') 
                                    and getattr(self.config.features, attr))
        }
    
    def print_summary(self):
        """Print configuration summary"""
        summary = self.get_summary()
        
        print("="*70)
        print("HYBRID COMPLIANCE CHECKER CONFIGURATION")
        print("="*70)
        print(f"\nMode: {'AI + Rules' if summary['ai_enabled'] and summary['rule_enabled'] else 'Rules Only' if summary['rule_enabled'] else 'AI Only'}")
        print(f"Enhancement Level: {summary['enhancement_level'].upper()}")
        print(f"Confidence Threshold: {summary['confidence_threshold']}%")
        print(f"Cache: {'Enabled' if summary['cache_enabled'] else 'Disabled'} (max size: {summary['cache_size']})")
        print(f"Fallback to Rules: {'Yes' if summary['fallback_to_rules'] else 'No'}")
        print(f"Backward Compatible: {'Yes' if summary['backward_compatible'] else 'No'}")
        print(f"AI Checks Enabled: {summary['ai_checks_enabled']}/8")
        print("="*70)


# Global instance
_config_manager = None


def get_config_manager(config_path: str = "hybrid_config.json") -> ConfigManager:
    """
    Get or create global configuration manager
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


if __name__ == "__main__":
    # Test configuration manager
    print("="*70)
    print("Configuration Manager - Test")
    print("="*70)
    
    # Create manager
    manager = get_config_manager()
    
    # Print summary
    manager.print_summary()
    
    # Test enhancement levels
    print("\n📊 Testing Enhancement Levels:")
    for level in EnhancementLevel:
        print(f"\n  {level.value.upper()}:")
        manager.set_enhancement_level(level)
        print(f"    AI Enabled: {manager.config.ai_enabled}")
        print(f"    Rule Enabled: {manager.config.rule_enabled}")
        print(f"    Promotional AI: {manager.config.features.enable_promotional_ai}")
        print(f"    Performance AI: {manager.config.features.enable_performance_ai}")
    
    # Reset to full
    manager.set_enhancement_level(EnhancementLevel.FULL)
    
    # Test runtime updates
    print("\n📊 Testing Runtime Updates:")
    print(f"  Before: confidence.threshold = {manager.config.confidence.threshold}")
    manager.update_config(**{'confidence.threshold': 80})
    print(f"  After: confidence.threshold = {manager.config.confidence.threshold}")
    
    # Test check enabled
    print("\n📊 Testing Check Enabled:")
    print(f"  Promotional AI: {manager.is_check_enabled('promotional')}")
    print(f"  Performance AI: {manager.is_check_enabled('performance')}")
    
    print("\n" + "="*70)
    print("✓ Configuration manager tests complete")
    print("="*70)
