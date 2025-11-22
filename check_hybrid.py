#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Compliance Checker Integration Module
Provides backward-compatible interface for check.py
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridCheckIntegration:
    """
    Integration layer between existing check.py and HybridComplianceChecker
    Maintains backward compatibility while enabling AI enhancements
    """
    
    def __init__(self, config_path: str = "hybrid_config.json"):
        """
        Initialize hybrid integration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.hybrid_checker = None
        self.ai_engine = None
        self.confidence_scorer = None
        
        # Initialize components if AI is enabled
        if self.config.get('ai_enabled', False):
            self._initialize_hybrid_components()
        
        logger.info(f"HybridCheckIntegration initialized (AI: {self.config.get('ai_enabled', False)})")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file using ConfigManager"""
        try:
            from config_manager import get_config_manager
            
            config_manager = get_config_manager(config_path)
            compliance_config = config_manager.get_config()
            
            # Convert to simple dict for backward compatibility
            config = {
                'ai_enabled': compliance_config.ai_enabled,
                'rule_enabled': compliance_config.rule_enabled,
                'confidence_threshold': compliance_config.confidence.threshold,
                'ai_timeout': compliance_config.ai_service.timeout,
                'cache_enabled': compliance_config.cache.enabled,
                'cache_size': compliance_config.cache.max_size,
                'fallback_to_rules': compliance_config.fallback_to_rules,
                'batch_size': compliance_config.batch_size,
                'enhancement_level': compliance_config.enhancement_level,
                'backward_compatible': compliance_config.backward_compatible,
                'use_legacy_format': compliance_config.use_legacy_format,
                'enable_performance_monitoring': compliance_config.features.enable_performance_monitoring,
                'enable_error_handling': compliance_config.features.enable_error_handling,
                'enable_feedback_loop': compliance_config.features.enable_feedback_loop,
                'hitl': getattr(compliance_config, 'hitl', {})
            }
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'ai_enabled': False,
            'rule_enabled': True,
            'confidence_threshold': 70,
            'fallback_to_rules': True,
            'backward_compatible': True,
            'use_legacy_format': True
        }
    
    def _initialize_hybrid_components(self):
        """Initialize hybrid checker components"""
        try:
            from hybrid_compliance_checker import HybridComplianceChecker, HybridConfig
            from ai_engine import create_ai_engine_from_env
            from confidence_scorer import ConfidenceScorer
            
            # Create AI engine
            self.ai_engine = create_ai_engine_from_env()
            
            if not self.ai_engine:
                logger.warning("AI engine initialization failed, falling back to rule-only mode")
                self.config['ai_enabled'] = False
                return
            
            # Create confidence scorer
            self.confidence_scorer = ConfidenceScorer()
            
            # Create hybrid config
            hybrid_config = HybridConfig(
                ai_enabled=self.config.get('ai_enabled', True),
                rule_enabled=self.config.get('rule_enabled', True),
                confidence_threshold=self.config.get('confidence_threshold', 70),
                ai_timeout=self.config.get('ai_timeout', 30),
                cache_enabled=self.config.get('cache_enabled', True),
                fallback_to_rules=self.config.get('fallback_to_rules', True),
                batch_size=self.config.get('batch_size', 5)
            )
            
            # Initialize error handler if enabled
            error_handler = None
            if self.config.get('enable_error_handling', True):
                try:
                    from error_handler import ErrorHandler
                    error_handler = ErrorHandler()
                except ImportError:
                    logger.warning("Error handler not available")
            
            # Initialize performance monitor if enabled
            performance_monitor = None
            if self.config.get('enable_performance_monitoring', True):
                try:
                    from performance_monitor import PerformanceMonitor
                    performance_monitor = PerformanceMonitor()
                except ImportError:
                    logger.warning("Performance monitor not available")
            
            # Initialize review manager if HITL is enabled
            review_manager = None
            hitl_config = self.config.get('hitl', {})
            if hitl_config.get('enabled', False):
                try:
                    from review_manager import ReviewManager
                    review_manager = ReviewManager(
                        queue_file="review_queue.json",
                        max_queue_size=hitl_config.get('queue_max_size', 10000)
                    )
                    logger.info("Review manager initialized for HITL")
                except ImportError:
                    logger.warning("Review manager not available")
            
            # Create hybrid checker
            self.hybrid_checker = HybridComplianceChecker(
                ai_engine=self.ai_engine,
                confidence_scorer=self.confidence_scorer,
                config=hybrid_config,
                error_handler=error_handler,
                performance_monitor=performance_monitor,
                review_manager=review_manager
            )
            
            logger.info("Hybrid components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid components: {e}")
            self.config['ai_enabled'] = False
            self.hybrid_checker = None
    
    def is_hybrid_enabled(self) -> bool:
        """Check if hybrid mode is enabled and available"""
        return self.config.get('ai_enabled', False) and self.hybrid_checker is not None
    
    def convert_to_legacy_format(self, result: Any) -> Dict:
        """
        Convert hybrid result to legacy violation format
        
        Args:
            result: ComplianceResult from hybrid checker
            
        Returns:
            Legacy format violation dict
        """
        if result is None:
            return None
        
        # Handle dict format (already converted)
        if isinstance(result, dict):
            violation = {
                'type': result.get('check_type', 'UNKNOWN').upper(),
                'severity': result.get('severity', 'CRITICAL'),
                'slide': result.get('slide', 'Unknown'),
                'location': result.get('location', 'Unknown'),
                'rule': result.get('rule', 'Unknown rule'),
                'message': result.get('message', 'Violation detected'),
                'evidence': result.get('evidence', ''),
                'confidence': result.get('confidence', 0)
            }
            
            # Add AI-specific fields if not using legacy format
            if not self.config.get('use_legacy_format', False):
                violation['ai_reasoning'] = result.get('ai_reasoning', '')
                violation['status'] = result.get('status', '')
                violation['needs_review'] = result.get('needs_review', False)
            
            return violation
        
        # Handle ComplianceResult object
        violation = {
            'type': str(result.check_type).split('.')[-1].upper() if hasattr(result, 'check_type') else 'UNKNOWN',
            'severity': result.severity if hasattr(result, 'severity') else 'CRITICAL',
            'slide': result.slide if hasattr(result, 'slide') else 'Unknown',
            'location': result.location if hasattr(result, 'location') else 'Unknown',
            'rule': result.rule if hasattr(result, 'rule') else 'Unknown rule',
            'message': result.message if hasattr(result, 'message') else 'Violation detected',
            'evidence': result.evidence if hasattr(result, 'evidence') else '',
            'confidence': result.confidence if hasattr(result, 'confidence') else 0
        }
        
        # Add AI-specific fields if not using legacy format
        if not self.config.get('use_legacy_format', False):
            violation['ai_reasoning'] = result.ai_reasoning if hasattr(result, 'ai_reasoning') else ''
            violation['status'] = result.status.value if hasattr(result, 'status') else ''
            violation['needs_review'] = result.confidence < 70 if hasattr(result, 'confidence') else False
        
        return violation
    
    def check_enhanced(self, check_func, *args, **kwargs) -> List[Dict]:
        """
        Enhanced check wrapper that uses hybrid checker if available
        
        Args:
            check_func: Original check function from agent.py
            *args: Arguments for check function
            **kwargs: Keyword arguments for check function
            
        Returns:
            List of violations in legacy format
        """
        # If hybrid mode is not enabled, use original function
        if not self.is_hybrid_enabled():
            return check_func(*args, **kwargs)
        
        # Try to use hybrid checker
        try:
            # Extract check type from function name
            func_name = check_func.__name__
            
            # Map function names to hybrid check methods
            if 'structure' in func_name.lower():
                results = self.hybrid_checker.check_structure_compliance(*args, **kwargs)
            elif 'performance' in func_name.lower():
                results = self.hybrid_checker.check_performance_compliance(*args, **kwargs)
            elif 'prospectus' in func_name.lower():
                results = self.hybrid_checker.check_prospectus_compliance(*args, **kwargs)
            elif 'registration' in func_name.lower():
                results = self.hybrid_checker.check_registration_compliance(*args, **kwargs)
            elif 'general' in func_name.lower():
                results = self.hybrid_checker.check_general_compliance(*args, **kwargs)
            elif 'values' in func_name.lower() or 'securities' in func_name.lower():
                results = self.hybrid_checker.check_values_compliance(*args, **kwargs)
            elif 'esg' in func_name.lower():
                results = self.hybrid_checker.check_esg_compliance(*args, **kwargs)
            elif 'disclaimer' in func_name.lower():
                results = self.hybrid_checker.check_disclaimers_compliance(*args, **kwargs)
            else:
                # Unknown check type, use original function
                logger.warning(f"Unknown check type for {func_name}, using original function")
                return check_func(*args, **kwargs)
            
            # Convert results to legacy format
            violations = []
            for result in results:
                violation = self.convert_to_legacy_format(result)
                if violation:
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Hybrid check failed: {e}, falling back to original function")
            # Fallback to original function
            return check_func(*args, **kwargs)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics from hybrid checker"""
        if self.hybrid_checker:
            return self.hybrid_checker.get_performance_metrics()
        return {}
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if self.ai_engine:
            return self.ai_engine.get_cache_stats()
        return {}
    
    def print_performance_dashboard(self):
        """Print performance dashboard"""
        if self.hybrid_checker:
            self.hybrid_checker.get_performance_dashboard()
    
    def update_config(self, **kwargs):
        """Update configuration at runtime"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Updated config: {key} = {value}")
        
        # Update hybrid checker config if available
        if self.hybrid_checker:
            self.hybrid_checker.update_config(**kwargs)


# Global instance for easy access
_hybrid_integration = None


def get_hybrid_integration(config_path: str = "hybrid_config.json") -> HybridCheckIntegration:
    """
    Get or create global hybrid integration instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        HybridCheckIntegration instance
    """
    global _hybrid_integration
    
    if _hybrid_integration is None:
        _hybrid_integration = HybridCheckIntegration(config_path)
    
    return _hybrid_integration


def enable_hybrid_mode():
    """Enable hybrid AI+Rules mode"""
    integration = get_hybrid_integration()
    integration.update_config(ai_enabled=True)
    logger.info("Hybrid mode enabled")


def disable_hybrid_mode():
    """Disable hybrid mode (rules only)"""
    integration = get_hybrid_integration()
    integration.update_config(ai_enabled=False)
    logger.info("Hybrid mode disabled (rules only)")


def is_hybrid_available() -> bool:
    """Check if hybrid mode is available"""
    integration = get_hybrid_integration()
    return integration.is_hybrid_enabled()


if __name__ == "__main__":
    # Test integration
    print("="*70)
    print("Hybrid Check Integration - Test")
    print("="*70)
    
    integration = get_hybrid_integration()
    
    print(f"\nHybrid mode enabled: {integration.is_hybrid_enabled()}")
    print(f"Configuration: {json.dumps(integration.config, indent=2)}")
    
    if integration.is_hybrid_enabled():
        print("\n✓ Hybrid components initialized successfully")
        print(f"  AI Engine: {integration.ai_engine is not None}")
        print(f"  Confidence Scorer: {integration.confidence_scorer is not None}")
        print(f"  Hybrid Checker: {integration.hybrid_checker is not None}")
        
        # Test cache stats
        cache_stats = integration.get_cache_stats()
        if cache_stats:
            print(f"\n📊 Cache Stats: {cache_stats}")
    else:
        print("\n⚠ Hybrid mode not available (using rules only)")
    
    print("\n" + "="*70)
