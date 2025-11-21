#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitoring System
Tracks timing metrics, API usage, costs, and accuracy for the hybrid compliance checker
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingLayer(Enum):
    """Processing layers in the hybrid architecture"""
    RULE_PREFILTER = "rule_prefilter"
    AI_ANALYSIS = "ai_analysis"
    VALIDATION = "validation"
    TOTAL = "total"


@dataclass
class LayerMetrics:
    """Metrics for a single processing layer"""
    layer: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    errors: int = 0
    
    def update(self, duration_ms: float, error: bool = False):
        """Update metrics with new timing"""
        self.total_calls += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.avg_time_ms = self.total_time_ms / self.total_calls
        if error:
            self.errors += 1


@dataclass
class APIUsageMetrics:
    """Metrics for API usage and costs"""
    provider: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    cached_calls: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    
    def update(self, tokens: int, latency_ms: float, cached: bool = False, 
               success: bool = True, cost_per_1k_tokens: float = 0.0):
        """Update API usage metrics"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        if cached:
            self.cached_calls += 1
        else:
            self.total_tokens += tokens
            self.estimated_cost_usd += (tokens / 1000.0) * cost_per_1k_tokens
        
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_calls


@dataclass
class AccuracyMetrics:
    """Metrics for tracking accuracy over time"""
    check_type: str
    total_checks: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    human_reviews: int = 0
    corrections: int = 0
    
    @property
    def precision(self) -> float:
        """Calculate precision (TP / (TP + FP))"""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Calculate recall (TP / (TP + FN))"""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score"""
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: str
    layer_metrics: Dict[str, Dict]
    api_metrics: Dict[str, Dict]
    accuracy_metrics: Dict[str, Dict]
    cache_stats: Dict
    error_stats: Dict
    summary: Dict


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    
    Tracks:
    - Timing metrics for each processing layer
    - API usage and cost monitoring
    - Accuracy tracking over time
    - Cache performance
    - Error rates and patterns
    """
    
    def __init__(self, cost_config: Optional[Dict[str, float]] = None):
        """
        Initialize performance monitor
        
        Args:
            cost_config: Dict mapping provider names to cost per 1K tokens
        """
        # Layer timing metrics
        self.layer_metrics: Dict[str, LayerMetrics] = {
            layer.value: LayerMetrics(layer=layer.value)
            for layer in ProcessingLayer
        }
        
        # API usage metrics by provider
        self.api_metrics: Dict[str, APIUsageMetrics] = {}
        
        # Accuracy metrics by check type
        self.accuracy_metrics: Dict[str, AccuracyMetrics] = {}
        
        # Cost configuration (USD per 1K tokens)
        default_config = {
            'token_factory': 0.001,  # Example cost
            'gemini': 0.0005,        # Example cost
            'default': 0.001
        }
        self.cost_config = cost_config if cost_config is not None else default_config
        # Ensure default key exists
        if 'default' not in self.cost_config:
            self.cost_config['default'] = 0.001
        
        # Performance history for trending
        self.history: List[PerformanceSnapshot] = []
        self.max_history_size = 1000
        
        # Start time for uptime tracking
        self.start_time = time.time()
        
        logger.info("PerformanceMonitor initialized")
    
    # ========================================================================
    # TIMING METRICS
    # ========================================================================
    
    def start_timer(self) -> float:
        """
        Start a timer
        
        Returns:
            Start time in seconds
        """
        return time.time()
    
    def record_layer_timing(self, layer: ProcessingLayer, start_time: float, 
                           error: bool = False):
        """
        Record timing for a processing layer
        
        Args:
            layer: Processing layer
            start_time: Start time from start_timer()
            error: Whether an error occurred
        """
        duration_ms = (time.time() - start_time) * 1000
        
        if layer.value in self.layer_metrics:
            self.layer_metrics[layer.value].update(duration_ms, error)
        
        logger.debug(f"{layer.value} completed in {duration_ms:.2f}ms")
    
    def get_layer_metrics(self, layer: Optional[ProcessingLayer] = None) -> Dict:
        """
        Get timing metrics for layer(s)
        
        Args:
            layer: Specific layer or None for all layers
            
        Returns:
            Dict of metrics
        """
        if layer:
            metrics = self.layer_metrics.get(layer.value)
            return asdict(metrics) if metrics else {}
        
        return {
            name: asdict(metrics) 
            for name, metrics in self.layer_metrics.items()
        }
    
    # ========================================================================
    # API USAGE TRACKING
    # ========================================================================
    
    def record_api_call(self, provider: str, tokens: int, latency_ms: float,
                       cached: bool = False, success: bool = True):
        """
        Record an API call
        
        Args:
            provider: API provider name
            tokens: Number of tokens used
            latency_ms: API call latency in milliseconds
            cached: Whether response was cached
            success: Whether call was successful
        """
        if provider not in self.api_metrics:
            self.api_metrics[provider] = APIUsageMetrics(provider=provider)
        
        cost_per_1k = self.cost_config.get(provider, self.cost_config['default'])
        self.api_metrics[provider].update(
            tokens=tokens,
            latency_ms=latency_ms,
            cached=cached,
            success=success,
            cost_per_1k_tokens=cost_per_1k
        )
    
    def get_api_metrics(self, provider: Optional[str] = None) -> Dict:
        """
        Get API usage metrics
        
        Args:
            provider: Specific provider or None for all
            
        Returns:
            Dict of API metrics
        """
        if provider:
            metrics = self.api_metrics.get(provider)
            return asdict(metrics) if metrics else {}
        
        return {
            name: asdict(metrics)
            for name, metrics in self.api_metrics.items()
        }
    
    def get_total_cost(self) -> float:
        """
        Get total estimated cost across all providers
        
        Returns:
            Total cost in USD
        """
        return sum(
            metrics.estimated_cost_usd 
            for metrics in self.api_metrics.values()
        )
    
    def get_cache_efficiency(self) -> Dict[str, float]:
        """
        Calculate cache efficiency metrics
        
        Returns:
            Dict with cache hit rate and cost savings
        """
        total_calls = sum(m.total_calls for m in self.api_metrics.values())
        cached_calls = sum(m.cached_calls for m in self.api_metrics.values())
        
        hit_rate = (cached_calls / total_calls * 100) if total_calls > 0 else 0.0
        
        # Estimate cost savings from caching
        avg_tokens_per_call = 0
        if total_calls > cached_calls:
            total_tokens = sum(m.total_tokens for m in self.api_metrics.values())
            avg_tokens_per_call = total_tokens / (total_calls - cached_calls)
        
        estimated_savings = 0.0
        for provider, metrics in self.api_metrics.items():
            cost_per_1k = self.cost_config.get(provider, self.cost_config['default'])
            estimated_savings += (metrics.cached_calls * avg_tokens_per_call / 1000.0) * cost_per_1k
        
        return {
            'cache_hit_rate': round(hit_rate, 2),
            'total_calls': total_calls,
            'cached_calls': cached_calls,
            'estimated_savings_usd': round(estimated_savings, 4)
        }
    
    # ========================================================================
    # ACCURACY TRACKING
    # ========================================================================
    
    def record_check_result(self, check_type: str, predicted_violation: bool,
                           actual_violation: Optional[bool] = None,
                           human_review: bool = False):
        """
        Record a compliance check result
        
        Args:
            check_type: Type of compliance check
            predicted_violation: Whether system predicted a violation
            actual_violation: Ground truth (if known)
            human_review: Whether result was flagged for human review
        """
        if check_type not in self.accuracy_metrics:
            self.accuracy_metrics[check_type] = AccuracyMetrics(check_type=check_type)
        
        metrics = self.accuracy_metrics[check_type]
        metrics.total_checks += 1
        
        if human_review:
            metrics.human_reviews += 1
        
        # Update confusion matrix if ground truth is known
        if actual_violation is not None:
            if predicted_violation and actual_violation:
                metrics.true_positives += 1
            elif predicted_violation and not actual_violation:
                metrics.false_positives += 1
            elif not predicted_violation and actual_violation:
                metrics.false_negatives += 1
            else:  # not predicted and not actual
                metrics.true_negatives += 1
    
    def record_correction(self, check_type: str, was_correct: bool):
        """
        Record a human correction
        
        Args:
            check_type: Type of check
            was_correct: Whether the system was correct
        """
        if check_type not in self.accuracy_metrics:
            self.accuracy_metrics[check_type] = AccuracyMetrics(check_type=check_type)
        
        self.accuracy_metrics[check_type].corrections += 1
        
        if not was_correct:
            logger.info(f"Correction recorded for {check_type}")
    
    def get_accuracy_metrics(self, check_type: Optional[str] = None) -> Dict:
        """
        Get accuracy metrics
        
        Args:
            check_type: Specific check type or None for all
            
        Returns:
            Dict of accuracy metrics
        """
        if check_type:
            metrics = self.accuracy_metrics.get(check_type)
            if metrics:
                data = asdict(metrics)
                data['precision'] = metrics.precision
                data['recall'] = metrics.recall
                data['f1_score'] = metrics.f1_score
                data['accuracy'] = metrics.accuracy
                return data
            return {}
        
        result = {}
        for name, metrics in self.accuracy_metrics.items():
            data = asdict(metrics)
            data['precision'] = metrics.precision
            data['recall'] = metrics.recall
            data['f1_score'] = metrics.f1_score
            data['accuracy'] = metrics.accuracy
            result[name] = data
        
        return result
    
    # ========================================================================
    # COMPREHENSIVE REPORTING
    # ========================================================================
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dict with all performance metrics
        """
        uptime_seconds = time.time() - self.start_time
        
        # Calculate overall statistics
        total_checks = sum(
            metrics.total_calls 
            for metrics in self.layer_metrics.values()
            if metrics.layer == ProcessingLayer.TOTAL.value
        )
        
        total_time_ms = sum(
            metrics.total_time_ms
            for metrics in self.layer_metrics.values()
        )
        
        avg_check_time_ms = (
            total_time_ms / total_checks if total_checks > 0 else 0
        )
        
        return {
            'uptime_seconds': round(uptime_seconds, 2),
            'uptime_hours': round(uptime_seconds / 3600, 2),
            'total_checks': total_checks,
            'avg_check_time_ms': round(avg_check_time_ms, 2),
            'total_cost_usd': round(self.get_total_cost(), 4),
            'cache_efficiency': self.get_cache_efficiency(),
            'layer_performance': self.get_layer_metrics(),
            'api_usage': self.get_api_metrics(),
            'accuracy': self.get_accuracy_metrics()
        }
    
    def take_snapshot(self, cache_stats: Optional[Dict] = None,
                     error_stats: Optional[Dict] = None) -> PerformanceSnapshot:
        """
        Take a snapshot of current performance metrics
        
        Args:
            cache_stats: Optional cache statistics
            error_stats: Optional error statistics
            
        Returns:
            PerformanceSnapshot
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            layer_metrics=self.get_layer_metrics(),
            api_metrics=self.get_api_metrics(),
            accuracy_metrics=self.get_accuracy_metrics(),
            cache_stats=cache_stats or {},
            error_stats=error_stats or {},
            summary=self.get_performance_summary()
        )
        
        # Add to history
        self.history.append(snapshot)
        
        # Trim history if needed
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        
        return snapshot
    
    def get_history(self, last_n: Optional[int] = None) -> List[PerformanceSnapshot]:
        """
        Get performance history
        
        Args:
            last_n: Number of recent snapshots to return (None for all)
            
        Returns:
            List of PerformanceSnapshot objects
        """
        if last_n:
            return self.history[-last_n:]
        return self.history
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file
        
        Args:
            filepath: Path to output file
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_performance_summary(),
            'history': [asdict(snapshot) for snapshot in self.history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def print_dashboard(self):
        """Print a formatted performance dashboard to console"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*70)
        print("📊 PERFORMANCE DASHBOARD")
        print("="*70)
        
        # System uptime
        print(f"\n⏱️  System Uptime: {summary['uptime_hours']:.2f} hours")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   Avg Check Time: {summary['avg_check_time_ms']:.2f}ms")
        
        # Layer performance
        print(f"\n🔄 Layer Performance:")
        for layer_name, metrics in summary['layer_performance'].items():
            if metrics['total_calls'] > 0:
                print(f"   {layer_name}:")
                print(f"     Calls: {metrics['total_calls']}")
                print(f"     Avg: {metrics['avg_time_ms']:.2f}ms")
                print(f"     Min/Max: {metrics['min_time_ms']:.2f}ms / {metrics['max_time_ms']:.2f}ms")
                if metrics['errors'] > 0:
                    print(f"     Errors: {metrics['errors']}")
        
        # API usage
        print(f"\n🌐 API Usage:")
        for provider, metrics in summary['api_usage'].items():
            print(f"   {provider}:")
            print(f"     Total Calls: {metrics['total_calls']}")
            print(f"     Success Rate: {metrics['successful_calls']}/{metrics['total_calls']}")
            print(f"     Tokens Used: {metrics['total_tokens']:,}")
            print(f"     Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
            print(f"     Estimated Cost: ${metrics['estimated_cost_usd']:.4f}")
        
        print(f"\n   💰 Total Cost: ${summary['total_cost_usd']:.4f}")
        
        # Cache efficiency
        cache_eff = summary['cache_efficiency']
        print(f"\n💾 Cache Efficiency:")
        print(f"   Hit Rate: {cache_eff['cache_hit_rate']:.2f}%")
        print(f"   Cached Calls: {cache_eff['cached_calls']}/{cache_eff['total_calls']}")
        print(f"   Estimated Savings: ${cache_eff['estimated_savings_usd']:.4f}")
        
        # Accuracy metrics
        if summary['accuracy']:
            print(f"\n🎯 Accuracy Metrics:")
            for check_type, metrics in summary['accuracy'].items():
                if metrics['total_checks'] > 0:
                    print(f"   {check_type}:")
                    print(f"     Total Checks: {metrics['total_checks']}")
                    if metrics['accuracy'] > 0:
                        print(f"     Accuracy: {metrics['accuracy']*100:.1f}%")
                        print(f"     Precision: {metrics['precision']*100:.1f}%")
                        print(f"     Recall: {metrics['recall']*100:.1f}%")
                        print(f"     F1 Score: {metrics['f1_score']:.3f}")
                    if metrics['human_reviews'] > 0:
                        print(f"     Human Reviews: {metrics['human_reviews']}")
        
        print("\n" + "="*70)
    
    def reset(self):
        """Reset all metrics"""
        self.layer_metrics = {
            layer.value: LayerMetrics(layer=layer.value)
            for layer in ProcessingLayer
        }
        self.api_metrics = {}
        self.accuracy_metrics = {}
        self.history = []
        self.start_time = time.time()
        logger.info("Performance metrics reset")


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Performance Monitoring System")
    print("="*70)
    
    # Initialize monitor
    monitor = PerformanceMonitor(cost_config={
        'token_factory': 0.001,
        'gemini': 0.0005
    })
    
    # Simulate some operations
    print("\n🧪 Simulating compliance checks...")
    
    for i in range(10):
        # Rule prefilter
        start = monitor.start_timer()
        time.sleep(0.001)  # Simulate work
        monitor.record_layer_timing(ProcessingLayer.RULE_PREFILTER, start)
        
        # AI analysis
        start = monitor.start_timer()
        time.sleep(0.05)  # Simulate AI call
        monitor.record_layer_timing(ProcessingLayer.AI_ANALYSIS, start)
        monitor.record_api_call(
            provider='token_factory',
            tokens=500,
            latency_ms=50,
            cached=(i % 3 == 0),  # Every 3rd call is cached
            success=True
        )
        
        # Validation
        start = monitor.start_timer()
        time.sleep(0.002)  # Simulate validation
        monitor.record_layer_timing(ProcessingLayer.VALIDATION, start)
        
        # Total
        start_total = monitor.start_timer()
        time.sleep(0.001)
        monitor.record_layer_timing(ProcessingLayer.TOTAL, start_total)
        
        # Record accuracy
        monitor.record_check_result(
            check_type='promotional_mention',
            predicted_violation=(i % 4 == 0),
            actual_violation=(i % 4 == 0),
            human_review=(i % 5 == 0)
        )
    
    # Print dashboard
    monitor.print_dashboard()
    
    # Take snapshot
    snapshot = monitor.take_snapshot()
    print(f"\n📸 Snapshot taken at {snapshot.timestamp}")
    
    # Export metrics
    monitor.export_metrics('performance_metrics.json')
    print(f"\n✓ Metrics exported to performance_metrics.json")
    
    print("\n" + "="*70)
