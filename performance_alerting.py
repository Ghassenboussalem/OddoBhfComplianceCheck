#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Alerting System
Monitors performance metrics and triggers alerts when thresholds are exceeded
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of performance alerts"""
    HIGH_LATENCY = "high_latency"
    HIGH_COST = "high_cost"
    LOW_CACHE_HIT_RATE = "low_cache_hit_rate"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_ACCURACY = "low_accuracy"
    HIGH_API_USAGE = "high_api_usage"


@dataclass
class Alert:
    """Performance alert"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: str
    metric_value: float
    threshold: float
    details: Dict


@dataclass
class AlertThresholds:
    """Configurable thresholds for alerts"""
    # Latency thresholds (milliseconds)
    max_avg_latency_ms: float = 2000.0
    max_total_latency_ms: float = 5000.0
    
    # Cost thresholds (USD)
    max_cost_per_check_usd: float = 0.01
    max_total_cost_usd: float = 10.0
    
    # Cache thresholds (percentage)
    min_cache_hit_rate: float = 30.0
    
    # Error rate thresholds (percentage)
    max_error_rate: float = 10.0
    
    # Accuracy thresholds (percentage)
    min_accuracy: float = 80.0
    min_precision: float = 75.0
    min_recall: float = 75.0
    
    # API usage thresholds
    max_tokens_per_hour: int = 100000
    max_calls_per_minute: int = 60


class PerformanceAlerting:
    """
    Performance alerting system
    
    Monitors performance metrics and triggers alerts when thresholds are exceeded
    """
    
    def __init__(self, thresholds: Optional[AlertThresholds] = None,
                 alert_callback: Optional[Callable[[Alert], None]] = None):
        """
        Initialize alerting system
        
        Args:
            thresholds: Custom alert thresholds
            alert_callback: Optional callback function for alerts
        """
        self.thresholds = thresholds or AlertThresholds()
        self.alert_callback = alert_callback
        self.alerts: List[Alert] = []
        self.max_alerts_history = 1000
        
        logger.info("PerformanceAlerting initialized")
    
    def check_metrics(self, performance_summary: Dict) -> List[Alert]:
        """
        Check performance metrics against thresholds
        
        Args:
            performance_summary: Performance summary from PerformanceMonitor
            
        Returns:
            List of triggered alerts
        """
        new_alerts = []
        
        # Check latency
        new_alerts.extend(self._check_latency(performance_summary))
        
        # Check costs
        new_alerts.extend(self._check_costs(performance_summary))
        
        # Check cache efficiency
        new_alerts.extend(self._check_cache(performance_summary))
        
        # Check error rates
        new_alerts.extend(self._check_errors(performance_summary))
        
        # Check accuracy
        new_alerts.extend(self._check_accuracy(performance_summary))
        
        # Store alerts
        for alert in new_alerts:
            self._record_alert(alert)
        
        return new_alerts
    
    def _check_latency(self, summary: Dict) -> List[Alert]:
        """Check latency metrics"""
        alerts = []
        
        # Check average check time
        avg_time = summary.get('avg_check_time_ms', 0)
        if avg_time > self.thresholds.max_avg_latency_ms:
            alerts.append(Alert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                message=f"Average check time ({avg_time:.2f}ms) exceeds threshold ({self.thresholds.max_avg_latency_ms}ms)",
                timestamp=datetime.now().isoformat(),
                metric_value=avg_time,
                threshold=self.thresholds.max_avg_latency_ms,
                details={'metric': 'avg_check_time_ms'}
            ))
        
        # Check layer performance
        layer_perf = summary.get('layer_performance', {})
        for layer_name, metrics in layer_perf.items():
            avg_layer_time = metrics.get('avg_time_ms', 0)
            if layer_name == 'total' and avg_layer_time > self.thresholds.max_total_latency_ms:
                alerts.append(Alert(
                    alert_type=AlertType.HIGH_LATENCY,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Total processing time ({avg_layer_time:.2f}ms) exceeds critical threshold ({self.thresholds.max_total_latency_ms}ms)",
                    timestamp=datetime.now().isoformat(),
                    metric_value=avg_layer_time,
                    threshold=self.thresholds.max_total_latency_ms,
                    details={'metric': 'total_time_ms', 'layer': layer_name}
                ))
        
        return alerts
    
    def _check_costs(self, summary: Dict) -> List[Alert]:
        """Check cost metrics"""
        alerts = []
        
        # Check total cost
        total_cost = summary.get('total_cost_usd', 0)
        if total_cost > self.thresholds.max_total_cost_usd:
            alerts.append(Alert(
                alert_type=AlertType.HIGH_COST,
                severity=AlertSeverity.CRITICAL,
                message=f"Total cost (${total_cost:.4f}) exceeds budget (${self.thresholds.max_total_cost_usd:.4f})",
                timestamp=datetime.now().isoformat(),
                metric_value=total_cost,
                threshold=self.thresholds.max_total_cost_usd,
                details={'metric': 'total_cost_usd'}
            ))
        
        # Check cost per check
        total_checks = summary.get('total_checks', 0)
        if total_checks > 0:
            cost_per_check = total_cost / total_checks
            if cost_per_check > self.thresholds.max_cost_per_check_usd:
                alerts.append(Alert(
                    alert_type=AlertType.HIGH_COST,
                    severity=AlertSeverity.WARNING,
                    message=f"Cost per check (${cost_per_check:.4f}) exceeds threshold (${self.thresholds.max_cost_per_check_usd:.4f})",
                    timestamp=datetime.now().isoformat(),
                    metric_value=cost_per_check,
                    threshold=self.thresholds.max_cost_per_check_usd,
                    details={'metric': 'cost_per_check', 'total_checks': total_checks}
                ))
        
        return alerts
    
    def _check_cache(self, summary: Dict) -> List[Alert]:
        """Check cache efficiency"""
        alerts = []
        
        cache_eff = summary.get('cache_efficiency', {})
        hit_rate = cache_eff.get('cache_hit_rate', 0)
        
        if hit_rate < self.thresholds.min_cache_hit_rate:
            alerts.append(Alert(
                alert_type=AlertType.LOW_CACHE_HIT_RATE,
                severity=AlertSeverity.WARNING,
                message=f"Cache hit rate ({hit_rate:.1f}%) below threshold ({self.thresholds.min_cache_hit_rate:.1f}%)",
                timestamp=datetime.now().isoformat(),
                metric_value=hit_rate,
                threshold=self.thresholds.min_cache_hit_rate,
                details={
                    'metric': 'cache_hit_rate',
                    'total_calls': cache_eff.get('total_calls', 0),
                    'cached_calls': cache_eff.get('cached_calls', 0)
                }
            ))
        
        return alerts
    
    def _check_errors(self, summary: Dict) -> List[Alert]:
        """Check error rates"""
        alerts = []
        
        # Check layer errors
        layer_perf = summary.get('layer_performance', {})
        for layer_name, metrics in layer_perf.items():
            total_calls = metrics.get('total_calls', 0)
            errors = metrics.get('errors', 0)
            
            if total_calls > 0:
                error_rate = (errors / total_calls) * 100
                if error_rate > self.thresholds.max_error_rate:
                    alerts.append(Alert(
                        alert_type=AlertType.HIGH_ERROR_RATE,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Error rate in {layer_name} ({error_rate:.1f}%) exceeds threshold ({self.thresholds.max_error_rate:.1f}%)",
                        timestamp=datetime.now().isoformat(),
                        metric_value=error_rate,
                        threshold=self.thresholds.max_error_rate,
                        details={
                            'metric': 'error_rate',
                            'layer': layer_name,
                            'errors': errors,
                            'total_calls': total_calls
                        }
                    ))
        
        return alerts
    
    def _check_accuracy(self, summary: Dict) -> List[Alert]:
        """Check accuracy metrics"""
        alerts = []
        
        accuracy_metrics = summary.get('accuracy', {})
        
        for check_type, metrics in accuracy_metrics.items():
            total_checks = metrics.get('total_checks', 0)
            
            # Only check if we have enough data
            if total_checks < 10:
                continue
            
            # Check accuracy
            accuracy = metrics.get('accuracy', 0) * 100
            if accuracy < self.thresholds.min_accuracy:
                alerts.append(Alert(
                    alert_type=AlertType.LOW_ACCURACY,
                    severity=AlertSeverity.WARNING,
                    message=f"Accuracy for {check_type} ({accuracy:.1f}%) below threshold ({self.thresholds.min_accuracy:.1f}%)",
                    timestamp=datetime.now().isoformat(),
                    metric_value=accuracy,
                    threshold=self.thresholds.min_accuracy,
                    details={
                        'metric': 'accuracy',
                        'check_type': check_type,
                        'total_checks': total_checks
                    }
                ))
            
            # Check precision
            precision = metrics.get('precision', 0) * 100
            if precision < self.thresholds.min_precision:
                alerts.append(Alert(
                    alert_type=AlertType.LOW_ACCURACY,
                    severity=AlertSeverity.WARNING,
                    message=f"Precision for {check_type} ({precision:.1f}%) below threshold ({self.thresholds.min_precision:.1f}%)",
                    timestamp=datetime.now().isoformat(),
                    metric_value=precision,
                    threshold=self.thresholds.min_precision,
                    details={
                        'metric': 'precision',
                        'check_type': check_type,
                        'total_checks': total_checks
                    }
                ))
            
            # Check recall
            recall = metrics.get('recall', 0) * 100
            if recall < self.thresholds.min_recall:
                alerts.append(Alert(
                    alert_type=AlertType.LOW_ACCURACY,
                    severity=AlertSeverity.WARNING,
                    message=f"Recall for {check_type} ({recall:.1f}%) below threshold ({self.thresholds.min_recall:.1f}%)",
                    timestamp=datetime.now().isoformat(),
                    metric_value=recall,
                    threshold=self.thresholds.min_recall,
                    details={
                        'metric': 'recall',
                        'check_type': check_type,
                        'total_checks': total_checks
                    }
                ))
        
        return alerts
    
    def _record_alert(self, alert: Alert):
        """Record an alert"""
        self.alerts.append(alert)
        
        # Trim history if needed
        if len(self.alerts) > self.max_alerts_history:
            self.alerts = self.alerts[-self.max_alerts_history:]
        
        # Log alert
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.info)
        
        log_func(f"[{alert.severity.value}] {alert.alert_type.value}: {alert.message}")
        
        # Call callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None,
                   alert_type: Optional[AlertType] = None,
                   last_n: Optional[int] = None) -> List[Alert]:
        """
        Get alerts with optional filtering
        
        Args:
            severity: Filter by severity
            alert_type: Filter by alert type
            last_n: Get last N alerts
            
        Returns:
            List of alerts
        """
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if last_n:
            alerts = alerts[-last_n:]
        
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        total = len(self.alerts)
        
        by_severity = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 0,
            AlertSeverity.CRITICAL: 0
        }
        
        by_type = {}
        
        for alert in self.alerts:
            by_severity[alert.severity] += 1
            by_type[alert.alert_type.value] = by_type.get(alert.alert_type.value, 0) + 1
        
        return {
            'total_alerts': total,
            'by_severity': {k.value: v for k, v in by_severity.items()},
            'by_type': by_type,
            'recent_alerts': [
                {
                    'type': a.alert_type.value,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp
                }
                for a in self.alerts[-5:]
            ]
        }
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        logger.info("Alerts cleared")
    
    def update_thresholds(self, **kwargs):
        """Update alert thresholds"""
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(f"Updated threshold: {key} = {value}")


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("Performance Alerting System")
    print("="*70)
    
    # Initialize alerting
    def alert_handler(alert: Alert):
        """Custom alert handler"""
        print(f"\n🚨 ALERT: [{alert.severity.value}] {alert.message}")
    
    alerting = PerformanceAlerting(
        thresholds=AlertThresholds(
            max_avg_latency_ms=1000,
            max_total_cost_usd=5.0,
            min_cache_hit_rate=40.0
        ),
        alert_callback=alert_handler
    )
    
    # Simulate performance summary with issues
    test_summary = {
        'total_checks': 100,
        'avg_check_time_ms': 1500,  # High latency
        'total_cost_usd': 6.0,       # High cost
        'cache_efficiency': {
            'cache_hit_rate': 25.0,  # Low cache hit rate
            'total_calls': 100,
            'cached_calls': 25
        },
        'layer_performance': {
            'total': {
                'total_calls': 100,
                'avg_time_ms': 1500,
                'errors': 15  # High error rate
            }
        },
        'accuracy': {
            'promotional_mention': {
                'total_checks': 50,
                'accuracy': 0.70,  # Low accuracy
                'precision': 0.65,
                'recall': 0.68
            }
        }
    }
    
    print("\n🔍 Checking metrics against thresholds...")
    alerts = alerting.check_metrics(test_summary)
    
    print(f"\n📊 Alert Summary:")
    summary = alerting.get_alert_summary()
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By type: {summary['by_type']}")
    
    print("\n" + "="*70)
