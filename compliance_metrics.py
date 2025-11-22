#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compliance Metrics System
Tracks false positive rate, false negative rate, precision, recall, and AI performance
for the context-aware compliance checker
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckMetrics:
    """Metrics for a single compliance check"""
    check_type: str
    total_checks: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # AI-specific metrics
    ai_calls: int = 0
    cache_hits: int = 0
    fallback_count: int = 0
    
    # Timing metrics
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    
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
    
    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate (FP / (FP + TN))"""
        denominator = self.false_positives + self.true_negatives
        return self.false_positives / denominator if denominator > 0 else 0.0
    
    @property
    def false_negative_rate(self) -> float:
        """Calculate false negative rate (FN / (FN + TP))"""
        denominator = self.false_negatives + self.true_positives
        return self.false_negatives / denominator if denominator > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_ai = self.ai_calls + self.cache_hits
        return (self.cache_hits / total_ai * 100) if total_ai > 0 else 0.0
    
    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate (when AI unavailable)"""
        return (self.fallback_count / self.total_checks * 100) if self.total_checks > 0 else 0.0


@dataclass
class DocumentMetrics:
    """Metrics for a single document check"""
    document_id: str
    timestamp: str
    total_violations: int
    false_positives: int
    false_negatives: int
    processing_time_ms: float
    ai_calls: int
    cache_hits: int
    fallback_count: int
    violations_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.violations_by_type is None:
            self.violations_by_type = {}


class ComplianceMetrics:
    """
    Comprehensive compliance metrics tracking system
    
    Tracks:
    - False positive rate, false negative rate
    - Precision, recall, F1 score
    - AI API calls, cache hits, fallback rate
    - Processing time per document
    - Metrics by check type
    """
    
    def __init__(self):
        """Initialize compliance metrics tracker"""
        # Metrics by check type
        self.check_metrics: Dict[str, CheckMetrics] = {}
        
        # Document-level metrics
        self.document_metrics: List[DocumentMetrics] = []
        
        # Overall statistics
        self.total_documents = 0
        self.total_violations = 0
        self.total_false_positives = 0
        self.total_false_negatives = 0
        self.total_processing_time_ms = 0.0
        self.total_ai_calls = 0
        self.total_cache_hits = 0
        self.total_fallbacks = 0
        
        # Start time for uptime tracking
        self.start_time = time.time()
        
        logger.info("ComplianceMetrics initialized")
    
    # ========================================================================
    # RECORDING METRICS
    # ========================================================================
    
    def start_document_check(self) -> float:
        """
        Start timing a document check
        
        Returns:
            Start time in seconds
        """
        return time.time()
    
    def record_check_result(self, check_type: str, predicted_violation: bool,
                           actual_violation: Optional[bool] = None,
                           duration_ms: Optional[float] = None,
                           ai_call: bool = False, cached: bool = False,
                           fallback: bool = False):
        """
        Record a compliance check result
        
        Args:
            check_type: Type of compliance check (e.g., "prohibited_phrases", "repeated_securities")
            predicted_violation: Whether system predicted a violation
            actual_violation: Ground truth (if known from validation)
            duration_ms: Time taken for check in milliseconds
            ai_call: Whether an AI call was made
            cached: Whether result was from cache
            fallback: Whether fallback to rules was used
        """
        # Initialize check metrics if needed
        if check_type not in self.check_metrics:
            self.check_metrics[check_type] = CheckMetrics(check_type=check_type)
        
        metrics = self.check_metrics[check_type]
        metrics.total_checks += 1
        
        # Update confusion matrix if ground truth is known
        if actual_violation is not None:
            if predicted_violation and actual_violation:
                metrics.true_positives += 1
            elif predicted_violation and not actual_violation:
                metrics.false_positives += 1
                self.total_false_positives += 1
            elif not predicted_violation and actual_violation:
                metrics.false_negatives += 1
                self.total_false_negatives += 1
            else:  # not predicted and not actual
                metrics.true_negatives += 1
        
        # Update AI metrics
        if ai_call:
            metrics.ai_calls += 1
            self.total_ai_calls += 1
        if cached:
            metrics.cache_hits += 1
            self.total_cache_hits += 1
        if fallback:
            metrics.fallback_count += 1
            self.total_fallbacks += 1
        
        # Update timing
        if duration_ms is not None:
            metrics.total_time_ms += duration_ms
            metrics.avg_time_ms = metrics.total_time_ms / metrics.total_checks
    
    def record_document_result(self, document_id: str, start_time: float,
                              violations: List[Dict], 
                              false_positives: int = 0,
                              false_negatives: int = 0,
                              ai_calls: int = 0,
                              cache_hits: int = 0,
                              fallback_count: int = 0):
        """
        Record complete document check result
        
        Args:
            document_id: Document identifier
            start_time: Start time from start_document_check()
            violations: List of violations found
            false_positives: Number of false positives (if known)
            false_negatives: Number of false negatives (if known)
            ai_calls: Number of AI calls made
            cache_hits: Number of cache hits
            fallback_count: Number of fallback operations
        """
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Count violations by type
        violations_by_type = defaultdict(int)
        for v in violations:
            vtype = v.get('type', 'UNKNOWN')
            violations_by_type[vtype] += 1
        
        # Create document metrics
        doc_metrics = DocumentMetrics(
            document_id=document_id,
            timestamp=datetime.now().isoformat(),
            total_violations=len(violations),
            false_positives=false_positives,
            false_negatives=false_negatives,
            processing_time_ms=processing_time_ms,
            ai_calls=ai_calls,
            cache_hits=cache_hits,
            fallback_count=fallback_count,
            violations_by_type=dict(violations_by_type)
        )
        
        self.document_metrics.append(doc_metrics)
        
        # Update overall statistics
        self.total_documents += 1
        self.total_violations += len(violations)
        self.total_processing_time_ms += processing_time_ms
        
        logger.info(f"Document {document_id}: {len(violations)} violations, "
                   f"{processing_time_ms:.2f}ms, {ai_calls} AI calls")
    
    # ========================================================================
    # QUERYING METRICS
    # ========================================================================
    
    def get_check_metrics(self, check_type: Optional[str] = None) -> Dict:
        """
        Get metrics for specific check type or all checks
        
        Args:
            check_type: Specific check type or None for all
            
        Returns:
            Dict of metrics
        """
        if check_type:
            metrics = self.check_metrics.get(check_type)
            if metrics:
                data = asdict(metrics)
                data['precision'] = metrics.precision
                data['recall'] = metrics.recall
                data['f1_score'] = metrics.f1_score
                data['accuracy'] = metrics.accuracy
                data['false_positive_rate'] = metrics.false_positive_rate
                data['false_negative_rate'] = metrics.false_negative_rate
                data['cache_hit_rate'] = metrics.cache_hit_rate
                data['fallback_rate'] = metrics.fallback_rate
                return data
            return {}
        
        # Return all check metrics
        result = {}
        for name, metrics in self.check_metrics.items():
            data = asdict(metrics)
            data['precision'] = metrics.precision
            data['recall'] = metrics.recall
            data['f1_score'] = metrics.f1_score
            data['accuracy'] = metrics.accuracy
            data['false_positive_rate'] = metrics.false_positive_rate
            data['false_negative_rate'] = metrics.false_negative_rate
            data['cache_hit_rate'] = metrics.cache_hit_rate
            data['fallback_rate'] = metrics.fallback_rate
            result[name] = data
        
        return result
    
    def get_document_metrics(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get document-level metrics
        
        Args:
            last_n: Number of recent documents to return (None for all)
            
        Returns:
            List of document metrics
        """
        docs = self.document_metrics[-last_n:] if last_n else self.document_metrics
        return [asdict(doc) for doc in docs]
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Get overall system metrics
        
        Returns:
            Dict with comprehensive metrics
        """
        uptime_seconds = time.time() - self.start_time
        
        # Calculate averages
        avg_processing_time_ms = (
            self.total_processing_time_ms / self.total_documents 
            if self.total_documents > 0 else 0
        )
        
        avg_violations_per_doc = (
            self.total_violations / self.total_documents
            if self.total_documents > 0 else 0
        )
        
        # Calculate overall rates
        total_ai_operations = self.total_ai_calls + self.total_cache_hits
        cache_hit_rate = (
            (self.total_cache_hits / total_ai_operations * 100)
            if total_ai_operations > 0 else 0
        )
        
        fallback_rate = (
            (self.total_fallbacks / self.total_documents * 100)
            if self.total_documents > 0 else 0
        )
        
        # Calculate overall accuracy metrics
        total_tp = sum(m.true_positives for m in self.check_metrics.values())
        total_fp = sum(m.false_positives for m in self.check_metrics.values())
        total_tn = sum(m.true_negatives for m in self.check_metrics.values())
        total_fn = sum(m.false_negatives for m in self.check_metrics.values())
        
        total_predictions = total_tp + total_fp + total_tn + total_fn
        
        overall_precision = (
            total_tp / (total_tp + total_fp) 
            if (total_tp + total_fp) > 0 else 0
        )
        
        overall_recall = (
            total_tp / (total_tp + total_fn)
            if (total_tp + total_fn) > 0 else 0
        )
        
        overall_accuracy = (
            (total_tp + total_tn) / total_predictions
            if total_predictions > 0 else 0
        )
        
        overall_fpr = (
            total_fp / (total_fp + total_tn)
            if (total_fp + total_tn) > 0 else 0
        )
        
        overall_fnr = (
            total_fn / (total_fn + total_tp)
            if (total_fn + total_tp) > 0 else 0
        )
        
        return {
            'uptime_seconds': round(uptime_seconds, 2),
            'uptime_hours': round(uptime_seconds / 3600, 2),
            'total_documents': self.total_documents,
            'total_violations': self.total_violations,
            'avg_violations_per_doc': round(avg_violations_per_doc, 2),
            'avg_processing_time_ms': round(avg_processing_time_ms, 2),
            'total_false_positives': self.total_false_positives,
            'total_false_negatives': self.total_false_negatives,
            'accuracy_metrics': {
                'precision': round(overall_precision, 4),
                'recall': round(overall_recall, 4),
                'accuracy': round(overall_accuracy, 4),
                'false_positive_rate': round(overall_fpr, 4),
                'false_negative_rate': round(overall_fnr, 4),
                'f1_score': round(
                    2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
                    if (overall_precision + overall_recall) > 0 else 0,
                    4
                )
            },
            'ai_performance': {
                'total_ai_calls': self.total_ai_calls,
                'total_cache_hits': self.total_cache_hits,
                'cache_hit_rate': round(cache_hit_rate, 2),
                'total_fallbacks': self.total_fallbacks,
                'fallback_rate': round(fallback_rate, 2)
            }
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Returns:
            Dict with all metrics
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'overall': self.get_overall_metrics(),
            'by_check_type': self.get_check_metrics(),
            'recent_documents': self.get_document_metrics(last_n=10)
        }
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_dashboard(self):
        """Print formatted metrics dashboard to console"""
        overall = self.get_overall_metrics()
        
        print("\n" + "="*70)
        print("📊 COMPLIANCE METRICS DASHBOARD")
        print("="*70)
        
        # System overview
        print(f"\n⏱️  System Uptime: {overall['uptime_hours']:.2f} hours")
        print(f"   Documents Processed: {overall['total_documents']}")
        print(f"   Total Violations: {overall['total_violations']}")
        print(f"   Avg Violations/Doc: {overall['avg_violations_per_doc']:.2f}")
        print(f"   Avg Processing Time: {overall['avg_processing_time_ms']:.2f}ms")
        
        # Accuracy metrics
        acc = overall['accuracy_metrics']
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   Precision: {acc['precision']*100:.2f}%")
        print(f"   Recall: {acc['recall']*100:.2f}%")
        print(f"   F1 Score: {acc['f1_score']:.3f}")
        print(f"   Overall Accuracy: {acc['accuracy']*100:.2f}%")
        print(f"   False Positive Rate: {acc['false_positive_rate']*100:.2f}%")
        print(f"   False Negative Rate: {acc['false_negative_rate']*100:.2f}%")
        
        # AI performance
        ai = overall['ai_performance']
        print(f"\n🤖 AI Performance:")
        print(f"   Total AI Calls: {ai['total_ai_calls']}")
        print(f"   Cache Hits: {ai['total_cache_hits']}")
        print(f"   Cache Hit Rate: {ai['cache_hit_rate']:.2f}%")
        print(f"   Fallback Count: {ai['total_fallbacks']}")
        print(f"   Fallback Rate: {ai['fallback_rate']:.2f}%")
        
        # Metrics by check type
        check_metrics = self.get_check_metrics()
        if check_metrics:
            print(f"\n📋 Metrics by Check Type:")
            for check_type, metrics in check_metrics.items():
                if metrics['total_checks'] > 0:
                    print(f"\n   {check_type}:")
                    print(f"     Total Checks: {metrics['total_checks']}")
                    print(f"     Precision: {metrics['precision']*100:.2f}%")
                    print(f"     Recall: {metrics['recall']*100:.2f}%")
                    print(f"     FP Rate: {metrics['false_positive_rate']*100:.2f}%")
                    print(f"     FN Rate: {metrics['false_negative_rate']*100:.2f}%")
                    print(f"     Avg Time: {metrics['avg_time_ms']:.2f}ms")
                    if metrics['ai_calls'] > 0:
                        print(f"     AI Calls: {metrics['ai_calls']}")
                        print(f"     Cache Hit Rate: {metrics['cache_hit_rate']:.2f}%")
                    if metrics['fallback_count'] > 0:
                        print(f"     Fallback Rate: {metrics['fallback_rate']:.2f}%")
        
        print("\n" + "="*70)
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file
        
        Args:
            filepath: Path to output file
        """
        data = self.get_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self):
        """Reset all metrics"""
        self.check_metrics = {}
        self.document_metrics = []
        self.total_documents = 0
        self.total_violations = 0
        self.total_false_positives = 0
        self.total_false_negatives = 0
        self.total_processing_time_ms = 0.0
        self.total_ai_calls = 0
        self.total_cache_hits = 0
        self.total_fallbacks = 0
        self.start_time = time.time()
        logger.info("Compliance metrics reset")


# Singleton instance
_metrics_instance = None


def get_compliance_metrics() -> ComplianceMetrics:
    """
    Get singleton ComplianceMetrics instance
    
    Returns:
        ComplianceMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ComplianceMetrics()
    return _metrics_instance


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Compliance Metrics System")
    print("="*70)
    
    # Initialize metrics
    metrics = ComplianceMetrics()
    
    # Simulate document checks
    print("\n🧪 Simulating compliance checks...")
    
    for doc_num in range(5):
        start = metrics.start_document_check()
        violations = []
        ai_calls = 0
        cache_hits = 0
        fallbacks = 0
        
        # Simulate various checks
        check_types = [
            'prohibited_phrases',
            'repeated_securities',
            'performance_disclaimers',
            'structure_validation'
        ]
        
        for check_type in check_types:
            # Simulate check with varying results
            predicted = (doc_num + len(check_type)) % 3 == 0
            actual = predicted  # Assume correct for demo
            
            # Simulate AI usage
            use_ai = check_type in ['prohibited_phrases', 'repeated_securities']
            cached = use_ai and (doc_num % 2 == 0)
            fallback = use_ai and (doc_num % 5 == 0)
            
            if use_ai:
                if cached:
                    cache_hits += 1
                else:
                    ai_calls += 1
            
            if fallback:
                fallbacks += 1
            
            # Record check result
            metrics.record_check_result(
                check_type=check_type,
                predicted_violation=predicted,
                actual_violation=actual,
                duration_ms=50.0 if use_ai else 5.0,
                ai_call=use_ai and not cached,
                cached=cached,
                fallback=fallback
            )
            
            if predicted:
                violations.append({
                    'type': check_type,
                    'message': f'Test violation in {check_type}'
                })
        
        # Record document result
        metrics.record_document_result(
            document_id=f'doc_{doc_num}.json',
            start_time=start,
            violations=violations,
            false_positives=0,
            false_negatives=0,
            ai_calls=ai_calls,
            cache_hits=cache_hits,
            fallback_count=fallbacks
        )
    
    # Print dashboard
    metrics.print_dashboard()
    
    # Export metrics
    metrics.export_metrics('compliance_metrics_test.json')
    print(f"\n✓ Metrics exported to compliance_metrics_test.json")
    
    print("\n" + "="*70)
