#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Compliance Checker - Core Architecture
Three-layer hybrid approach: Rules → AI → Validation
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import error handling components
try:
    from error_handler import ErrorHandler, ServiceHealthMonitor, GracefulDegradation
except ImportError:
    ErrorHandler = None
    ServiceHealthMonitor = None
    GracefulDegradation = None

# Import performance monitoring
try:
    from performance_monitor import PerformanceMonitor, ProcessingLayer
except ImportError:
    PerformanceMonitor = None
    ProcessingLayer = None

# Import review manager for HITL
try:
    from review_manager import ReviewManager, ReviewItem, ReviewStatus
except ImportError:
    ReviewManager = None
    ReviewItem = None
    ReviewStatus = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckType(Enum):
    """Enumeration of all compliance check types"""
    STRUCTURE = "structure"
    PERFORMANCE = "performance"
    PROSPECTUS = "prospectus"
    REGISTRATION = "registration"
    GENERAL = "general"
    VALUES = "values"
    ESG = "esg"
    DISCLAIMERS = "disclaimers"


class ComplianceStatus(Enum):
    """Status categories for compliance results"""
    VERIFIED_BY_BOTH = "VERIFIED_BY_BOTH"
    AI_DETECTED_VARIATION = "AI_DETECTED_VARIATION"
    FALSE_POSITIVE_FILTERED = "FALSE_POSITIVE_FILTERED"
    VIOLATION_CONFIRMED = "VIOLATION_CONFIRMED"
    RULE_ONLY = "RULE_ONLY"
    AI_ONLY = "AI_ONLY"


@dataclass
class ComplianceResult:
    """Result of a compliance check"""
    violation: bool
    confidence: int  # 0-100
    status: ComplianceStatus
    evidence: str
    reasoning: str
    check_type: CheckType
    slide: str
    location: str
    rule: str
    message: str
    severity: str = "CRITICAL"
    ai_reasoning: Optional[str] = None
    rule_hints: Optional[str] = None


@dataclass
class HybridConfig:
    """Configuration for hybrid compliance checking"""
    ai_enabled: bool = True
    rule_enabled: bool = True
    confidence_threshold: int = 70
    ai_timeout: int = 30
    cache_enabled: bool = True
    fallback_to_rules: bool = True
    batch_size: int = 5


class HybridComplianceChecker:
    """
    Main orchestrator for three-layer hybrid compliance checking
    
    Architecture:
    Layer 1: Rule-based pre-filtering (fast screening)
    Layer 2: AI analysis (deep understanding) 
    Layer 3: Rule-based validation (confidence scoring)
    """
    
    def __init__(self, ai_engine=None, confidence_scorer=None, config: HybridConfig = None,
                 error_handler=None, performance_monitor=None, feedback_interface=None, review_manager=None):
        """
        Initialize the hybrid compliance checker
        
        Args:
            ai_engine: AIEngine instance for semantic analysis
            confidence_scorer: ConfidenceScorer instance for result combination
            config: HybridConfig for system settings
            error_handler: ErrorHandler instance for error handling and fallback
            performance_monitor: PerformanceMonitor instance for metrics tracking
            feedback_interface: FeedbackInterface instance for human corrections
            review_manager: ReviewManager instance for HITL queue management
        """
        self.ai_engine = ai_engine
        self.confidence_scorer = confidence_scorer
        self.config = config or HybridConfig()
        
        # Initialize error handling
        if error_handler:
            self.error_handler = error_handler
        elif ErrorHandler:
            self.error_handler = ErrorHandler()
        else:
            self.error_handler = None
        
        # Initialize graceful degradation
        if self.error_handler and GracefulDegradation:
            self.degradation = GracefulDegradation(self.error_handler)
        else:
            self.degradation = None
        
        # Initialize performance monitoring
        if performance_monitor:
            self.performance_monitor = performance_monitor
        elif PerformanceMonitor:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None
        
        # Initialize feedback interface
        self.feedback_interface = feedback_interface
        
        # Initialize review manager for HITL
        self.review_manager = review_manager
        
        # Initialize rule engines for each check type
        self._rule_engines = {}
        
        logger.info("HybridComplianceChecker initialized")
        logger.info(f"AI enabled: {self.config.ai_enabled}")
        logger.info(f"Confidence threshold: {self.config.confidence_threshold}")
        logger.info(f"Error handling: {'enabled' if self.error_handler else 'disabled'}")
        logger.info(f"Performance monitoring: {'enabled' if self.performance_monitor else 'disabled'}")
        logger.info(f"Feedback loop: {'enabled' if self.feedback_interface else 'disabled'}")
        logger.info(f"Review manager: {'enabled' if self.review_manager else 'disabled'}")
    
    def check_compliance(self, document: Dict, check_type: CheckType, **kwargs) -> Optional[ComplianceResult]:
        """
        Main compliance checking method using three-layer architecture
        
        Args:
            document: Document data to check
            check_type: Type of compliance check to perform
            **kwargs: Additional parameters for specific checks
            
        Returns:
            ComplianceResult if violation found, None if compliant
        """
        logger.info(f"Starting {check_type.value} compliance check")
        
        # Start total timing
        total_start = self.performance_monitor.start_timer() if self.performance_monitor else None
        
        # Check service health and adjust degradation level
        if self.degradation:
            self.degradation.check_and_adjust()
        
        try:
            # Layer 1: Rule-based pre-filtering
            rule_start = self.performance_monitor.start_timer() if self.performance_monitor else None
            rule_result = self._rule_pre_filter(document, check_type, **kwargs)
            if self.performance_monitor and rule_start and ProcessingLayer:
                self.performance_monitor.record_layer_timing(ProcessingLayer.RULE_PREFILTER, rule_start)
            
            # Layer 2: AI analysis (if enabled, available, and service is healthy)
            ai_result = None
            should_use_ai = (
                self.config.ai_enabled and 
                self.ai_engine and 
                (not self.degradation or self.degradation.should_use_ai())
            )
            
            if should_use_ai:
                ai_start = self.performance_monitor.start_timer() if self.performance_monitor else None
                ai_result = self._ai_analysis_with_error_handling(
                    document, check_type, rule_result, **kwargs
                )
                if self.performance_monitor and ai_start and ProcessingLayer:
                    self.performance_monitor.record_layer_timing(
                        ProcessingLayer.AI_ANALYSIS, ai_start, error=(ai_result is None)
                    )
            
            # Layer 3: Confidence scoring and validation
            validation_start = self.performance_monitor.start_timer() if self.performance_monitor else None
            final_result = self._validate_and_score(rule_result, ai_result, check_type)
            if self.performance_monitor and validation_start and ProcessingLayer:
                self.performance_monitor.record_layer_timing(ProcessingLayer.VALIDATION, validation_start)
            
            # Record total timing
            if self.performance_monitor and total_start and ProcessingLayer:
                self.performance_monitor.record_layer_timing(ProcessingLayer.TOTAL, total_start)
            
            # Record accuracy metrics
            if self.performance_monitor:
                self.performance_monitor.record_check_result(
                    check_type=check_type.value,
                    predicted_violation=(final_result is not None),
                    human_review=(final_result and final_result.confidence < 70) if final_result else False
                )
            
            # Queue for human review if confidence is below threshold
            if final_result and final_result.confidence < self.config.confidence_threshold:
                # Add to review queue if ReviewManager is available
                if self.review_manager and ReviewItem and ReviewStatus:
                    document_id = document.get('document_id', kwargs.get('document_id', 'unknown'))
                    
                    # Calculate priority score
                    priority_score = self.review_manager.calculate_priority_score(
                        confidence=final_result.confidence,
                        severity=final_result.severity,
                        age_hours=0.0
                    )
                    
                    # Create review item
                    review_item = ReviewItem(
                        review_id=str(uuid.uuid4()),
                        document_id=document_id,
                        check_type=check_type.value,
                        slide=final_result.slide,
                        location=final_result.location,
                        predicted_violation=final_result.violation,
                        confidence=final_result.confidence,
                        ai_reasoning=final_result.ai_reasoning or final_result.reasoning,
                        evidence=final_result.evidence,
                        rule=final_result.rule,
                        severity=final_result.severity,
                        created_at=datetime.now().isoformat(),
                        priority_score=priority_score,
                        status=ReviewStatus.PENDING
                    )
                    
                    # Add to queue
                    self.review_manager.add_to_queue(review_item)
                    logger.info(f"Queued for human review: {check_type.value} (confidence: {final_result.confidence}%)")
                
                # Also submit to feedback interface if available (backward compatibility)
                elif self.feedback_interface:
                    document_id = document.get('document_id', kwargs.get('document_id', 'unknown'))
                    self.feedback_interface.submit_for_review(
                        check_type=check_type.value,
                        document_id=document_id,
                        slide=final_result.slide,
                        predicted_violation=final_result.violation,
                        predicted_confidence=final_result.confidence,
                        predicted_reasoning=final_result.reasoning,
                        predicted_evidence=final_result.evidence,
                        processing_time_ms=self.performance_monitor.get_last_timing() if self.performance_monitor else None,
                        ai_provider=ai_result.get('provider') if ai_result else None
                    )
                    logger.info(f"Submitted for human review: {check_type.value} (confidence: {final_result.confidence}%)")
            
            if final_result and final_result.confidence >= self.config.confidence_threshold:
                logger.info(f"Violation detected: {final_result.message} (confidence: {final_result.confidence}%)")
                return final_result
            
            logger.info(f"{check_type.value} check passed")
            return None
            
        except Exception as e:
            logger.error(f"Error in {check_type.value} check: {e}")
            
            # Record error in health monitor
            if self.error_handler:
                self.error_handler.health_monitor.record_error("check")
            
            # Record error in performance monitor
            if self.performance_monitor and total_start and ProcessingLayer:
                self.performance_monitor.record_layer_timing(ProcessingLayer.TOTAL, total_start, error=True)
            
            # Fallback to rule-only if configured
            if self.config.fallback_to_rules:
                return self._fallback_rule_check(document, check_type, **kwargs)
            
            raise
    
    def check_all_compliance(self, document: Dict, **kwargs) -> List[ComplianceResult]:
        """
        Run all compliance checks on a document
        
        Args:
            document: Document data to check
            **kwargs: Additional parameters
            
        Returns:
            List of ComplianceResult objects for violations found
        """
        violations = []
        
        for check_type in CheckType:
            try:
                result = self.check_compliance(document, check_type, **kwargs)
                if result:
                    violations.append(result)
            except Exception as e:
                logger.error(f"Failed to run {check_type.value} check: {e}")
                continue
        
        logger.info(f"Compliance check complete: {len(violations)} violations found")
        return violations
    
    # ========================================================================
    # INDIVIDUAL CHECK TYPE METHODS
    # ========================================================================
    
    def check_structure_compliance(self, document: Dict, client_type: str = "retail", **kwargs) -> List[ComplianceResult]:
        """Check structure compliance (promotional mention, target audience, etc.)"""
        violations = []
        
        # Promotional document mention
        result = self.check_compliance(document, CheckType.STRUCTURE, 
                                     subcheck="promotional_mention", **kwargs)
        if result:
            violations.append(result)
        
        # Target audience specification
        result = self.check_compliance(document, CheckType.STRUCTURE,
                                     subcheck="target_audience", client_type=client_type, **kwargs)
        if result:
            violations.append(result)
        
        # Management company mention
        result = self.check_compliance(document, CheckType.STRUCTURE,
                                     subcheck="management_company", **kwargs)
        if result:
            violations.append(result)
        
        # Slide 2 disclaimers
        result = self.check_compliance(document, CheckType.STRUCTURE,
                                     subcheck="slide2_disclaimers", **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    def check_performance_compliance(self, document: Dict, client_type: str = "retail", 
                                   fund_age_years: Optional[int] = None, **kwargs) -> List[ComplianceResult]:
        """Check performance compliance (disclaimers, benchmark comparison, etc.)"""
        violations = []
        
        # Performance disclaimers
        result = self.check_compliance(document, CheckType.PERFORMANCE,
                                     subcheck="performance_disclaimers", 
                                     client_type=client_type, **kwargs)
        if result:
            violations.append(result)
        
        # Benchmark comparison
        result = self.check_compliance(document, CheckType.PERFORMANCE,
                                     subcheck="benchmark_comparison", **kwargs)
        if result:
            violations.append(result)
        
        # Fund age restrictions
        if fund_age_years is not None:
            result = self.check_compliance(document, CheckType.PERFORMANCE,
                                         subcheck="fund_age_restrictions",
                                         fund_age_years=fund_age_years, **kwargs)
            if result:
                violations.append(result)
        
        return violations
    
    def check_prospectus_compliance(self, document: Dict, prospectus_data: Dict, **kwargs) -> List[ComplianceResult]:
        """Check prospectus compliance (fund name matching, etc.)"""
        violations = []
        
        # Fund name semantic matching
        result = self.check_compliance(document, CheckType.PROSPECTUS,
                                     subcheck="fund_name_match",
                                     prospectus_data=prospectus_data, **kwargs)
        if result:
            violations.append(result)
        
        # Investment strategy consistency
        result = self.check_compliance(document, CheckType.PROSPECTUS,
                                     subcheck="strategy_consistency",
                                     prospectus_data=prospectus_data, **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    def check_registration_compliance(self, document: Dict, fund_isin: str, 
                                    authorized_countries: List[str], **kwargs) -> List[ComplianceResult]:
        """Check registration compliance (country authorization, etc.)"""
        violations = []
        
        # Country authorization validation
        result = self.check_compliance(document, CheckType.REGISTRATION,
                                     subcheck="country_authorization",
                                     fund_isin=fund_isin,
                                     authorized_countries=authorized_countries, **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    def check_general_compliance(self, document: Dict, client_type: str = "retail", 
                               country_code: Optional[str] = None, **kwargs) -> List[ComplianceResult]:
        """Check general compliance (glossary, technical terms, etc.)"""
        violations = []
        
        # Glossary requirement for retail clients
        if client_type.lower() == "retail":
            result = self.check_compliance(document, CheckType.GENERAL,
                                         subcheck="glossary_requirement",
                                         client_type=client_type, **kwargs)
            if result:
                violations.append(result)
        
        # Morningstar date validation
        result = self.check_compliance(document, CheckType.GENERAL,
                                     subcheck="morningstar_date", **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    def check_values_compliance(self, document: Dict, **kwargs) -> List[ComplianceResult]:
        """Check values/securities compliance (company mentions, disclaimers, etc.)"""
        violations = []
        
        # Company mention validation
        result = self.check_compliance(document, CheckType.VALUES,
                                     subcheck="company_mentions", **kwargs)
        if result:
            violations.append(result)
        
        # Securities disclaimer validation
        result = self.check_compliance(document, CheckType.VALUES,
                                     subcheck="securities_disclaimers", **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    def check_esg_compliance(self, document: Dict, esg_classification: str = "other",
                           client_type: str = "retail", **kwargs) -> List[ComplianceResult]:
        """Check ESG compliance (classification validation, etc.)"""
        violations = []
        
        # ESG classification validation
        result = self.check_compliance(document, CheckType.ESG,
                                     subcheck="classification_validation",
                                     esg_classification=esg_classification,
                                     client_type=client_type, **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    def check_disclaimers_compliance(self, document: Dict, doc_type: str = "fund_presentation",
                                   client_type: str = "retail", **kwargs) -> List[ComplianceResult]:
        """Check disclaimers compliance (required disclaimers present, etc.)"""
        violations = []
        
        # Required disclaimers validation
        result = self.check_compliance(document, CheckType.DISCLAIMERS,
                                     subcheck="required_disclaimers",
                                     doc_type=doc_type,
                                     client_type=client_type, **kwargs)
        if result:
            violations.append(result)
        
        return violations
    
    # ========================================================================
    # INTERNAL LAYER METHODS
    # ========================================================================
    
    def _rule_pre_filter(self, document: Dict, check_type: CheckType, **kwargs) -> Dict:
        """
        Layer 1: Rule-based pre-filtering for quick screening
        
        Returns:
            Dict with rule-based analysis results and hints for AI
        """
        if not self.config.rule_enabled:
            return {"found": False, "confidence": 0, "hints": []}
        
        # Get appropriate rule engine for check type
        rule_engine = self._get_rule_engine(check_type)
        
        if rule_engine:
            return rule_engine.quick_scan(document, **kwargs)
        
        return {"found": False, "confidence": 0, "hints": []}
    
    def _ai_analysis(self, document: Dict, check_type: CheckType, rule_hints: Dict, **kwargs) -> Optional[Dict]:
        """
        Layer 2: AI analysis with semantic understanding
        
        Args:
            document: Document to analyze
            check_type: Type of check
            rule_hints: Hints from rule-based pre-filtering
            
        Returns:
            Dict with AI analysis results
        """
        if not self.ai_engine:
            return None
        
        try:
            return self.ai_engine.analyze(document, check_type, rule_hints, **kwargs)
        except Exception as e:
            logger.error(f"AI analysis failed for {check_type.value}: {e}")
            return None
    
    def _ai_analysis_with_error_handling(self, document: Dict, check_type: CheckType, 
                                        rule_hints: Dict, **kwargs) -> Optional[Dict]:
        """
        Layer 2: AI analysis with comprehensive error handling
        
        Args:
            document: Document to analyze
            check_type: Type of check
            rule_hints: Hints from rule-based pre-filtering
            
        Returns:
            Dict with AI analysis results
        """
        if not self.ai_engine:
            return None
        
        # Use error handler if available
        if self.error_handler:
            try:
                result = self.error_handler.handle_ai_call(
                    lambda: self.ai_engine.analyze(document, check_type, rule_hints, **kwargs),
                    fallback_func=None  # No fallback, will return None on error
                )
                
                # Record success
                self.error_handler.health_monitor.record_success()
                return result
                
            except Exception as e:
                logger.error(f"AI analysis failed with error handling: {e}")
                self.error_handler.health_monitor.record_error("ai")
                return None
        else:
            # Fallback to simple error handling
            return self._ai_analysis(document, check_type, rule_hints, **kwargs)
    
    def _validate_and_score(self, rule_result: Dict, ai_result: Optional[Dict], 
                           check_type: CheckType) -> Optional[ComplianceResult]:
        """
        Layer 3: Validate AI output and calculate confidence scores
        
        Args:
            rule_result: Results from rule-based analysis
            ai_result: Results from AI analysis
            check_type: Type of check performed
            
        Returns:
            ComplianceResult if violation detected with sufficient confidence
        """
        if not self.confidence_scorer:
            # Fallback scoring logic
            return self._fallback_scoring(rule_result, ai_result, check_type)
        
        return self.confidence_scorer.combine_results(rule_result, ai_result, check_type)
    
    def _fallback_rule_check(self, document: Dict, check_type: CheckType, **kwargs) -> Optional[ComplianceResult]:
        """
        Fallback to rule-only checking when AI fails
        
        Args:
            document: Document to check
            check_type: Type of check
            
        Returns:
            ComplianceResult if violation found
        """
        logger.warning(f"Falling back to rule-only check for {check_type.value}")
        
        rule_result = self._rule_pre_filter(document, check_type, **kwargs)
        
        if rule_result.get("found", False):
            return ComplianceResult(
                violation=True,
                confidence=rule_result.get("confidence", 70),
                status=ComplianceStatus.RULE_ONLY,
                evidence=rule_result.get("evidence", "Rule-based detection"),
                reasoning="AI unavailable - rule-based fallback",
                check_type=check_type,
                slide=rule_result.get("slide", "Unknown"),
                location=rule_result.get("location", "Unknown"),
                rule=rule_result.get("rule", "Unknown rule"),
                message=rule_result.get("message", "Violation detected"),
                rule_hints=str(rule_result.get("hints", []))
            )
        
        return None
    
    def _fallback_scoring(self, rule_result: Dict, ai_result: Optional[Dict], 
                         check_type: CheckType) -> Optional[ComplianceResult]:
        """
        Simple fallback scoring when ConfidenceScorer is not available
        """
        if not ai_result:
            return self._fallback_rule_check({}, check_type)
        
        ai_violation = ai_result.get("violation", False)
        rule_violation = rule_result.get("found", False)
        
        if ai_violation:
            confidence = ai_result.get("confidence", 50)
            
            if rule_violation:
                confidence = min(100, confidence + 15)
                status = ComplianceStatus.VERIFIED_BY_BOTH
            else:
                status = ComplianceStatus.AI_DETECTED_VARIATION
            
            return ComplianceResult(
                violation=True,
                confidence=confidence,
                status=status,
                evidence=ai_result.get("evidence", "AI detection"),
                reasoning=ai_result.get("reasoning", "AI analysis"),
                check_type=check_type,
                slide=ai_result.get("slide", "Unknown"),
                location=ai_result.get("location", "Unknown"),
                rule=ai_result.get("rule", "Unknown rule"),
                message=ai_result.get("message", "Violation detected"),
                ai_reasoning=ai_result.get("reasoning", ""),
                rule_hints=str(rule_result.get("hints", []))
            )
        
        return None
    
    def _get_rule_engine(self, check_type: CheckType):
        """Get rule engine for specific check type"""
        # This will be implemented when we create specific rule engines
        return self._rule_engines.get(check_type)
    
    # ========================================================================
    # CONFIGURATION METHODS
    # ========================================================================
    
    def update_config(self, **kwargs):
        """Update configuration settings"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
    
    def get_config(self) -> HybridConfig:
        """Get current configuration"""
        return self.config
    
    def set_ai_engine(self, ai_engine):
        """Set or update AI engine"""
        self.ai_engine = ai_engine
        logger.info("AI engine updated")
    
    def set_confidence_scorer(self, confidence_scorer):
        """Set or update confidence scorer"""
        self.confidence_scorer = confidence_scorer
        logger.info("Confidence scorer updated")
    
    def get_service_status(self) -> str:
        """Get current AI service status"""
        if self.error_handler:
            return self.error_handler.get_service_status().value
        return "UNKNOWN"
    
    def get_error_metrics(self) -> Dict:
        """Get error metrics"""
        if self.error_handler:
            metrics = self.error_handler.get_metrics()
            return {
                'total_errors': metrics.total_errors,
                'api_errors': metrics.api_errors,
                'timeout_errors': metrics.timeout_errors,
                'parsing_errors': metrics.parsing_errors,
                'fallback_activations': metrics.fallback_activations,
                'successful_retries': metrics.successful_retries,
                'failed_retries': metrics.failed_retries
            }
        return {}
    
    def reset_error_metrics(self):
        """Reset error metrics"""
        if self.error_handler:
            self.error_handler.reset()
            logger.info("Error metrics reset")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance monitoring metrics"""
        if self.performance_monitor:
            return self.performance_monitor.get_performance_summary()
        return {}
    
    def get_performance_dashboard(self):
        """Print performance dashboard"""
        if self.performance_monitor:
            self.performance_monitor.print_dashboard()
        else:
            logger.warning("Performance monitoring not enabled")
    
    def take_performance_snapshot(self) -> Optional[Dict]:
        """Take a snapshot of current performance metrics"""
        if self.performance_monitor:
            cache_stats = self.ai_engine.get_cache_stats() if self.ai_engine else {}
            error_stats = self.get_error_metrics()
            snapshot = self.performance_monitor.take_snapshot(cache_stats, error_stats)
            return snapshot
        return None
    
    def export_performance_metrics(self, filepath: str):
        """Export performance metrics to file"""
        if self.performance_monitor:
            self.performance_monitor.export_metrics(filepath)
        else:
            logger.warning("Performance monitoring not enabled")
    
    def set_feedback_interface(self, feedback_interface):
        """Set or update feedback interface"""
        self.feedback_interface = feedback_interface
        logger.info("Feedback interface updated")
    
    def set_review_manager(self, review_manager):
        """Set or update review manager"""
        self.review_manager = review_manager
        logger.info("Review manager updated")
    
    def get_review_queue_stats(self):
        """Get review queue statistics"""
        if self.review_manager:
            return self.review_manager.get_queue_stats()
        return None
    
    def provide_feedback(self, feedback_id: str, actual_violation: bool,
                        reviewer_notes: str, corrected_confidence: Optional[int] = None,
                        reviewer_id: Optional[str] = None) -> bool:
        """
        Provide human feedback for a prediction
        
        Args:
            feedback_id: ID of the feedback record
            actual_violation: Actual ground truth
            reviewer_notes: Notes from reviewer
            corrected_confidence: Corrected confidence score
            reviewer_id: ID of reviewer
            
        Returns:
            True if feedback was recorded successfully
        """
        if not self.feedback_interface:
            logger.warning("Feedback interface not enabled")
            return False
        
        return self.feedback_interface.provide_correction(
            feedback_id=feedback_id,
            actual_violation=actual_violation,
            reviewer_notes=reviewer_notes,
            corrected_confidence=corrected_confidence,
            reviewer_id=reviewer_id
        )
    
    def get_pending_reviews(self, check_type: Optional[str] = None) -> List:
        """
        Get list of predictions pending human review
        
        Args:
            check_type: Filter by check type
            
        Returns:
            List of pending feedback records
        """
        if not self.feedback_interface:
            logger.warning("Feedback interface not enabled")
            return []
        
        return self.feedback_interface.get_pending_reviews(check_type=check_type)
    
    def export_feedback(self, filepath: str, check_type: Optional[str] = None):
        """
        Export feedback data to file
        
        Args:
            filepath: Path to output file
            check_type: Filter by check type
        """
        if not self.feedback_interface:
            logger.warning("Feedback interface not enabled")
            return
        
        self.feedback_interface.export_feedback(filepath, check_type=check_type)


if __name__ == "__main__":
    # Example usage
    config = HybridConfig(
        ai_enabled=True,
        confidence_threshold=75,
        cache_enabled=True
    )
    
    checker = HybridComplianceChecker(config=config)
    print("HybridComplianceChecker initialized successfully")
    print(f"Configuration: {config}")