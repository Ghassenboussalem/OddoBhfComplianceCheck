#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence Scorer - Result Combination Logic
Combines AI and rule-based results with confidence scoring and status categorization
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Status categories for compliance results"""
    VERIFIED_BY_BOTH = "VERIFIED_BY_BOTH"  # AI and rules both detected violation
    AI_DETECTED_VARIATION = "AI_DETECTED_VARIATION"  # AI found, rules missed
    FALSE_POSITIVE_FILTERED = "FALSE_POSITIVE_FILTERED"  # Rules flagged, AI cleared
    VIOLATION_CONFIRMED = "VIOLATION_CONFIRMED"  # High confidence violation
    RULE_ONLY = "RULE_ONLY"  # Only rules available (AI failed)
    AI_ONLY = "AI_ONLY"  # Only AI available (rules not applicable)
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Low confidence, requires human review


@dataclass
class ScoringThresholds:
    """Configurable thresholds for confidence scoring"""
    high_confidence: int = 85  # Threshold for high confidence violations
    medium_confidence: int = 70  # Threshold for medium confidence
    review_threshold: int = 60  # Below this, flag for human review
    agreement_boost: int = 15  # Confidence boost when AI and rules agree
    disagreement_penalty: int = 20  # Confidence penalty when they disagree
    min_confidence: int = 40  # Minimum confidence score
    max_confidence: int = 100  # Maximum confidence score


class ConfidenceScorer:
    """
    Combines AI and rule-based results with confidence scoring
    
    Implements three-layer validation:
    1. Analyze individual results (AI and rules)
    2. Calculate combined confidence based on agreement
    3. Categorize status and flag for review if needed
    """
    
    def __init__(self, thresholds: Optional[ScoringThresholds] = None):
        """
        Initialize confidence scorer
        
        Args:
            thresholds: Custom scoring thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or ScoringThresholds()
        logger.info("ConfidenceScorer initialized")
        logger.info(f"Thresholds: high={self.thresholds.high_confidence}, "
                   f"medium={self.thresholds.medium_confidence}, "
                   f"review={self.thresholds.review_threshold}")
    
    def combine_results(self, rule_result: Dict, ai_result: Optional[Dict], 
                       check_type: Any) -> Optional[Dict]:
        """
        Combine AI and rule-based results with confidence scoring
        
        Args:
            rule_result: Results from rule-based analysis
            ai_result: Results from AI analysis (can be None if AI unavailable)
            check_type: Type of compliance check
            
        Returns:
            Combined result dict with confidence and status, or None if compliant
        """
        # Handle case where AI is unavailable
        if not ai_result:
            return self._handle_rule_only(rule_result, check_type)
        
        # Extract violation flags
        ai_violation = ai_result.get('violation', False)
        rule_violation = rule_result.get('found', False)
        
        # Extract base confidence scores
        ai_confidence = ai_result.get('confidence', 50)
        rule_confidence = rule_result.get('confidence', 50)
        
        # Determine agreement status and calculate combined confidence
        if ai_violation and rule_violation:
            # Both detected violation - HIGH confidence
            final_confidence, status = self._handle_both_agree_violation(
                ai_confidence, rule_confidence
            )
        elif ai_violation and not rule_violation:
            # AI detected, rules missed - MEDIUM confidence
            final_confidence, status = self._handle_ai_only_violation(
                ai_confidence, ai_result
            )
        elif not ai_violation and rule_violation:
            # Rules flagged, AI cleared - likely false positive
            final_confidence, status = self._handle_false_positive(
                rule_confidence
            )
        else:
            # Both agree it's compliant
            logger.debug(f"{check_type} check: Both AI and rules agree - compliant")
            return None
        
        # Only return violation if confidence meets threshold
        if not ai_violation or final_confidence < self.thresholds.review_threshold:
            logger.debug(f"{check_type} check: Confidence {final_confidence}% "
                        f"below threshold {self.thresholds.review_threshold}%")
            return None
        
        # Build combined result
        result = self._build_result(
            ai_result=ai_result,
            rule_result=rule_result,
            confidence=final_confidence,
            status=status,
            check_type=check_type
        )
        
        logger.info(f"{check_type} violation detected: {status.value} "
                   f"(confidence: {final_confidence}%)")
        
        return result
    
    def _handle_both_agree_violation(self, ai_confidence: int, 
                                    rule_confidence: int) -> tuple[int, ComplianceStatus]:
        """
        Handle case where both AI and rules detect violation
        
        Returns:
            (final_confidence, status)
        """
        # Average the confidences and add agreement boost
        base_confidence = (ai_confidence + rule_confidence) / 2
        final_confidence = min(
            self.thresholds.max_confidence,
            base_confidence + self.thresholds.agreement_boost
        )
        
        # Determine status based on final confidence
        if final_confidence >= self.thresholds.high_confidence:
            status = ComplianceStatus.VIOLATION_CONFIRMED
        else:
            status = ComplianceStatus.VERIFIED_BY_BOTH
        
        logger.debug(f"Both agree on violation: AI={ai_confidence}%, "
                    f"Rule={rule_confidence}%, Final={final_confidence}%")
        
        return int(final_confidence), status
    
    def _handle_ai_only_violation(self, ai_confidence: int, 
                                  ai_result: Dict) -> tuple[int, ComplianceStatus]:
        """
        Handle case where only AI detects violation
        
        Returns:
            (final_confidence, status)
        """
        # Use AI confidence as base
        final_confidence = ai_confidence
        
        # Check if AI provides strong evidence
        has_evidence = bool(ai_result.get('evidence'))
        has_reasoning = bool(ai_result.get('reasoning'))
        
        # Boost confidence if AI provides detailed evidence
        if has_evidence and has_reasoning:
            final_confidence = min(
                self.thresholds.max_confidence,
                final_confidence + 5
            )
        
        status = ComplianceStatus.AI_DETECTED_VARIATION
        
        logger.debug(f"AI-only violation: confidence={final_confidence}%, "
                    f"evidence={has_evidence}, reasoning={has_reasoning}")
        
        return int(final_confidence), status
    
    def _handle_false_positive(self, rule_confidence: int) -> tuple[int, ComplianceStatus]:
        """
        Handle case where rules flag but AI clears (likely false positive)
        
        Returns:
            (final_confidence, status)
        """
        # Apply disagreement penalty
        final_confidence = max(
            self.thresholds.min_confidence,
            rule_confidence - self.thresholds.disagreement_penalty
        )
        
        status = ComplianceStatus.FALSE_POSITIVE_FILTERED
        
        logger.debug(f"Potential false positive: rule={rule_confidence}%, "
                    f"final={final_confidence}%")
        
        return int(final_confidence), status
    
    def _handle_rule_only(self, rule_result: Dict, check_type: Any) -> Optional[Dict]:
        """
        Handle case where only rule-based analysis is available
        
        Returns:
            Result dict or None if compliant
        """
        if not rule_result.get('found', False):
            return None
        
        confidence = rule_result.get('confidence', 70)
        
        # Apply penalty for lack of AI validation
        confidence = max(
            self.thresholds.min_confidence,
            confidence - 10
        )
        
        logger.warning(f"{check_type} check: Rule-only mode (AI unavailable)")
        
        return {
            'violation': True,
            'confidence': int(confidence),
            'status': ComplianceStatus.RULE_ONLY.value,
            'evidence': rule_result.get('evidence', 'Rule-based detection'),
            'reasoning': 'AI unavailable - rule-based fallback',
            'check_type': str(check_type),
            'slide': rule_result.get('slide', 'Unknown'),
            'location': rule_result.get('location', 'Unknown'),
            'rule': rule_result.get('rule', 'Unknown rule'),
            'message': rule_result.get('message', 'Violation detected'),
            'severity': rule_result.get('severity', 'CRITICAL'),
            'rule_hints': str(rule_result.get('hints', []))
        }
    
    def _build_result(self, ai_result: Dict, rule_result: Dict, 
                     confidence: int, status: ComplianceStatus, 
                     check_type: Any) -> Dict:
        """
        Build combined result dictionary
        
        Args:
            ai_result: AI analysis results
            rule_result: Rule-based results
            confidence: Final confidence score
            status: Compliance status
            check_type: Type of check
            
        Returns:
            Combined result dict
        """
        # Determine if human review is needed
        needs_review = confidence < self.thresholds.review_threshold
        
        # Use AI results as primary source, fall back to rules
        result = {
            'violation': True,
            'confidence': confidence,
            'status': status.value,
            'needs_review': needs_review,
            'check_type': str(check_type),
            
            # Content from AI (preferred) or rules (fallback)
            'slide': ai_result.get('slide') or rule_result.get('slide', 'Unknown'),
            'location': ai_result.get('location') or rule_result.get('location', 'Unknown'),
            'rule': ai_result.get('rule') or rule_result.get('rule', 'Unknown rule'),
            'message': ai_result.get('message') or rule_result.get('message', 'Violation detected'),
            'evidence': ai_result.get('evidence') or rule_result.get('evidence', 'Detected'),
            'severity': ai_result.get('severity') or rule_result.get('severity', 'CRITICAL'),
            
            # Additional context
            'ai_reasoning': ai_result.get('reasoning', ''),
            'rule_hints': str(rule_result.get('hints', [])),
            
            # Metadata
            'ai_confidence': ai_result.get('confidence', 0),
            'rule_confidence': rule_result.get('confidence', 0)
        }
        
        return result
    
    def calculate_confidence(self, ai_confidence: int, rule_confidence: int,
                           agreement: bool) -> int:
        """
        Calculate combined confidence score
        
        Args:
            ai_confidence: AI confidence (0-100)
            rule_confidence: Rule confidence (0-100)
            agreement: Whether AI and rules agree
            
        Returns:
            Combined confidence score (0-100)
        """
        if agreement:
            # Average and boost
            base = (ai_confidence + rule_confidence) / 2
            final = min(
                self.thresholds.max_confidence,
                base + self.thresholds.agreement_boost
            )
        else:
            # Use higher confidence but apply penalty
            base = max(ai_confidence, rule_confidence)
            final = max(
                self.thresholds.min_confidence,
                base - self.thresholds.disagreement_penalty
            )
        
        return int(final)
    
    def should_flag_for_review(self, confidence: int) -> bool:
        """
        Determine if result should be flagged for human review
        
        Args:
            confidence: Confidence score
            
        Returns:
            True if should be reviewed by human
        """
        return confidence < self.thresholds.review_threshold
    
    def get_confidence_category(self, confidence: int) -> str:
        """
        Categorize confidence level
        
        Args:
            confidence: Confidence score
            
        Returns:
            Category string: "HIGH", "MEDIUM", "LOW", or "REVIEW"
        """
        if confidence >= self.thresholds.high_confidence:
            return "HIGH"
        elif confidence >= self.thresholds.medium_confidence:
            return "MEDIUM"
        elif confidence >= self.thresholds.review_threshold:
            return "LOW"
        else:
            return "REVIEW"
    
    def update_thresholds(self, **kwargs):
        """
        Update scoring thresholds
        
        Args:
            **kwargs: Threshold values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(f"Updated threshold: {key} = {value}")
    
    def get_thresholds(self) -> ScoringThresholds:
        """Get current thresholds"""
        return self.thresholds


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Confidence Scorer - Result Combination Logic")
    print("="*70)
    
    scorer = ConfidenceScorer()
    
    # Test case 1: Both agree on violation
    print("\n📊 Test 1: Both AI and rules detect violation")
    rule_result = {
        'found': True,
        'confidence': 80,
        'evidence': 'Keyword "promotional" found',
        'slide': 'Cover Page',
        'location': 'Top section',
        'rule': 'Promotional mention required',
        'message': 'Missing promotional document mention'
    }
    
    ai_result = {
        'violation': True,
        'confidence': 85,
        'evidence': 'Document lacks promotional indication',
        'reasoning': 'No clear promotional language found on cover',
        'slide': 'Cover Page',
        'location': 'Header',
        'rule': 'Promotional mention required',
        'message': 'Promotional document mention missing'
    }
    
    result = scorer.combine_results(rule_result, ai_result, "PROMOTIONAL_MENTION")
    if result:
        print(f"  ✓ Violation detected")
        print(f"    Status: {result['status']}")
        print(f"    Confidence: {result['confidence']}%")
        print(f"    Category: {scorer.get_confidence_category(result['confidence'])}")
        print(f"    Needs review: {result['needs_review']}")
    
    # Test case 2: AI detects, rules miss
    print("\n📊 Test 2: AI detects variation, rules miss")
    rule_result = {
        'found': False,
        'confidence': 0,
        'hints': []
    }
    
    ai_result = {
        'violation': True,
        'confidence': 75,
        'evidence': 'Found "document à caractère promotionnel"',
        'reasoning': 'Variation of promotional mention detected',
        'slide': 'Cover Page',
        'rule': 'Promotional mention required',
        'message': 'Non-standard promotional mention'
    }
    
    result = scorer.combine_results(rule_result, ai_result, "PROMOTIONAL_MENTION")
    if result:
        print(f"  ✓ Violation detected")
        print(f"    Status: {result['status']}")
        print(f"    Confidence: {result['confidence']}%")
        print(f"    Category: {scorer.get_confidence_category(result['confidence'])}")
    
    # Test case 3: Rules flag, AI clears (false positive)
    print("\n📊 Test 3: Rules flag, AI clears (false positive)")
    rule_result = {
        'found': True,
        'confidence': 70,
        'evidence': 'Keyword match',
        'slide': 'Page 5',
        'rule': 'Performance disclaimer required',
        'message': 'Missing disclaimer'
    }
    
    ai_result = {
        'violation': False,
        'confidence': 90,
        'reasoning': 'Disclaimer present in different wording',
        'evidence': 'Semantic match found'
    }
    
    result = scorer.combine_results(rule_result, ai_result, "PERFORMANCE_DISCLAIMER")
    if result:
        print(f"  ⚠ Potential false positive")
        print(f"    Status: {result['status']}")
        print(f"    Confidence: {result['confidence']}%")
    else:
        print(f"  ✓ False positive filtered - no violation")
    
    # Test case 4: Rule-only mode
    print("\n📊 Test 4: Rule-only mode (AI unavailable)")
    rule_result = {
        'found': True,
        'confidence': 75,
        'evidence': 'Rule-based detection',
        'slide': 'Page 3',
        'rule': 'Fund name must match',
        'message': 'Fund name mismatch'
    }
    
    result = scorer.combine_results(rule_result, None, "FUND_NAME_MATCH")
    if result:
        print(f"  ✓ Violation detected (rule-only)")
        print(f"    Status: {result['status']}")
        print(f"    Confidence: {result['confidence']}%")
    
    print("\n" + "="*70)
    print("✓ Confidence Scorer tests complete")
    print("="*70)
