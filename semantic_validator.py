#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Validator - Meaning-Based Validation
Integrates all components for AI-powered semantic compliance validation
"""

import json
import re
import logging
from typing import Dict, List, Optional, Set, Any
from data_models import ValidationResult, ValidationMethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticValidator:
    """
    Validates compliance based on semantic meaning, not keywords
    
    Integrates:
    - ContextAnalyzer: Understand semantic meaning and context
    - IntentClassifier: Classify text intent (advice vs description)
    - EvidenceExtractor: Extract and quote evidence
    - WhitelistManager: Filter allowed terms
    
    Key capabilities:
    - Validate securities mentions with whitelist filtering
    - Detect actual performance data vs keywords
    - Check prospectus consistency semantically
    - Combine AI and rule-based validation
    - Provide confidence scoring and explainable reasoning
    """
    
    def __init__(self, ai_engine, context_analyzer=None, intent_classifier=None, 
                 evidence_extractor=None, whitelist_manager=None):
        """
        Initialize Semantic Validator
        
        Args:
            ai_engine: AIEngine instance for LLM calls
            context_analyzer: Optional ContextAnalyzer instance (created if None)
            intent_classifier: Optional IntentClassifier instance (created if None)
            evidence_extractor: Optional EvidenceExtractor instance (created if None)
            whitelist_manager: Optional WhitelistManager instance (created if None)
        """
        self.ai_engine = ai_engine
        
        # Import components (lazy import to avoid circular dependencies)
        from context_analyzer import ContextAnalyzer
        from intent_classifier import IntentClassifier
        from evidence_extractor import EvidenceExtractor
        from whitelist_manager import WhitelistManager
        
        # Initialize or use provided components
        self.context_analyzer = context_analyzer or ContextAnalyzer(ai_engine)
        self.intent_classifier = intent_classifier or IntentClassifier(ai_engine)
        self.evidence_extractor = evidence_extractor or EvidenceExtractor(ai_engine)
        self.whitelist_manager = whitelist_manager or WhitelistManager()
        
        logger.info("SemanticValidator initialized with all components")
    
    def validate_securities_mention(self, text: str, term: str, whitelist: Set[str], 
                                   mention_count: int = 0) -> ValidationResult:
        """
        Validate if a term is a prohibited securities mention
        
        Uses whitelist filtering and semantic analysis to distinguish:
        - Fund name components (ALLOWED)
        - Strategy terminology (ALLOWED)
        - Regulatory terms (ALLOWED)
        - External company names (PROHIBITED if mentioned 3+ times)
        
        Args:
            text: Full text to analyze
            term: Term to validate
            whitelist: Set of whitelisted terms
            mention_count: Number of times term appears
            
        Returns:
            ValidationResult with violation flag and reasoning
            
        Examples:
            >>> validator.validate_securities_mention(text, "ODDO", whitelist, 31)
            ValidationResult(is_violation=False, reasoning="Whitelisted: fund_name")
            
            >>> validator.validate_securities_mention(text, "Apple", whitelist, 5)
            ValidationResult(is_violation=True, reasoning="External company mentioned 5 times")
        """
        try:
            term_lower = term.lower().strip()
            
            # Check whitelist first (fast path)
            if term_lower in whitelist or self.whitelist_manager.is_whitelisted(term):
                reason = self.whitelist_manager.get_whitelist_reason(term)
                return ValidationResult(
                    is_violation=False,
                    confidence=95,
                    reasoning=f"Whitelisted term: {reason or 'allowed term'}",
                    evidence=[f"'{term}' is in whitelist"],
                    method=ValidationMethod.RULES_ONLY.value
                )
            
            # If mentioned less than 3 times, not a violation
            if mention_count < 3:
                return ValidationResult(
                    is_violation=False,
                    confidence=90,
                    reasoning=f"Term mentioned only {mention_count} times (threshold: 3+)",
                    evidence=[f"'{term}' appears {mention_count} times"],
                    method=ValidationMethod.RULES_ONLY.value
                )
            
            # Use AI to verify if it's actually a company name
            ai_result = self._ai_verify_company_name(text, term)
            
            # Use rule-based heuristics as backup
            rule_result = self._rule_based_company_check(text, term)
            
            # Combine AI and rule-based results
            if ai_result and rule_result:
                # Both agree it's a company
                is_company = ai_result.get('is_company', False) and rule_result['is_company']
                confidence = min(ai_result.get('confidence', 70), rule_result['confidence'])
                
                if is_company:
                    return ValidationResult(
                        is_violation=True,
                        confidence=confidence,
                        reasoning=f"External company '{term}' mentioned {mention_count} times. "
                                f"AI: {ai_result.get('reasoning', 'N/A')}. "
                                f"Rules: {rule_result['reasoning']}",
                        evidence=[f"'{term}' appears {mention_count} times in document"],
                        method=ValidationMethod.AI_AND_RULES.value
                    )
                else:
                    return ValidationResult(
                        is_violation=False,
                        confidence=confidence,
                        reasoning=f"'{term}' not identified as external company. "
                                f"AI: {ai_result.get('reasoning', 'N/A')}",
                        evidence=[f"'{term}' appears {mention_count} times but not a company"],
                        method=ValidationMethod.AI_AND_RULES.value
                    )
            
            elif ai_result:
                # AI only
                is_company = ai_result.get('is_company', False)
                confidence = ai_result.get('confidence', 60)
                
                return ValidationResult(
                    is_violation=is_company,
                    confidence=confidence,
                    reasoning=f"AI analysis: {ai_result.get('reasoning', 'Company name detected')}",
                    evidence=[f"'{term}' appears {mention_count} times"],
                    method=ValidationMethod.AI_ONLY.value
                )
            
            else:
                # Rules only (fallback)
                is_company = rule_result['is_company']
                confidence = rule_result['confidence']
                
                return ValidationResult(
                    is_violation=is_company,
                    confidence=confidence,
                    reasoning=f"Rule-based: {rule_result['reasoning']}",
                    evidence=[f"'{term}' appears {mention_count} times"],
                    method=ValidationMethod.RULES_ONLY.value
                )
        
        except Exception as e:
            logger.error(f"Securities validation error for '{term}': {e}")
            return ValidationResult(
                is_violation=False,
                confidence=0,
                reasoning=f"Validation error: {str(e)}",
                evidence=[],
                method=ValidationMethod.RULES_ONLY.value
            )
    
    def validate_performance_claim(self, text: str, location: str = "") -> ValidationResult:
        """
        Validate if text contains actual performance data requiring disclaimers
        
        Distinguishes:
        - ACTUAL DATA: "15% return in 2024" → needs disclaimer
        - KEYWORDS: "attractive performance", "performance objective" → no disclaimer needed
        
        Args:
            text: Text to analyze
            location: Location in document (e.g., "Slide 2")
            
        Returns:
            ValidationResult indicating if actual performance data is present
            
        Examples:
            >>> validator.validate_performance_claim("Le fonds a généré 15% en 2023")
            ValidationResult(is_violation=True, reasoning="Actual performance data found")
            
            >>> validator.validate_performance_claim("Performance objective attractive")
            ValidationResult(is_violation=False, reasoning="Only descriptive keywords")
        """
        try:
            # Use EvidenceExtractor to find actual performance data
            performance_data = self.evidence_extractor.find_performance_data(text)
            
            if not performance_data:
                # No actual performance data found
                return ValidationResult(
                    is_violation=False,
                    confidence=90,
                    reasoning="No actual performance data found (only descriptive keywords)",
                    evidence=["Text contains performance keywords but no numerical data"],
                    method=ValidationMethod.AI_AND_RULES.value
                )
            
            # Found actual performance data
            # Extract evidence
            evidence_list = [
                f"{pd.value} at {pd.location} (confidence: {pd.confidence}%)"
                for pd in performance_data[:3]
            ]
            
            # Calculate overall confidence
            avg_confidence = sum(pd.confidence for pd in performance_data) / len(performance_data)
            
            return ValidationResult(
                is_violation=True,
                confidence=int(avg_confidence),
                reasoning=f"Found {len(performance_data)} actual performance data point(s) requiring disclaimer",
                evidence=evidence_list,
                method=ValidationMethod.AI_AND_RULES.value
            )
        
        except Exception as e:
            logger.error(f"Performance claim validation error: {e}")
            return ValidationResult(
                is_violation=False,
                confidence=0,
                reasoning=f"Validation error: {str(e)}",
                evidence=[],
                method=ValidationMethod.RULES_ONLY.value
            )
    
    def validate_prospectus_consistency(self, doc_text: str, prospectus_text: str,
                                       check_type: str = "general") -> ValidationResult:
        """
        Check for contradictions between marketing text and prospectus
        
        Distinguishes:
        - CONTRADICTION: "invests in European stocks" vs "US stocks only" → violation
        - MISSING DETAIL: "invests in S&P 500" vs "at least 70% in S&P 500" → not violation
        
        Args:
            doc_text: Marketing document text
            prospectus_text: Prospectus text
            check_type: Type of consistency check
            
        Returns:
            ValidationResult indicating if contradiction exists
            
        Examples:
            >>> validator.validate_prospectus_consistency(
            ...     "Le fonds investit en actions européennes",
            ...     "The fund invests exclusively in US equities"
            ... )
            ValidationResult(is_violation=True, reasoning="Contradiction detected")
        """
        try:
            # Use AI for semantic consistency check
            ai_result = self._ai_consistency_check(doc_text, prospectus_text, check_type)
            
            if ai_result:
                is_contradiction = ai_result.get('is_contradiction', False)
                confidence = ai_result.get('confidence', 60)
                reasoning = ai_result.get('reasoning', 'Consistency check completed')
                evidence = ai_result.get('evidence', [])
                
                # Lower confidence for borderline cases
                if confidence < 70:
                    reasoning += " (Low confidence - recommend human review)"
                
                return ValidationResult(
                    is_violation=is_contradiction,
                    confidence=confidence,
                    reasoning=reasoning,
                    evidence=evidence if isinstance(evidence, list) else [str(evidence)],
                    method=ValidationMethod.AI_ONLY.value
                )
            else:
                # Fallback to rule-based check
                return self._rule_based_consistency_check(doc_text, prospectus_text)
        
        except Exception as e:
            logger.error(f"Prospectus consistency validation error: {e}")
            return ValidationResult(
                is_violation=False,
                confidence=0,
                reasoning=f"Validation error: {str(e)}",
                evidence=[],
                method=ValidationMethod.RULES_ONLY.value
            )
    
    def validate_with_confidence_scoring(self, ai_result: Dict, rule_result: Dict) -> ValidationResult:
        """
        Combine AI and rule-based results with confidence scoring
        
        Decision Matrix:
        - AI + Rules agree (both high confidence) → High confidence result
        - AI + Rules agree (one low confidence) → Medium confidence result
        - AI + Rules disagree → Flag for human review (low confidence)
        - AI only (rules unavailable) → Medium confidence result
        - Rules only (AI unavailable) → Low confidence result
        
        Args:
            ai_result: Result from AI analysis
            rule_result: Result from rule-based analysis
            
        Returns:
            ValidationResult with combined confidence score
        """
        try:
            # Extract values from results
            ai_violation = ai_result.get('is_violation', False) if ai_result else None
            rule_violation = rule_result.get('is_violation', False) if rule_result else None
            
            ai_confidence = ai_result.get('confidence', 0) if ai_result else 0
            rule_confidence = rule_result.get('confidence', 0) if rule_result else 0
            
            # Both available
            if ai_result and rule_result:
                if ai_violation == rule_violation:
                    # Agreement
                    if ai_confidence >= 80 and rule_confidence >= 80:
                        # High confidence agreement
                        confidence = min(95, (ai_confidence + rule_confidence) // 2)
                        method = ValidationMethod.AI_AND_RULES.value
                    else:
                        # Medium confidence agreement
                        confidence = (ai_confidence + rule_confidence) // 2
                        method = ValidationMethod.AI_AND_RULES.value
                    
                    reasoning = f"AI and rules agree. AI: {ai_result.get('reasoning', 'N/A')}. Rules: {rule_result.get('reasoning', 'N/A')}"
                    is_violation = ai_violation
                else:
                    # Disagreement - flag for review
                    confidence = 50
                    method = ValidationMethod.AI_AND_RULES.value
                    reasoning = f"AI and rules disagree (recommend human review). AI: {ai_result.get('reasoning', 'N/A')}. Rules: {rule_result.get('reasoning', 'N/A')}"
                    is_violation = ai_violation if ai_confidence > rule_confidence else rule_violation
            
            # AI only
            elif ai_result:
                confidence = min(ai_confidence, 85)  # Cap at 85 for AI-only
                method = ValidationMethod.AI_ONLY.value
                reasoning = f"AI analysis only: {ai_result.get('reasoning', 'N/A')}"
                is_violation = ai_violation
            
            # Rules only
            elif rule_result:
                confidence = min(rule_confidence, 70)  # Cap at 70 for rules-only
                method = ValidationMethod.RULES_ONLY.value
                reasoning = f"Rule-based analysis only: {rule_result.get('reasoning', 'N/A')}"
                is_violation = rule_violation
            
            else:
                # Neither available
                return ValidationResult(
                    is_violation=False,
                    confidence=0,
                    reasoning="No analysis results available",
                    evidence=[],
                    method=ValidationMethod.RULES_ONLY.value
                )
            
            # Combine evidence
            evidence = []
            if ai_result and 'evidence' in ai_result:
                evidence.extend(ai_result['evidence'] if isinstance(ai_result['evidence'], list) else [ai_result['evidence']])
            if rule_result and 'evidence' in rule_result:
                evidence.extend(rule_result['evidence'] if isinstance(rule_result['evidence'], list) else [rule_result['evidence']])
            
            return ValidationResult(
                is_violation=is_violation,
                confidence=confidence,
                reasoning=reasoning,
                evidence=evidence,
                method=method
            )
        
        except Exception as e:
            logger.error(f"Confidence scoring error: {e}")
            return ValidationResult(
                is_violation=False,
                confidence=0,
                reasoning=f"Scoring error: {str(e)}",
                evidence=[],
                method=ValidationMethod.RULES_ONLY.value
            )
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _ai_verify_company_name(self, text: str, term: str) -> Optional[Dict]:
        """
        Use AI to verify if a term is a company name
        
        Args:
            text: Full text context
            term: Term to verify
            
        Returns:
            Dict with verification result or None
        """
        try:
            if not self.ai_engine:
                return None
            
            prompt = f"""Is this term an external company name in the context of this financial document?

TERM: {term}

CONTEXT: {text[:1000]}

Consider:
- Is it a publicly traded company?
- Is it mentioned as an investment or holding?
- Is it the fund name itself? (NOT external)
- Is it a strategy term? (NOT a company)
- Is it a regulatory term? (NOT a company)

Respond with JSON only:
{{
  "is_company": true/false,
  "confidence": 0-100,
  "reasoning": "explanation",
  "company_type": "public|private|fund_name|strategy_term|regulatory_term|other"
}}"""
            
            response = self.ai_engine.call_with_cache(
                prompt=prompt,
                system_message="You are a financial analyst. Identify company names accurately. Return only valid JSON."
            )
            
            if response and response.parsed_json:
                return response.parsed_json
            
            return None
        
        except Exception as e:
            logger.error(f"AI company verification error: {e}")
            return None
    
    def _rule_based_company_check(self, text: str, term: str) -> Dict:
        """
        Rule-based heuristics to check if term is a company name
        
        Args:
            text: Full text context
            term: Term to check
            
        Returns:
            Dict with check result
        """
        try:
            term_lower = term.lower()
            text_lower = text.lower()
            
            # Heuristics for company names
            is_company = False
            confidence = 50
            reasoning = "Insufficient evidence"
            
            # Check if term is capitalized consistently
            if term[0].isupper() and len(term) > 2:
                confidence += 10
                reasoning = "Capitalized term"
            
            # Check for company indicators in context
            company_indicators = [
                r'\b' + re.escape(term_lower) + r'\b.*\b(stock|share|equity|action)\b',
                r'\b(invest|holding|position).*\b' + re.escape(term_lower) + r'\b',
                r'\b' + re.escape(term_lower) + r'\b.*\b(ticker|symbol|nasdaq|nyse)\b',
            ]
            
            for pattern in company_indicators:
                if re.search(pattern, text_lower):
                    is_company = True
                    confidence += 20
                    reasoning = "Company context indicators found"
                    break
            
            # Check if it's a known non-company term
            non_company_terms = ['fund', 'fonds', 'strategy', 'stratégie', 'portfolio', 'portefeuille']
            if any(nct in term_lower for nct in non_company_terms):
                is_company = False
                confidence = 80
                reasoning = "Identified as fund/strategy term, not company"
            
            return {
                'is_company': is_company,
                'confidence': min(confidence, 85),
                'reasoning': reasoning
            }
        
        except Exception as e:
            logger.error(f"Rule-based company check error: {e}")
            return {
                'is_company': False,
                'confidence': 0,
                'reasoning': f"Error: {str(e)}"
            }
    
    def _ai_consistency_check(self, doc_text: str, prospectus_text: str, 
                             check_type: str) -> Optional[Dict]:
        """
        Use AI to check semantic consistency between documents
        
        Args:
            doc_text: Marketing document text
            prospectus_text: Prospectus text
            check_type: Type of check
            
        Returns:
            Dict with consistency check result or None
        """
        try:
            if not self.ai_engine:
                return None
            
            prompt = f"""Compare these two texts for contradictions (not missing details).

MARKETING TEXT: {doc_text[:800]}

PROSPECTUS TEXT: {prospectus_text[:800]}

CHECK TYPE: {check_type}

DISTINGUISH:
1. CONTRADICTION (violation):
   - Marketing: "invests in European stocks"
   - Prospectus: "invests exclusively in US stocks"
   - These directly contradict each other

2. MISSING DETAIL (not violation):
   - Marketing: "invests in S&P 500"
   - Prospectus: "invests at least 70% in S&P 500 equities"
   - Marketing is simplified but not contradictory

TASK:
- Identify actual contradictions
- Ignore missing technical details
- Consider semantic equivalence
- Provide confidence score

Respond with JSON only:
{{
  "is_contradiction": true/false,
  "confidence": 0-100,
  "reasoning": "detailed explanation",
  "evidence": ["specific contradictions found"],
  "contradiction_type": "investment_strategy|geography|asset_class|risk_level|other"
}}"""
            
            response = self.ai_engine.call_with_cache(
                prompt=prompt,
                system_message="You are a compliance expert. Identify contradictions, not missing details. Return only valid JSON."
            )
            
            if response and response.parsed_json:
                return response.parsed_json
            
            return None
        
        except Exception as e:
            logger.error(f"AI consistency check error: {e}")
            return None
    
    def _rule_based_consistency_check(self, doc_text: str, prospectus_text: str) -> ValidationResult:
        """
        Fallback rule-based consistency check
        
        Args:
            doc_text: Marketing document text
            prospectus_text: Prospectus text
            
        Returns:
            ValidationResult with consistency check
        """
        try:
            doc_lower = doc_text.lower()
            prosp_lower = prospectus_text.lower()
            
            # Simple keyword-based contradiction detection
            contradictions = []
            
            # Geography contradictions
            if 'europe' in doc_lower and 'us' in prosp_lower and 'only' in prosp_lower:
                contradictions.append("Geographic focus mismatch")
            
            if 'us' in doc_lower and 'europe' in prosp_lower and 'only' in prosp_lower:
                contradictions.append("Geographic focus mismatch")
            
            # Asset class contradictions
            if 'equity' in doc_lower and 'bond' in prosp_lower and 'only' in prosp_lower:
                contradictions.append("Asset class mismatch")
            
            if 'bond' in doc_lower and 'equity' in prosp_lower and 'only' in prosp_lower:
                contradictions.append("Asset class mismatch")
            
            if contradictions:
                return ValidationResult(
                    is_violation=True,
                    confidence=60,
                    reasoning=f"Rule-based: Potential contradictions detected: {', '.join(contradictions)}",
                    evidence=contradictions,
                    method=ValidationMethod.RULES_ONLY.value
                )
            else:
                return ValidationResult(
                    is_violation=False,
                    confidence=50,
                    reasoning="Rule-based: No obvious contradictions detected (low confidence)",
                    evidence=[],
                    method=ValidationMethod.RULES_ONLY.value
                )
        
        except Exception as e:
            logger.error(f"Rule-based consistency check error: {e}")
            return ValidationResult(
                is_violation=False,
                confidence=0,
                reasoning=f"Check error: {str(e)}",
                evidence=[],
                method=ValidationMethod.RULES_ONLY.value
            )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_semantic_validator(ai_engine) -> SemanticValidator:
    """
    Factory function to create SemanticValidator with all components
    
    Args:
        ai_engine: AIEngine instance
        
    Returns:
        Configured SemanticValidator
    """
    return SemanticValidator(ai_engine)


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Semantic Validator - Standalone Test")
    print("="*70)
    
    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    if not ai_engine:
        print("\n✗ Failed to initialize AI Engine")
        print("  Using fallback rule-based validation only")
        print("\nNote: Set GEMINI_API_KEY or TOKENFACTORY_API_KEY in .env for full AI validation")
    
    # Create validator
    validator = SemanticValidator(ai_engine) if ai_engine else None
    
    if validator:
        print("\n✓ Semantic Validator initialized")
        print("  - Context Analyzer: Ready")
        print("  - Intent Classifier: Ready")
        print("  - Evidence Extractor: Ready")
        print("  - Whitelist Manager: Ready")
        
        # Test 1: Securities mention validation
        print("\n" + "="*70)
        print("Test 1: Securities Mention Validation")
        print("="*70)
        
        test_text = """
        ODDO BHF Algo Trend US est un fonds qui investit dans des actions américaines.
        Le fonds utilise une stratégie momentum quantitative.
        """
        
        whitelist = {'oddo', 'bhf', 'momentum', 'quantitative'}
        
        # Test whitelisted term
        print("\n[Test 1a] Whitelisted term: 'ODDO'")
        result = validator.validate_securities_mention(test_text, "ODDO", whitelist, 31)
        print(f"  Violation: {result.is_violation}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Method: {result.method}")
        print(f"  Result: {'✓ PASS' if not result.is_violation else '✗ FAIL'}")
        
        # Test external company
        print("\n[Test 1b] External company: 'Apple'")
        result = validator.validate_securities_mention(test_text, "Apple", whitelist, 5)
        print(f"  Violation: {result.is_violation}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Method: {result.method}")
        print(f"  Result: {'✓ PASS' if result.is_violation else '✗ FAIL'}")
        
        # Test 2: Performance claim validation
        print("\n" + "="*70)
        print("Test 2: Performance Claim Validation")
        print("="*70)
        
        # Test actual performance data
        print("\n[Test 2a] Actual performance data")
        perf_text = "Le fonds a généré une performance de +15.5% en 2023"
        result = validator.validate_performance_claim(perf_text, "Slide 2")
        print(f"  Violation: {result.is_violation}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Evidence: {result.evidence}")
        print(f"  Result: {'✓ PASS' if result.is_violation else '✗ FAIL'}")
        
        # Test descriptive keywords
        print("\n[Test 2b] Descriptive keywords only")
        desc_text = "L'objectif de performance est d'obtenir des résultats attractifs"
        result = validator.validate_performance_claim(desc_text, "Slide 3")
        print(f"  Violation: {result.is_violation}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Result: {'✓ PASS' if not result.is_violation else '✗ FAIL'}")
        
        # Test 3: Confidence scoring
        print("\n" + "="*70)
        print("Test 3: Confidence Scoring (AI + Rules)")
        print("="*70)
        
        ai_result = {
            'is_violation': True,
            'confidence': 85,
            'reasoning': 'AI detected investment advice',
            'evidence': ['Phrase: "vous devriez investir"']
        }
        
        rule_result = {
            'is_violation': True,
            'confidence': 80,
            'reasoning': 'Rules detected advice pattern',
            'evidence': ['Pattern match: advice indicator']
        }
        
        print("\n[Test 3a] AI and Rules agree (high confidence)")
        result = validator.validate_with_confidence_scoring(ai_result, rule_result)
        print(f"  Violation: {result.is_violation}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Method: {result.method}")
        print(f"  Reasoning: {result.reasoning[:100]}...")
        print(f"  Result: {'✓ PASS' if result.confidence >= 85 else '✗ FAIL'}")
        
        # Test disagreement
        rule_result_disagree = {
            'is_violation': False,
            'confidence': 70,
            'reasoning': 'Rules did not detect violation',
            'evidence': []
        }
        
        print("\n[Test 3b] AI and Rules disagree")
        result = validator.validate_with_confidence_scoring(ai_result, rule_result_disagree)
        print(f"  Violation: {result.is_violation}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Method: {result.method}")
        print(f"  Reasoning: {result.reasoning[:100]}...")
        print(f"  Result: {'✓ PASS' if result.confidence <= 60 else '✗ FAIL'} (should flag for review)")
        
        print("\n" + "="*70)
        print("Test Complete")
        print("="*70)
    else:
        print("\n✗ Cannot run tests without AI Engine")
        print("  Set API keys in .env file to enable testing")
