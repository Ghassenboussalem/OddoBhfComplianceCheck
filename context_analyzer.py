#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Analyzer - AI-Powered Semantic Understanding
Understands the semantic meaning and context of text passages to distinguish
fund strategy descriptions from investment advice.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any
from data_models import ContextAnalysis, SubjectType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """
    Analyzes text context using AI to understand semantic meaning and intent.
    
    Key capabilities:
    - Distinguish fund strategy descriptions from investment advice
    - Identify WHO performs actions (fund vs client)
    - Understand intent (describe vs advise)
    - Provide explainable reasoning
    """
    
    def __init__(self, ai_engine):
        """
        Initialize Context Analyzer
        
        Args:
            ai_engine: AIEngine instance for LLM calls
        """
        self.ai_engine = ai_engine
        self._init_prompt_templates()
        logger.info("ContextAnalyzer initialized")
    
    def _init_prompt_templates(self):
        """Initialize AI prompt templates for context analysis"""
        
        # Investment advice vs fund description template
        self.investment_advice_prompt = """Analyze this text and determine if it contains investment advice to clients or describes the fund's strategy.

TEXT: {text}

CONTEXT:
- This is from a fund marketing document
- French/EU regulations prohibit direct investment advice in marketing materials
- Fund strategy descriptions are ALLOWED
- Client advice is PROHIBITED

DISTINGUISH:
1. Fund strategy description (ALLOWED):
   - "Le fonds investit dans..." (The fund invests in...)
   - "La stratégie vise à..." (The strategy aims to...)
   - "Tirer parti du momentum" (Take advantage of momentum - describing fund goal)
   - Subject is the FUND or STRATEGY

2. Client advice (PROHIBITED):
   - "Vous devriez investir" (You should invest)
   - "Nous recommandons d'acheter" (We recommend buying)
   - "Bon moment pour investir" (Good time to invest)
   - Subject is the CLIENT or INVESTOR

TASK:
1. Identify WHO is performing the action (fund/strategy vs client/investor)
2. Determine the INTENT (describe fund vs advise client)
3. Provide confidence score (0-100)
4. Explain your reasoning with specific evidence

Respond with JSON only, no additional text:
{{
  "is_investment_advice": true/false,
  "is_fund_description": true/false,
  "subject": "fund|client|general",
  "intent": "describe|advise|state_fact",
  "confidence": 0-100,
  "reasoning": "detailed explanation",
  "evidence": ["key phrase 1", "key phrase 2"]
}}"""

        # Generic context analysis template
        self.context_analysis_prompt = """Analyze the context and meaning of this text.

TEXT: {text}

CHECK TYPE: {check_type}

TASK:
1. Identify the subject (WHO performs the action)
2. Determine the intent (WHAT is the purpose)
3. Assess if this is fund description or client advice
4. Provide confidence and reasoning

Respond with JSON only, no additional text:
{{
  "subject": "fund|client|general",
  "intent": "describe|advise|state_fact",
  "confidence": 0-100,
  "reasoning": "explanation",
  "evidence": ["supporting quotes"],
  "is_fund_description": true/false,
  "is_client_advice": true/false
}}"""
    
    def analyze_context(self, text: str, check_type: str = "general") -> ContextAnalysis:
        """
        Analyze text context using AI
        
        Args:
            text: Text to analyze
            check_type: Type of check being performed
            
        Returns:
            ContextAnalysis with subject, intent, confidence, and reasoning
        """
        try:
            # Try AI analysis first
            if check_type == "investment_advice":
                prompt = self.investment_advice_prompt.format(text=text)
            else:
                prompt = self.context_analysis_prompt.format(
                    text=text,
                    check_type=check_type
                )
            
            system_message = "You are a financial compliance expert. Analyze text for semantic meaning and intent. Return only valid JSON with no additional text."
            
            response = self.ai_engine.call_with_cache(prompt, system_message)
            
            if response and response.parsed_json:
                result = response.parsed_json
                
                # Determine is_fund_description and is_client_advice from result or infer from subject/intent
                is_fund_desc = result.get("is_fund_description", False)
                is_client_adv = result.get("is_client_advice", False)
                
                # If not explicitly set, infer from subject and intent
                subject = result.get("subject", "general")
                intent = result.get("intent", "describe")
                
                if not is_fund_desc and subject in ["fund", "strategy"] and intent == "describe":
                    is_fund_desc = True
                
                if not is_client_adv and subject == "client" and intent == "advise":
                    is_client_adv = True
                
                return ContextAnalysis(
                    subject=subject,
                    intent=intent,
                    confidence=result.get("confidence", 50),
                    reasoning=result.get("reasoning", "AI analysis completed"),
                    evidence=result.get("evidence", []),
                    is_fund_description=is_fund_desc,
                    is_client_advice=is_client_adv
                )
            else:
                # AI failed, use fallback
                logger.warning(f"AI analysis failed for {check_type}, using fallback")
                return self._fallback_rule_based_analysis(text, check_type)
                
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return self._fallback_rule_based_analysis(text, check_type)
    
    def is_fund_strategy_description(self, text: str) -> bool:
        """
        Check if text describes fund strategy (ALLOWED)
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is fund strategy description
            
        Examples:
            - "Le fonds investit dans..." → True
            - "Tirer parti du momentum" → True
            - "Vous devriez investir" → False
        """
        try:
            analysis = self.analyze_context(text, "investment_advice")
            
            # High confidence fund description
            if analysis.is_fund_description and analysis.confidence >= 70:
                return True
            
            # Check subject is fund/strategy
            if analysis.subject in ["fund", "strategy"] and analysis.confidence >= 60:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Fund strategy detection error: {e}")
            # Fallback to rule-based
            return self._is_fund_description_fallback(text)
    
    def is_investment_advice(self, text: str) -> bool:
        """
        Check if text advises clients (PROHIBITED)
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is investment advice to clients
            
        Examples:
            - "Vous devriez investir" → True
            - "Nous recommandons d'acheter" → True
            - "Le fonds investit" → False
        """
        try:
            analysis = self.analyze_context(text, "investment_advice")
            
            # High confidence client advice
            if analysis.is_client_advice and analysis.confidence >= 70:
                return True
            
            # Check subject is client and intent is advise
            if analysis.subject == "client" and analysis.intent == "advise" and analysis.confidence >= 60:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Investment advice detection error: {e}")
            # Fallback to rule-based
            return self._is_client_advice_fallback(text)
    
    def extract_subject(self, text: str) -> str:
        """
        Extract WHO is performing the action
        
        Args:
            text: Text to analyze
            
        Returns:
            Subject: "fund", "client", or "general"
        """
        try:
            analysis = self.analyze_context(text, "subject_extraction")
            return analysis.subject
            
        except Exception as e:
            logger.error(f"Subject extraction error: {e}")
            return self._extract_subject_fallback(text)
    
    def _fallback_rule_based_analysis(self, text: str, check_type: str) -> ContextAnalysis:
        """
        Fallback rule-based analysis when AI fails
        
        Args:
            text: Text to analyze
            check_type: Type of check
            
        Returns:
            ContextAnalysis based on rules
        """
        text_lower = text.lower()
        
        # Fund description indicators
        fund_indicators = [
            r'\ble fonds\b', r'\bla stratégie\b', r'\bthe fund\b', r'\bthe strategy\b',
            r'\bfonds investit\b', r'\bstrategy invests\b', r'\bstratégie vise\b',
            r'\bfund aims\b', r'\bfund seeks\b', r'\bfonds cherche\b'
        ]
        
        # Client advice indicators
        advice_indicators = [
            r'\bvous devriez\b', r'\byou should\b', r'\bnous recommandons\b',
            r'\bwe recommend\b', r'\bil faut investir\b', r'\bmust invest\b',
            r'\bbon moment pour\b', r'\bgood time to\b', r'\binvestissez\b'
        ]
        
        # Check for fund indicators
        fund_matches = sum(1 for pattern in fund_indicators if re.search(pattern, text_lower))
        advice_matches = sum(1 for pattern in advice_indicators if re.search(pattern, text_lower))
        
        # Determine subject and intent
        if fund_matches > advice_matches:
            subject = "fund"
            intent = "describe"
            is_fund_description = True
            is_client_advice = False
            confidence = min(60 + (fund_matches * 10), 85)
            reasoning = f"Rule-based: Found {fund_matches} fund description indicators"
        elif advice_matches > 0:
            subject = "client"
            intent = "advise"
            is_fund_description = False
            is_client_advice = True
            confidence = min(60 + (advice_matches * 10), 85)
            reasoning = f"Rule-based: Found {advice_matches} advice indicators"
        else:
            subject = "general"
            intent = "describe"
            is_fund_description = False
            is_client_advice = False
            confidence = 40
            reasoning = "Rule-based: No clear indicators found"
        
        return ContextAnalysis(
            subject=subject,
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            evidence=[],
            is_fund_description=is_fund_description,
            is_client_advice=is_client_advice
        )
    
    def _is_fund_description_fallback(self, text: str) -> bool:
        """Fallback rule-based fund description detection"""
        text_lower = text.lower()
        
        fund_patterns = [
            r'\ble fonds\b', r'\bla stratégie\b', r'\bthe fund\b',
            r'\bfonds investit\b', r'\bstrategy invests\b',
            r'\bstratégie vise\b', r'\bfund aims\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in fund_patterns)
    
    def _is_client_advice_fallback(self, text: str) -> bool:
        """Fallback rule-based client advice detection"""
        text_lower = text.lower()
        
        advice_patterns = [
            r'\bvous devriez\b', r'\byou should\b',
            r'\bnous recommandons\b', r'\bwe recommend\b',
            r'\binvestissez\b', r'\binvest now\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in advice_patterns)
    
    def _extract_subject_fallback(self, text: str) -> str:
        """Fallback rule-based subject extraction"""
        text_lower = text.lower()
        
        # Check for fund/strategy subjects
        if any(word in text_lower for word in ['le fonds', 'la stratégie', 'the fund', 'the strategy']):
            return "fund"
        
        # Check for client subjects
        if any(word in text_lower for word in ['vous', 'you', 'investisseur', 'investor']):
            return "client"
        
        return "general"


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Context Analyzer - Standalone Test")
    print("="*70)
    
    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    if not ai_engine:
        print("\n✗ Failed to initialize AI Engine")
        print("  Using fallback rule-based analysis only")
        print("\nNote: Set GEMINI_API_KEY or TOKENFACTORY_API_KEY in .env for full AI analysis")
    
    # Create analyzer
    analyzer = ContextAnalyzer(ai_engine) if ai_engine else None
    
    if analyzer:
        print("\n✓ Context Analyzer initialized")
        
        # Test cases
        test_cases = [
            {
                "text": "Tirer parti du momentum des marchés américains grâce à une stratégie quantitative",
                "expected": "fund_description",
                "description": "Fund strategy description (French)"
            },
            {
                "text": "Le fonds investit dans des actions américaines",
                "expected": "fund_description",
                "description": "Fund investment description"
            },
            {
                "text": "Vous devriez investir dans ce fonds maintenant",
                "expected": "client_advice",
                "description": "Client investment advice"
            },
            {
                "text": "Nous recommandons d'acheter des actions",
                "expected": "client_advice",
                "description": "Recommendation to client"
            }
        ]
        
        print("\n" + "="*70)
        print("Running Test Cases")
        print("="*70)
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[Test {i}] {test['description']}")
            print(f"Text: \"{test['text']}\"")
            
            # Analyze context
            analysis = analyzer.analyze_context(test['text'], "investment_advice")
            
            print(f"\nResults:")
            print(f"  Subject: {analysis.subject}")
            print(f"  Intent: {analysis.intent}")
            print(f"  Fund Description: {analysis.is_fund_description}")
            print(f"  Client Advice: {analysis.is_client_advice}")
            print(f"  Confidence: {analysis.confidence}%")
            print(f"  Reasoning: {analysis.reasoning}")
            
            # Check if result matches expected
            if test['expected'] == 'fund_description':
                result = "✓ PASS" if analysis.is_fund_description else "✗ FAIL"
            else:
                result = "✓ PASS" if analysis.is_client_advice else "✗ FAIL"
            
            print(f"\n  {result}")
        
        print("\n" + "="*70)
        print("Test Complete")
        print("="*70)
    else:
        print("\n✗ Cannot run tests without AI Engine")
        print("  Set API keys in .env file to enable testing")
