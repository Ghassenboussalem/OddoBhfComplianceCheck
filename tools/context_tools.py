#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Context Analysis Tools for Multi-Agent Compliance System

This module provides tools for analyzing text context and intent to distinguish
between fund strategy descriptions (ALLOWED) and client investment advice (PROHIBITED).

Key Features:
- Analyze semantic context using AI
- Classify text intent (ADVICE, DESCRIPTION, FACT, EXAMPLE)
- Extract subject (WHO performs the action)
- Detect fund strategy descriptions
- Detect client investment advice

Requirements: 2.3, 7.2, 7.3, 7.5, 9.1, 9.2
"""

import logging
import re
import sys
import os
from typing import Dict, Optional, Set

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tool_registry import tool, ToolCategory
from data_models import ContextAnalysis, IntentClassification

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# CONTEXT ANALYSIS TOOLS
# ============================================================================

@tool(
    name="analyze_context",
    category=ToolCategory.ANALYSIS,
    description="Analyze text context using AI to understand semantic meaning and intent",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def analyze_context(text: str, check_type: str, ai_engine) -> ContextAnalysis:
    """
    Analyze text context using AI to understand semantic meaning and intent

    Distinguishes:
    - Fund strategy descriptions (ALLOWED): "Le fonds investit dans..."
    - Client advice (PROHIBITED): "Vous devriez investir..."

    Args:
        text: Text to analyze
        check_type: Type of check being performed (e.g., "investment_advice", "general")
        ai_engine: AIEngine instance for LLM calls

    Returns:
        ContextAnalysis with subject, intent, confidence, and reasoning

    Examples:
        >>> analyze_context("Le fonds investit dans des actions", "general", ai_engine)
        ContextAnalysis(subject="fund", intent="describe", is_fund_description=True, ...)

        >>> analyze_context("Vous devriez investir maintenant", "investment_advice", ai_engine)
        ContextAnalysis(subject="client", intent="advise", is_client_advice=True, ...)
    """
    try:
        # Build AI prompt based on check type
        if check_type == "investment_advice":
            prompt = _build_investment_advice_prompt(text)
        else:
            prompt = _build_context_analysis_prompt(text, check_type)

        system_message = "You are a financial compliance expert. Analyze text for semantic meaning and intent. Return only valid JSON with no additional text."

        # Call AI engine
        response = ai_engine.call_with_cache(prompt, system_message)

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
            return _fallback_rule_based_analysis(text, check_type)

    except Exception as e:
        logger.error(f"Context analysis error: {e}")
        return _fallback_rule_based_analysis(text, check_type)


@tool(
    name="classify_intent",
    category=ToolCategory.ANALYSIS,
    description="Classify text intent as ADVICE, DESCRIPTION, FACT, or EXAMPLE",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def classify_intent(text: str, ai_engine) -> IntentClassification:
    """
    Classify text intent using AI

    Intent Types:
    - ADVICE: Tells clients what they should do (PROHIBITED)
    - DESCRIPTION: Describes what the fund does (ALLOWED)
    - FACT: States objective information (ALLOWED)
    - EXAMPLE: Illustrative scenario (ALLOWED)

    Args:
        text: Text to classify
        ai_engine: AIEngine instance for LLM calls

    Returns:
        IntentClassification with intent_type, confidence, subject, reasoning, and evidence

    Examples:
        >>> classify_intent("Le fonds investit dans des actions", ai_engine)
        IntentClassification(intent_type="DESCRIPTION", confidence=95, subject="fund", ...)

        >>> classify_intent("Vous devriez investir maintenant", ai_engine)
        IntentClassification(intent_type="ADVICE", confidence=95, subject="client", ...)
    """
    try:
        # Build AI prompt for intent classification
        prompt = _build_intent_classification_prompt(text)
        system_message = "You are a financial compliance expert. Classify text intent accurately. Return only valid JSON with no additional text."

        # Call AI engine
        response = ai_engine.call_with_cache(prompt, system_message)

        if response and response.parsed_json:
            result = response.parsed_json

            return IntentClassification(
                intent_type=result.get("intent_type", "DESCRIPTION"),
                confidence=result.get("confidence", 50),
                subject=result.get("subject", "general"),
                reasoning=result.get("reasoning", "AI classification completed"),
                evidence=result.get("evidence", "")
            )
        else:
            # AI failed, use fallback
            logger.warning("AI intent classification failed, using fallback")
            return _fallback_rule_based_classification(text)

    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        return _fallback_rule_based_classification(text)


@tool(
    name="extract_subject",
    category=ToolCategory.ANALYSIS,
    description="Extract WHO is performing the action (fund, client, or general)",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def extract_subject(text: str, ai_engine) -> str:
    """
    Extract WHO is performing the action

    Subjects:
    - "fund": The fund or strategy performs the action
    - "client": The client or investor performs the action
    - "general": Neutral or unclear subject

    Args:
        text: Text to analyze
        ai_engine: AIEngine instance for LLM calls

    Returns:
        Subject string: "fund", "client", or "general"

    Examples:
        >>> extract_subject("Le fonds investit dans des actions", ai_engine)
        "fund"

        >>> extract_subject("Vous devriez investir", ai_engine)
        "client"
    """
    try:
        # Use analyze_context to extract subject
        analysis_result = analyze_context(text, "subject_extraction", ai_engine)
        # Handle ToolResult wrapper
        analysis = analysis_result.result if hasattr(analysis_result, 'result') else analysis_result
        return analysis.subject

    except Exception as e:
        logger.error(f"Subject extraction error: {e}")
        return _extract_subject_fallback(text)


@tool(
    name="is_fund_strategy_description",
    category=ToolCategory.ANALYSIS,
    description="Check if text describes fund strategy (ALLOWED)",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def is_fund_strategy_description(text: str, ai_engine) -> bool:
    """
    Check if text describes fund strategy (ALLOWED)

    Fund strategy descriptions include:
    - "Le fonds investit dans..." (The fund invests in...)
    - "La stratégie vise à..." (The strategy aims to...)
    - "Tirer parti du momentum" (Take advantage of momentum)

    Args:
        text: Text to analyze
        ai_engine: AIEngine instance for LLM calls

    Returns:
        True if text is fund strategy description, False otherwise

    Examples:
        >>> is_fund_strategy_description("Le fonds investit dans des actions", ai_engine)
        True

        >>> is_fund_strategy_description("Vous devriez investir", ai_engine)
        False
    """
    try:
        analysis_result = analyze_context(text, "investment_advice", ai_engine)
        # Handle ToolResult wrapper
        analysis = analysis_result.result if hasattr(analysis_result, 'result') else analysis_result

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
        return _is_fund_description_fallback(text)


@tool(
    name="is_investment_advice",
    category=ToolCategory.ANALYSIS,
    description="Check if text advises clients (PROHIBITED)",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def is_investment_advice(text: str, ai_engine) -> bool:
    """
    Check if text advises clients (PROHIBITED)

    Investment advice includes:
    - "Vous devriez investir" (You should invest)
    - "Nous recommandons d'acheter" (We recommend buying)
    - "Bon moment pour investir" (Good time to invest)

    Args:
        text: Text to analyze
        ai_engine: AIEngine instance for LLM calls

    Returns:
        True if text is investment advice to clients, False otherwise

    Examples:
        >>> is_investment_advice("Vous devriez investir maintenant", ai_engine)
        True

        >>> is_investment_advice("Le fonds investit dans des actions", ai_engine)
        False
    """
    try:
        analysis_result = analyze_context(text, "investment_advice", ai_engine)
        # Handle ToolResult wrapper
        analysis = analysis_result.result if hasattr(analysis_result, 'result') else analysis_result

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
        return _is_client_advice_fallback(text)


# ============================================================================
# PRIVATE HELPER FUNCTIONS - AI PROMPTS
# ============================================================================

def _build_investment_advice_prompt(text: str) -> str:
    """Build AI prompt for investment advice analysis"""
    return f"""Analyze this text and determine if it contains investment advice to clients or describes the fund's strategy.

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


def _build_context_analysis_prompt(text: str, check_type: str) -> str:
    """Build AI prompt for generic context analysis"""
    return f"""Analyze the context and meaning of this text.

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


def _build_intent_classification_prompt(text: str) -> str:
    """Build AI prompt for intent classification"""
    return f"""Analyze this text and classify its intent.

TEXT: {text}

CLASSIFICATION TYPES:
1. ADVICE: Tells clients what they should do
   - "Vous devriez investir" (You should invest)
   - "Nous recommandons d'acheter" (We recommend buying)
   - "Bon moment pour investir" (Good time to invest)
   - "Il faut acheter maintenant" (Must buy now)
   - Subject: CLIENT/INVESTOR receives advice

2. DESCRIPTION: Describes what the fund does
   - "Le fonds investit dans..." (The fund invests in...)
   - "La stratégie vise à..." (The strategy aims to...)
   - "Tirer parti du momentum" (Take advantage of momentum)
   - "Le portefeuille est composé de..." (The portfolio consists of...)
   - Subject: FUND/STRATEGY performs action

3. FACT: States objective information
   - "Le fonds a généré 5% en 2023" (The fund generated 5% in 2023)
   - "Le SRRI est de 4/7" (The SRRI is 4/7)
   - "Le fonds est domicilié au Luxembourg" (The fund is domiciled in Luxembourg)
   - Subject: Neutral statement of fact

4. EXAMPLE: Illustrative scenario
   - "Par exemple, un investissement de 1000€..." (For example, an investment of 1000€...)
   - "Si vous aviez investi 1000€ en 2020..." (If you had invested 1000€ in 2020...)
   - "Illustration: avec un rendement de 5%..." (Illustration: with a return of 5%...)
   - Subject: Hypothetical scenario

TASK:
1. Identify the intent type (ADVICE, DESCRIPTION, FACT, or EXAMPLE)
2. Determine WHO performs the action (fund, client, general)
3. Extract key phrases that support your classification
4. Provide confidence score (0-100)
5. Explain your reasoning clearly

IMPORTANT:
- ADVICE is PROHIBITED in marketing materials
- DESCRIPTION is ALLOWED in marketing materials
- Focus on WHO is the subject and WHAT they are doing

Respond with JSON only, no additional text:
{{
  "intent_type": "ADVICE|DESCRIPTION|FACT|EXAMPLE",
  "confidence": 0-100,
  "subject": "fund|client|general",
  "reasoning": "detailed explanation of why this classification was chosen",
  "evidence": "key phrases that support this classification"
}}"""


# ============================================================================
# PRIVATE HELPER FUNCTIONS - FALLBACK RULE-BASED ANALYSIS
# ============================================================================

def _fallback_rule_based_analysis(text: str, check_type: str) -> ContextAnalysis:
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


def _fallback_rule_based_classification(text: str) -> IntentClassification:
    """
    Fallback rule-based intent classification when AI fails

    Args:
        text: Text to classify

    Returns:
        IntentClassification based on rules
    """
    text_lower = text.lower()

    # ADVICE indicators
    advice_patterns = [
        r'\bvous devriez\b', r'\byou should\b', r'\bnous recommandons\b',
        r'\bwe recommend\b', r'\binvestissez\b', r'\binvest now\b',
        r'\bachetez\b', r'\bbuy\b', r'\bil faut\b', r'\bmust\b',
        r'\bbon moment\b', r'\bgood time\b', r'\bmaintenant\b.*\binvest'
    ]

    # DESCRIPTION indicators
    description_patterns = [
        r'\ble fonds investit\b', r'\bthe fund invests\b', r'\bla stratégie\b',
        r'\bthe strategy\b', r'\bvise à\b', r'\baims to\b', r'\bcherche à\b',
        r'\bseeks to\b', r'\btirer parti\b', r'\btake advantage\b',
        r'\bcomposé de\b', r'\bconsists of\b', r'\bportefeuille\b', r'\bportfolio\b'
    ]

    # FACT indicators
    fact_patterns = [
        r'\ba généré\b', r'\bgenerated\b', r'\best de\b', r'\bis\b',
        r'\bdomicilié\b', r'\bdomiciled\b', r'\bsrri\b', r'\brisk\b',
        r'\ben \d{4}\b', r'\bin \d{4}\b', r'\d+%'
    ]

    # EXAMPLE indicators
    example_patterns = [
        r'\bpar exemple\b', r'\bfor example\b', r'\bsi vous aviez\b',
        r'\bif you had\b', r'\billustration\b', r'\bhypothèse\b',
        r'\bhypothetical\b', r'\bscénario\b', r'\bscenario\b'
    ]

    # Count matches for each type
    advice_matches = sum(1 for pattern in advice_patterns if re.search(pattern, text_lower))
    description_matches = sum(1 for pattern in description_patterns if re.search(pattern, text_lower))
    fact_matches = sum(1 for pattern in fact_patterns if re.search(pattern, text_lower))
    example_matches = sum(1 for pattern in example_patterns if re.search(pattern, text_lower))

    # Determine intent type based on highest match count
    max_matches = max(advice_matches, description_matches, fact_matches, example_matches)

    if max_matches == 0:
        # No clear indicators, default to DESCRIPTION with low confidence
        return IntentClassification(
            intent_type="DESCRIPTION",
            confidence=40,
            subject="general",
            reasoning="Rule-based: No clear indicators found, defaulting to DESCRIPTION",
            evidence=""
        )

    if advice_matches == max_matches:
        return IntentClassification(
            intent_type="ADVICE",
            confidence=min(60 + (advice_matches * 10), 85),
            subject="client",
            reasoning=f"Rule-based: Found {advice_matches} advice indicators",
            evidence="Advice patterns detected"
        )
    elif description_matches == max_matches:
        return IntentClassification(
            intent_type="DESCRIPTION",
            confidence=min(60 + (description_matches * 10), 85),
            subject="fund",
            reasoning=f"Rule-based: Found {description_matches} description indicators",
            evidence="Description patterns detected"
        )
    elif fact_matches == max_matches:
        return IntentClassification(
            intent_type="FACT",
            confidence=min(60 + (fact_matches * 10), 85),
            subject="general",
            reasoning=f"Rule-based: Found {fact_matches} fact indicators",
            evidence="Fact patterns detected"
        )
    else:  # example_matches == max_matches
        return IntentClassification(
            intent_type="EXAMPLE",
            confidence=min(60 + (example_matches * 10), 85),
            subject="general",
            reasoning=f"Rule-based: Found {example_matches} example indicators",
            evidence="Example patterns detected"
        )


def _is_fund_description_fallback(text: str) -> bool:
    """Fallback rule-based fund description detection"""
    text_lower = text.lower()

    fund_patterns = [
        r'\ble fonds\b', r'\bla stratégie\b', r'\bthe fund\b',
        r'\bfonds investit\b', r'\bstrategy invests\b',
        r'\bstratégie vise\b', r'\bfund aims\b'
    ]

    return any(re.search(pattern, text_lower) for pattern in fund_patterns)


def _is_client_advice_fallback(text: str) -> bool:
    """Fallback rule-based client advice detection"""
    text_lower = text.lower()

    advice_patterns = [
        r'\bvous devriez\b', r'\byou should\b',
        r'\bnous recommandons\b', r'\bwe recommend\b',
        r'\binvestissez\b', r'\binvest now\b'
    ]

    return any(re.search(pattern, text_lower) for pattern in advice_patterns)


def _extract_subject_fallback(text: str) -> str:
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
    logger.info("="*70)
    logger.info("Context Analysis Tools - Standalone Test")
    logger.info("="*70)

    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()

    if not ai_engine:
        logger.info("\n✗ Failed to initialize AI Engine")
        logger.info("  Using fallback rule-based analysis only")
        logger.info("\nNote: Set GEMINI_API_KEY or TOKENFACTORY_API_KEY in .env for full AI analysis")
    else:
        logger.info("\n✓ AI Engine initialized")

    # Test cases
    test_cases = [
        {
            "text": "Tirer parti du momentum des marchés américains grâce à une stratégie quantitative",
            "expected_subject": "fund",
            "expected_fund_desc": True,
            "expected_advice": False,
            "description": "Fund strategy description (French)"
        },
        {
            "text": "Le fonds investit dans des actions américaines",
            "expected_subject": "fund",
            "expected_fund_desc": True,
            "expected_advice": False,
            "description": "Fund investment description"
        },
        {
            "text": "Vous devriez investir dans ce fonds maintenant",
            "expected_subject": "client",
            "expected_fund_desc": False,
            "expected_advice": True,
            "description": "Client investment advice"
        },
        {
            "text": "Nous recommandons d'acheter des actions",
            "expected_subject": "client",
            "expected_fund_desc": False,
            "expected_advice": True,
            "description": "Recommendation to client"
        }
    ]

    logger.info("\n" + "="*70)
    logger.info("Running Test Cases")
    logger.info("="*70)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n[Test {i}] {test['description']}")
        logger.info(f"Text: \"{test['text']}\"")

        # Test analyze_context
        context_result = analyze_context(test['text'], "investment_advice", ai_engine)
        context_data = context_result.result if hasattr(context_result, 'result') else context_result
        logger.info(f"\nContext Analysis:")
        logger.info(f"  Subject: {context_data.subject}")
        logger.info(f"  Intent: {context_data.intent}")
        logger.info(f"  Fund Description: {context_data.is_fund_description}")
        logger.info(f"  Client Advice: {context_data.is_client_advice}")
        logger.info(f"  Confidence: {context_data.confidence}%")

        # Test classify_intent
        intent_result = classify_intent(test['text'], ai_engine)
        intent_data = intent_result.result if hasattr(intent_result, 'result') else intent_result
        logger.info(f"\nIntent Classification:")
        logger.info(f"  Intent Type: {intent_data.intent_type}")
        logger.info(f"  Confidence: {intent_data.confidence}%")

        # Test extract_subject
        subject_result = extract_subject(test['text'], ai_engine)
        subject_data = subject_result.result if hasattr(subject_result, 'result') else subject_result
        logger.info(f"\nSubject Extraction: {subject_data}")

        # Test is_fund_strategy_description
        fund_desc_result = is_fund_strategy_description(test['text'], ai_engine)
        fund_desc_data = fund_desc_result.result if hasattr(fund_desc_result, 'result') else fund_desc_result
        logger.info(f"Is Fund Description: {fund_desc_data}")

        # Test is_investment_advice
        advice_result = is_investment_advice(test['text'], ai_engine)
        advice_data = advice_result.result if hasattr(advice_result, 'result') else advice_result
        logger.info(f"Is Investment Advice: {advice_data}")

        # Check if results match expected
        subject_match = subject_data == test['expected_subject']
        fund_desc_match = fund_desc_data == test['expected_fund_desc']
        advice_match = advice_data == test['expected_advice']

        all_match = subject_match and fund_desc_match and advice_match

        if all_match:
            logger.info(f"\n  ✓ PASS - All checks match expected results")
            passed += 1
        else:
            logger.info(f"\n  ✗ FAIL - Mismatch detected:")
            if not subject_match:
                logger.info(f"    - Subject: expected {test['expected_subject']}, got {subject_data}")
            if not fund_desc_match:
                logger.info(f"    - Fund Description: expected {test['expected_fund_desc']}, got {fund_desc_data}")
            if not advice_match:
                logger.info(f"    - Investment Advice: expected {test['expected_advice']}, got {advice_data}")
            failed += 1

    logger.info("\n" + "="*70)
    logger.info("Test Summary")
    logger.info("="*70)
    logger.info(f"Total Tests: {len(test_cases)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    logger.info("="*70)
