#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Securities Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Securities Checking Tools for Multi-Agent Compliance System

These tools handle securities/values compliance checks:
- Prohibited investment advice phrases (context-aware)
- Repeated external company mentions (whitelist-aware)
- Investment advice detection
- Intent classification

Requirements: 2.1, 2.3, 7.2, 7.5
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Optional, List, Set
from collections import Counter
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_all_text_from_doc(doc: dict) -> str:
    """Extract all text from document"""
    all_text = []

    if 'page_de_garde' in doc:
        all_text.append(json.dumps(doc['page_de_garde']))

    if 'slide_2' in doc:
        all_text.append(json.dumps(doc['slide_2']))

    if 'pages_suivantes' in doc:
        for page in doc['pages_suivantes']:
            all_text.append(json.dumps(page))

    if 'page_de_fin' in doc:
        all_text.append(json.dumps(doc['page_de_fin']))

    return '\n'.join(all_text)


# ============================================================================
# PROHIBITED PHRASES TOOL (CONTEXT-AWARE)
# ============================================================================

@tool
def check_prohibited_phrases(document: dict, rule: dict, ai_engine) -> List[dict]:
    """
    AI context-aware check for prohibited investment advice phrases.

    Eliminates false positives by distinguishing:
    - Fund strategy descriptions (ALLOWED): "Le fonds investit dans...", "Tirer parti du momentum"
    - Investment advice to clients (PROHIBITED): "Vous devriez investir", "Nous recommandons"

    Uses ContextAnalyzer and IntentClassifier to understand:
    - WHO is performing the action (fund vs client)
    - WHAT is the intent (describe vs advise)

    Args:
        document: Document dictionary
        rule: Rule dictionary with prohibited phrases
        ai_engine: AI engine for context analysis

    Returns:
        List of violations (empty if no violations found)

    Requirements: 2.1, 2.3, 7.2, 7.5
    """
    violations = []

    try:
        # Import AI components
        from context_analyzer import ContextAnalyzer
        from intent_classifier import IntentClassifier

        if not ai_engine:
            logger.warning("AI engine not available, skipping AI context-aware check")
            return violations

        # Initialize components
        context_analyzer = ContextAnalyzer(ai_engine)
        intent_classifier = IntentClassifier(ai_engine)

        # Extract all text from document
        doc_text = extract_all_text_from_doc(document)

        if not doc_text or len(doc_text) < 50:
            return violations

        # Get rule details
        rule_id = rule.get('rule_id', 'VAL_001')
        severity = rule.get('severity', 'CRITICAL').upper()
        prohibited_phrases = rule.get('prohibited_phrases', [])

        # Quick check: if no prohibited phrases in text, skip AI analysis
        text_lower = doc_text.lower()
        has_potential_violation = False

        for phrase in prohibited_phrases[:10]:  # Check first 10 phrases
            if phrase.lower() in text_lower:
                has_potential_violation = True
                break

        if not has_potential_violation:
            # No prohibited phrases found, no violation
            return violations

        # Analyze context with AI
        logger.info(f"Analyzing context for {rule_id} with AI...")

        # Use ContextAnalyzer to understand the text
        context_analysis = context_analyzer.analyze_context(doc_text, "investment_advice")

        # Use IntentClassifier to determine intent
        intent_classification = intent_classifier.classify_intent(doc_text)

        # Decision logic: Only flag if BOTH conditions are met:
        # 1. Intent is ADVICE (not DESCRIPTION)
        # 2. Subject is CLIENT (not FUND)

        is_client_advice = (
            intent_classification.intent_type == "ADVICE" and
            context_analysis.subject == "client"
        )

        # Alternative check using helper methods
        is_advice_alt = intent_classifier.is_client_advice(doc_text)
        is_fund_desc = context_analyzer.is_fund_strategy_description(doc_text)

        # Combine results with confidence scoring
        combined_confidence = min(
            context_analysis.confidence,
            intent_classification.confidence
        )

        # Only flag if high confidence client advice
        if (is_client_advice or is_advice_alt) and not is_fund_desc and combined_confidence >= 70:
            # Extract evidence of prohibited phrases
            evidence_phrases = []
            for phrase in prohibited_phrases[:5]:
                if phrase.lower() in text_lower:
                    # Find context around phrase
                    idx = text_lower.find(phrase.lower())
                    if idx != -1:
                        start = max(0, idx - 50)
                        end = min(len(doc_text), idx + len(phrase) + 50)
                        context = doc_text[start:end].strip()
                        evidence_phrases.append(f'"{phrase}" in context: ...{context}...')

            violations.append({
                'type': 'SECURITIES/VALUES',
                'severity': severity,
                'slide': 'Multiple locations',
                'location': 'Document-wide',
                'rule': f"{rule_id}: {rule.get('rule_text', 'No investment recommendations')}",
                'message': f"Investment advice to clients detected (MAR violation)",
                'evidence': (
                    f"AI Analysis:\n"
                    f"  - Intent: {intent_classification.intent_type} (confidence: {intent_classification.confidence}%)\n"
                    f"  - Subject: {context_analysis.subject} (confidence: {context_analysis.confidence}%)\n"
                    f"  - Context Reasoning: {context_analysis.reasoning}\n"
                    f"  - Intent Reasoning: {intent_classification.reasoning}\n"
                    f"  - Evidence: {', '.join(evidence_phrases[:3]) if evidence_phrases else 'See prohibited phrases'}\n"
                    f"  - Method: AI_CONTEXT_AWARE"
                ),
                'confidence': combined_confidence,
                'method': 'AI_CONTEXT_AWARE'
            })

            logger.info(f"Violation detected: Client advice (confidence: {combined_confidence}%)")
        else:
            # Not a violation - likely fund description
            logger.info(f"No violation: Fund description detected")
            logger.info(f"  - Intent: {intent_classification.intent_type}")
            logger.info(f"  - Subject: {context_analysis.subject}")
            logger.info(f"  - Is fund description: {is_fund_desc}")
            logger.info(f"  - Confidence: {combined_confidence}%")

    except Exception as e:
        logger.error(f"AI context-aware check failed: {e}")
        import traceback
        traceback.print_exc()

    return violations


# ============================================================================
# REPEATED SECURITIES TOOL (WHITELIST-AWARE)
# ============================================================================

@tool
def check_repeated_securities(document: dict, whitelist: Set[str], ai_engine) -> List[dict]:
    """
    AI-enhanced check for repeated external company mentions with whitelist filtering.

    Eliminates false positives by whitelisting:
    - Fund name components (e.g., "ODDO", "BHF")
    - Strategy terminology (e.g., "momentum", "quantitative")
    - Regulatory terms (e.g., "SRI", "SRRI", "SFDR")
    - Benchmark names (e.g., "S&P", "MSCI")
    - Generic financial terms (e.g., "fund", "investment")

    Only flags external company names mentioned 3+ times after whitelist filtering.
    Uses SemanticValidator to verify if terms are actually company names.

    Args:
        document: Document dictionary
        whitelist: Set of whitelisted terms
        ai_engine: AI engine for semantic validation

    Returns:
        List of violations (empty if no violations found)

    Requirements: 2.1, 2.3, 7.2, 7.5

    Test cases:
        - "ODDO BHF" (31 mentions) → should NOT flag (fund name)
        - "momentum" (2 mentions) → should NOT flag (strategy term)
        - "SRI" (2 mentions) → should NOT flag (regulatory term)
        - "Apple" (5 mentions) → should flag (external company, 3+ mentions)
    """
    violations = []

    try:
        from whitelist_manager import WhitelistManager
        from semantic_validator import SemanticValidator

        # Initialize components
        whitelist_mgr = WhitelistManager()

        logger.info(f"Using whitelist with {len(whitelist)} terms")

        # Extract all text from document
        doc_text = extract_all_text_from_doc(document)

        if not doc_text or len(doc_text) < 50:
            return violations

        # Extract capitalized words as potential company names
        # Pattern: Words that start with capital letter (2+ chars)
        capitalized_words = re.findall(r'\b[A-Z][A-Za-z]+\b', doc_text)

        # Also extract acronyms (2-5 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', doc_text)

        # Combine and count
        all_terms = capitalized_words + acronyms
        term_counts = Counter(term.lower() for term in all_terms)

        logger.info(f"Found {len(term_counts)} unique capitalized terms/acronyms")

        # Filter out whitelisted terms BEFORE counting threshold
        non_whitelisted_terms = {}
        for term, count in term_counts.items():
            if term not in whitelist and not whitelist_mgr.is_whitelisted(term):
                non_whitelisted_terms[term] = count
            else:
                reason = whitelist_mgr.get_whitelist_reason(term) if whitelist_mgr.is_whitelisted(term) else "in whitelist"
                logger.info(f"Skipping whitelisted term: {term} ({count} mentions) - {reason}")

        logger.info(f"After whitelist filtering: {len(non_whitelisted_terms)} terms remain")

        # Only check terms mentioned 3+ times
        repeated_terms = {term: count for term, count in non_whitelisted_terms.items() if count >= 3}

        if not repeated_terms:
            logger.info("No repeated non-whitelisted terms found (threshold: 3+ mentions)")
            return violations

        logger.info(f"Found {len(repeated_terms)} terms with 3+ mentions: {list(repeated_terms.keys())}")

        if not ai_engine:
            logger.warning("AI engine not available, using rule-based validation only")
            # Fallback: flag all repeated non-whitelisted terms
            for term, count in repeated_terms.items():
                violations.append({
                    'type': 'SECURITIES/VALUES',
                    'severity': 'MAJOR',
                    'slide': 'Multiple slides',
                    'location': 'Document-wide',
                    'rule': 'VAL_005: Repeated external company mentions',
                    'message': f'External company "{term}" mentioned {count} times',
                    'evidence': f'Term appears {count} times in document (rule-based detection, no AI verification)',
                    'confidence': 60,
                    'method': 'RULE_BASED'
                })
            return violations

        # Use SemanticValidator to verify each term
        semantic_validator = SemanticValidator(ai_engine)

        # Limit to top 10 most mentioned terms for performance
        top_repeated = sorted(repeated_terms.items(), key=lambda x: x[1], reverse=True)[:10]

        for term, count in top_repeated:
            logger.info(f"Validating term: {term} ({count} mentions)")

            # Use SemanticValidator to check if it's actually an external company
            result = semantic_validator.validate_securities_mention(
                text=doc_text,
                term=term,
                whitelist=whitelist,
                mention_count=count
            )

            if result.is_violation:
                violations.append({
                    'type': 'SECURITIES/VALUES',
                    'severity': 'MAJOR',
                    'slide': 'Multiple slides',
                    'location': 'Document-wide',
                    'rule': 'VAL_005: Repeated external company mentions',
                    'message': f'External company "{term}" mentioned {count} times',
                    'evidence': f'{result.reasoning}. {result.evidence[0] if result.evidence else ""}',
                    'confidence': result.confidence,
                    'method': result.method
                })
                logger.info(f"  → VIOLATION: {result.reasoning} (confidence: {result.confidence}%)")
            else:
                logger.info(f"  → OK: {result.reasoning} (confidence: {result.confidence}%)")

        return violations

    except Exception as e:
        logger.error(f"Error in check_repeated_securities: {e}")
        import traceback
        traceback.print_exc()
        return violations


# ============================================================================
# INVESTMENT ADVICE TOOL
# ============================================================================

@tool
def check_investment_advice(document: dict, ai_engine) -> List[dict]:
    """
    Check for investment advice to clients using AI intent classification.

    Detects prohibited investment advice such as:
    - "Vous devriez investir" (You should invest)
    - "Nous recommandons d'acheter" (We recommend buying)
    - "Bon moment pour investir" (Good time to invest)
    - "Il faut acheter maintenant" (Must buy now)

    Distinguishes from allowed fund descriptions:
    - "Le fonds investit dans..." (The fund invests in...)
    - "La stratégie vise à..." (The strategy aims to...)

    Args:
        document: Document dictionary
        ai_engine: AI engine for intent classification

    Returns:
        List of violations (empty if no violations found)

    Requirements: 2.1, 2.3, 7.2, 7.5
    """
    violations = []

    try:
        from intent_classifier import IntentClassifier
        from context_analyzer import ContextAnalyzer

        if not ai_engine:
            logger.warning("AI engine not available, using rule-based fallback")
            return _check_investment_advice_fallback(document)

        # Initialize components
        intent_classifier = IntentClassifier(ai_engine)
        context_analyzer = ContextAnalyzer(ai_engine)

        # Extract all text from document
        doc_text = extract_all_text_from_doc(document)

        if not doc_text or len(doc_text) < 50:
            return violations

        logger.info("Checking for investment advice using AI...")

        # Check if text contains client advice
        is_advice = intent_classifier.is_client_advice(doc_text)

        if is_advice:
            # Get detailed classification
            classification = intent_classifier.classify_intent(doc_text)
            context_analysis = context_analyzer.analyze_context(doc_text, "investment_advice")

            # Extract evidence
            advice_patterns = [
                r'vous devriez', r'you should', r'nous recommandons',
                r'we recommend', r'investissez', r'invest now',
                r'achetez', r'buy', r'il faut', r'must',
                r'bon moment', r'good time'
            ]

            evidence_found = []
            text_lower = doc_text.lower()
            for pattern in advice_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in list(matches)[:3]:  # Limit to 3 examples
                    start = max(0, match.start() - 50)
                    end = min(len(doc_text), match.end() + 50)
                    context = doc_text[start:end].strip()
                    evidence_found.append(f"...{context}...")

            violations.append({
                'type': 'SECURITIES/VALUES',
                'severity': 'CRITICAL',
                'slide': 'Multiple locations',
                'location': 'Document-wide',
                'rule': 'VAL_001: No investment advice to clients',
                'message': 'Investment advice to clients detected',
                'evidence': (
                    f"AI Analysis:\n"
                    f"  - Intent: {classification.intent_type} (confidence: {classification.confidence}%)\n"
                    f"  - Subject: {context_analysis.subject}\n"
                    f"  - Reasoning: {classification.reasoning}\n"
                    f"  - Examples: {'; '.join(evidence_found[:2]) if evidence_found else 'See document'}\n"
                    f"  - Method: AI_INTENT_CLASSIFICATION"
                ),
                'confidence': classification.confidence,
                'method': 'AI_INTENT_CLASSIFICATION'
            })

            logger.info(f"Investment advice detected (confidence: {classification.confidence}%)")
        else:
            logger.info("No investment advice detected")

    except Exception as e:
        logger.error(f"Error checking investment advice: {e}")
        import traceback
        traceback.print_exc()

    return violations


def _check_investment_advice_fallback(document: dict) -> List[dict]:
    """
    Fallback rule-based investment advice detection when AI is unavailable.

    Args:
        document: Document dictionary

    Returns:
        List of violations
    """
    violations = []

    try:
        doc_text = extract_all_text_from_doc(document)
        text_lower = doc_text.lower()

        # Rule-based advice patterns
        advice_patterns = [
            r'\bvous devriez\b', r'\byou should\b', r'\bnous recommandons\b',
            r'\bwe recommend\b', r'\binvestissez\b', r'\binvest now\b',
            r'\bachetez\b', r'\bbuy\b', r'\bil faut investir\b', r'\bmust invest\b',
            r'\bbon moment pour\b', r'\bgood time to\b'
        ]

        found_patterns = []
        for pattern in advice_patterns:
            if re.search(pattern, text_lower):
                found_patterns.append(pattern.replace(r'\b', '').replace('\\', ''))

        if found_patterns:
            violations.append({
                'type': 'SECURITIES/VALUES',
                'severity': 'CRITICAL',
                'slide': 'Multiple locations',
                'location': 'Document-wide',
                'rule': 'VAL_001: No investment advice to clients',
                'message': 'Potential investment advice detected',
                'evidence': f'Rule-based detection found patterns: {", ".join(found_patterns[:3])}',
                'confidence': 70,
                'method': 'RULE_BASED_FALLBACK'
            })

            logger.info(f"Investment advice detected (rule-based, {len(found_patterns)} patterns)")

    except Exception as e:
        logger.error(f"Error in fallback investment advice check: {e}")

    return violations


# ============================================================================
# INTENT CLASSIFICATION TOOL
# ============================================================================

@tool
def classify_text_intent(text: str, ai_engine) -> dict:
    """
    Classify text intent using AI to distinguish between different types of content.

    Classification types:
    - ADVICE: Tells clients what they should do (PROHIBITED)
    - DESCRIPTION: Describes what the fund does (ALLOWED)
    - FACT: States objective information (ALLOWED)
    - EXAMPLE: Illustrative scenario (ALLOWED)

    Args:
        text: Text to classify
        ai_engine: AI engine for classification

    Returns:
        Dictionary with intent_type, confidence, subject, reasoning, and evidence

    Requirements: 2.3, 7.2, 7.5

    Examples:
        >>> classify_text_intent("Le fonds investit dans des actions", ai_engine)
        {'intent_type': 'DESCRIPTION', 'confidence': 95, 'subject': 'fund', ...}

        >>> classify_text_intent("Vous devriez investir maintenant", ai_engine)
        {'intent_type': 'ADVICE', 'confidence': 95, 'subject': 'client', ...}
    """
    try:
        from intent_classifier import IntentClassifier

        if not ai_engine:
            logger.warning("AI engine not available, using rule-based fallback")
            return _classify_intent_fallback(text)

        # Initialize classifier
        intent_classifier = IntentClassifier(ai_engine)

        # Classify intent
        classification = intent_classifier.classify_intent(text)

        return {
            'intent_type': classification.intent_type,
            'confidence': classification.confidence,
            'subject': classification.subject,
            'reasoning': classification.reasoning,
            'evidence': classification.evidence
        }

    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return {
            'intent_type': 'UNKNOWN',
            'confidence': 0,
            'subject': 'general',
            'reasoning': f'Error: {str(e)}',
            'evidence': ''
        }


def _classify_intent_fallback(text: str) -> dict:
    """
    Fallback rule-based intent classification.

    Args:
        text: Text to classify

    Returns:
        Dictionary with classification results
    """
    text_lower = text.lower()

    # ADVICE indicators
    advice_patterns = [
        r'\bvous devriez\b', r'\byou should\b', r'\bnous recommandons\b',
        r'\bwe recommend\b', r'\binvestissez\b', r'\binvest now\b'
    ]

    # DESCRIPTION indicators
    description_patterns = [
        r'\ble fonds\b', r'\bthe fund\b', r'\bla stratégie\b',
        r'\bthe strategy\b', r'\bfonds investit\b', r'\bfund invests\b'
    ]

    # FACT indicators
    fact_patterns = [
        r'\ba généré\b', r'\bgenerated\b', r'\best de\b', r'\bis\b',
        r'\d+%', r'\ben \d{4}\b'
    ]

    # EXAMPLE indicators
    example_patterns = [
        r'\bpar exemple\b', r'\bfor example\b', r'\bsi vous aviez\b',
        r'\bif you had\b', r'\billustration\b'
    ]

    # Count matches
    advice_matches = sum(1 for p in advice_patterns if re.search(p, text_lower))
    description_matches = sum(1 for p in description_patterns if re.search(p, text_lower))
    fact_matches = sum(1 for p in fact_patterns if re.search(p, text_lower))
    example_matches = sum(1 for p in example_patterns if re.search(p, text_lower))

    # Determine intent
    max_matches = max(advice_matches, description_matches, fact_matches, example_matches)

    if max_matches == 0:
        return {
            'intent_type': 'DESCRIPTION',
            'confidence': 40,
            'subject': 'general',
            'reasoning': 'Rule-based: No clear indicators, defaulting to DESCRIPTION',
            'evidence': ''
        }

    if advice_matches == max_matches:
        return {
            'intent_type': 'ADVICE',
            'confidence': min(60 + (advice_matches * 10), 85),
            'subject': 'client',
            'reasoning': f'Rule-based: Found {advice_matches} advice indicators',
            'evidence': 'Advice patterns detected'
        }
    elif description_matches == max_matches:
        return {
            'intent_type': 'DESCRIPTION',
            'confidence': min(60 + (description_matches * 10), 85),
            'subject': 'fund',
            'reasoning': f'Rule-based: Found {description_matches} description indicators',
            'evidence': 'Description patterns detected'
        }
    elif fact_matches == max_matches:
        return {
            'intent_type': 'FACT',
            'confidence': min(60 + (fact_matches * 10), 85),
            'subject': 'general',
            'reasoning': f'Rule-based: Found {fact_matches} fact indicators',
            'evidence': 'Fact patterns detected'
        }
    else:
        return {
            'intent_type': 'EXAMPLE',
            'confidence': min(60 + (example_matches * 10), 85),
            'subject': 'general',
            'reasoning': f'Rule-based: Found {example_matches} example indicators',
            'evidence': 'Example patterns detected'
        }


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# List of all securities checking tools for easy import
SECURITIES_TOOLS = [
    check_prohibited_phrases,
    check_repeated_securities,
    check_investment_advice,
    classify_text_intent
]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test with example document
    test_doc = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'content': 'Tirer parti du momentum des marchés américains'
        },
        'slide_2': {
            'content': 'Le fonds investit dans des actions américaines à forte capitalisation'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'content': 'La stratégie quantitative vise à identifier les tendances'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    test_whitelist = {'oddo', 'bhf', 'momentum', 'quantitative', 'trend'}

    test_rule = {
        'rule_id': 'VAL_001',
        'severity': 'CRITICAL',
        'rule_text': 'No investment recommendations',
        'prohibited_phrases': [
            'vous devriez investir',
            'nous recommandons',
            'bon moment pour investir'
        ]
    }

    logger.info("=" * 70)
    logger.info("SECURITIES CHECKING TOOLS TEST")
    logger.info("=" * 70)

    # Initialize AI engine
    try:
        from ai_engine import create_ai_engine_from_env
        ai_engine = create_ai_engine_from_env()
        if ai_engine:
            logger.info("\n✓ AI Engine initialized")
        else:
            logger.info("\n⚠️  AI Engine not available - using rule-based fallback")
    except Exception as e:
        logger.info(f"\n⚠️  Could not initialize AI Engine: {e}")
        ai_engine = None

    # Test prohibited phrases
    logger.info("\n1. Check Prohibited Phrases (Context-Aware):")
    try:
        result = check_prohibited_phrases.func(
            document=test_doc,
            rule=test_rule,
            ai_engine=ai_engine
        )
        if result:
            logger.info(f"   ❌ VIOLATIONS: {len(result)}")
            for v in result:
                logger.info(f"      - {v['message']}")
        else:
            logger.info("   ✓ PASS: No prohibited phrases detected")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test repeated securities
    logger.info("\n2. Check Repeated Securities (Whitelist-Aware):")
    try:
        result = check_repeated_securities.func(
            document=test_doc,
            whitelist=test_whitelist,
            ai_engine=ai_engine
        )
        if result:
            logger.info(f"   ❌ VIOLATIONS: {len(result)}")
            for v in result:
                logger.info(f"      - {v['message']}")
        else:
            logger.info("   ✓ PASS: No repeated external companies detected")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test investment advice
    logger.info("\n3. Check Investment Advice:")
    try:
        result = check_investment_advice.func(
            document=test_doc,
            ai_engine=ai_engine
        )
        if result:
            logger.info(f"   ❌ VIOLATIONS: {len(result)}")
            for v in result:
                logger.info(f"      - {v['message']}")
        else:
            logger.info("   ✓ PASS: No investment advice detected")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test intent classification
    logger.info("\n4. Classify Text Intent:")
    test_texts = [
        "Le fonds investit dans des actions américaines",
        "Vous devriez investir maintenant",
        "Le fonds a généré 15% en 2023"
    ]

    for text in test_texts:
        try:
            result = classify_text_intent.func(text=text, ai_engine=ai_engine)
            logger.info(f"\n   Text: \"{text}\"")
            logger.info(f"   Intent: {result['intent_type']} (confidence: {result['confidence']}%)")
            logger.info(f"   Subject: {result['subject']}")
        except Exception as e:
            logger.info(f"   ⚠️  ERROR: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("All securities checking tools tested!")
    logger.info("=" * 70)
