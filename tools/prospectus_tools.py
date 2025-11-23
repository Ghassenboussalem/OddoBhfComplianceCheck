#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prospectus Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Prospectus Checking Tools for Multi-Agent Compliance System

These tools handle prospectus compliance checks:
- Fund name semantic matching
- Investment strategy consistency
- Benchmark validation
- Investment objective validation

Requirements: 2.1, 7.2, 7.5
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Optional, List
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_all_text_from_doc(doc: dict) -> str:
    """Extract all text from document for analysis"""
    all_text = []

    if 'page_de_garde' in doc:
        all_text.append(json.dumps(doc['page_de_garde'], ensure_ascii=False))

    if 'slide_2' in doc:
        all_text.append(json.dumps(doc['slide_2'], ensure_ascii=False))

    if 'pages_suivantes' in doc:
        for page in doc['pages_suivantes']:
            all_text.append(json.dumps(page, ensure_ascii=False))

    if 'page_de_fin' in doc:
        all_text.append(json.dumps(doc['page_de_fin'], ensure_ascii=False))

    return '\n'.join(all_text)


def call_ai_for_semantic_check(prompt: str, ai_engine=None) -> Optional[dict]:
    """
    Call AI engine for semantic analysis

    Args:
        prompt: The prompt to send to AI
        ai_engine: Optional AI engine instance

    Returns:
        Parsed JSON response or None if AI unavailable
    """
    try:
        if ai_engine:
            response = ai_engine.call_with_cache(prompt)
            if response and not response.error:
                if response.parsed_json:
                    return response.parsed_json
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {'content': response.content}
        return None
    except Exception as e:
        logger.error(f"AI call error: {e}")
        return None


# ============================================================================
# FUND NAME MATCHING TOOL
# ============================================================================

@tool
def check_fund_name_match(document: dict, prospectus_data: dict, ai_engine=None) -> Optional[dict]:
    """
    Check if fund name in document matches prospectus using semantic matching.

    This tool performs semantic matching to detect:
    - Exact matches
    - Abbreviations (e.g., "SICAV" vs "Société d'Investissement")
    - Minor variations (e.g., "Fund" vs "Fonds")
    - Contradictions (different fund names)

    Args:
        document: Document dictionary with fund name
        prospectus_data: Prospectus data with official fund name
        ai_engine: Optional AI engine for semantic analysis

    Returns:
        Violation dictionary if fund name doesn't match, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: PROSP_011 - Investment objective must match prospectus exactly
    """
    try:
        # Extract fund name from document
        doc_metadata = document.get('document_metadata', {})
        doc_fund_name = doc_metadata.get('fund_name', '')

        # Extract fund name from prospectus
        prospectus_fund_name = prospectus_data.get('fund_name', '')

        if not doc_fund_name or not prospectus_fund_name:
            logger.warning("Fund name missing in document or prospectus")
            return None

        # Normalize for comparison
        doc_name_lower = doc_fund_name.lower().strip()
        prosp_name_lower = prospectus_fund_name.lower().strip()

        # Rule-based check: exact match or very similar
        if doc_name_lower == prosp_name_lower:
            logger.info("Fund names match exactly")
            return None

        # Check for common abbreviations
        common_abbrevs = {
            'sicav': 'société d\'investissement à capital variable',
            'fcp': 'fonds commun de placement',
            'fund': 'fonds',
            'ucits': 'ucit'
        }

        # Normalize abbreviations
        doc_normalized = doc_name_lower
        prosp_normalized = prosp_name_lower
        for abbrev, full in common_abbrevs.items():
            doc_normalized = doc_normalized.replace(abbrev, full)
            prosp_normalized = prosp_normalized.replace(abbrev, full)

        if doc_normalized == prosp_normalized:
            logger.info("Fund names match after normalization")
            return None

        # Use AI for semantic matching if available
        if ai_engine:
            prompt = f"""Analyze if these two fund names refer to the same fund:

DOCUMENT FUND NAME: {doc_fund_name}
PROSPECTUS FUND NAME: {prospectus_fund_name}

Consider:
1. Exact matches
2. Common abbreviations (SICAV, FCP, Fund/Fonds)
3. Minor variations in wording
4. Different funds with similar names

Respond in JSON format:
{{
    "is_same_fund": true/false,
    "confidence": 0-100,
    "reasoning": "explanation",
    "is_violation": true/false
}}

If they clearly refer to DIFFERENT funds, set is_violation=true.
If they are the same fund with minor variations, set is_violation=false.
"""

            ai_result = call_ai_for_semantic_check(prompt, ai_engine)

            if ai_result:
                is_violation = ai_result.get('is_violation', False)
                confidence = ai_result.get('confidence', 60)
                reasoning = ai_result.get('reasoning', 'Semantic analysis completed')

                if is_violation:
                    logger.warning(f"Fund name mismatch detected: {reasoning}")
                    return {
                        'type': 'PROSPECTUS',
                        'severity': 'CRITICAL',
                        'slide': 'Cover Page',
                        'location': 'document_metadata',
                        'rule': 'PROSP_011: Investment objective must match prospectus exactly',
                        'message': f'Fund name mismatch: Document "{doc_fund_name}" vs Prospectus "{prospectus_fund_name}"',
                        'evidence': f'Document: {doc_fund_name}, Prospectus: {prospectus_fund_name}',
                        'confidence': confidence,
                        'method': 'AI_SEMANTIC',
                        'ai_reasoning': reasoning
                    }
                else:
                    logger.info(f"Fund names semantically match: {reasoning}")
                    return None

        # Fallback: if names are very different and no AI, flag as potential violation
        # Calculate simple similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, doc_name_lower, prosp_name_lower).ratio()

        if similarity < 0.5:
            logger.warning(f"Fund name similarity low: {similarity:.2f}")
            return {
                'type': 'PROSPECTUS',
                'severity': 'CRITICAL',
                'slide': 'Cover Page',
                'location': 'document_metadata',
                'rule': 'PROSP_011: Investment objective must match prospectus exactly',
                'message': f'Potential fund name mismatch: Document "{doc_fund_name}" vs Prospectus "{prospectus_fund_name}"',
                'evidence': f'Document: {doc_fund_name}, Prospectus: {prospectus_fund_name}',
                'confidence': 60,
                'method': 'RULE_BASED',
                'rule_hints': f'Similarity score: {similarity:.2f}'
            }

        # Names are similar enough
        logger.info(f"Fund names sufficiently similar: {similarity:.2f}")
        return None

    except Exception as e:
        logger.error(f"Error checking fund name match: {e}")
        return None


# ============================================================================
# STRATEGY CONSISTENCY TOOL
# ============================================================================

@tool
def check_strategy_consistency(document: dict, prospectus_data: dict, ai_engine=None) -> Optional[dict]:
    """
    Check if investment strategy in document is consistent with prospectus.

    This tool detects CONTRADICTIONS, not missing details:
    - VIOLATION: "invests in European stocks" vs "invests exclusively in US stocks"
    - NOT VIOLATION: "invests in S&P 500" vs "invests at least 70% in S&P 500"

    Args:
        document: Document dictionary with strategy description
        prospectus_data: Prospectus data with official strategy
        ai_engine: Optional AI engine for semantic analysis

    Returns:
        Violation dictionary if strategy contradicts prospectus, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: PROSP_001 - Investment strategy must be presented in conformity with legal documentation
    """
    try:
        # Extract strategy from document
        doc_text = extract_all_text_from_doc(document)

        # Extract strategy from prospectus
        prospectus_strategy = prospectus_data.get('strategy', '')

        if not prospectus_strategy:
            logger.info("No prospectus strategy provided for comparison")
            return None

        # Use AI for semantic consistency check
        if ai_engine:
            prompt = f"""Analyze if the marketing document CONTRADICTS the prospectus strategy.

MARKETING DOCUMENT TEXT (excerpt):
{doc_text[:1500]}

PROSPECTUS STRATEGY:
{prospectus_strategy[:1000]}

IMPORTANT DISTINCTION:
1. CONTRADICTION (violation):
   - Marketing: "invests in European stocks"
   - Prospectus: "invests exclusively in US stocks"
   - These directly contradict each other

2. MISSING DETAIL (NOT violation):
   - Marketing: "invests in S&P 500"
   - Prospectus: "invests at least 70% in S&P 500 equities"
   - Marketing is simplified but not contradictory

3. SIMPLIFIED WORDING (NOT violation):
   - Marketing: "growth strategy"
   - Prospectus: "seeks capital appreciation through growth-oriented investments"
   - Same meaning, different detail level

Respond in JSON format:
{{
    "is_contradiction": true/false,
    "confidence": 0-100,
    "reasoning": "explanation",
    "evidence": ["specific contradictory statements"]
}}

Only set is_contradiction=true if there is a DIRECT CONTRADICTION, not just missing details.
"""

            ai_result = call_ai_for_semantic_check(prompt, ai_engine)

            if ai_result:
                is_contradiction = ai_result.get('is_contradiction', False)
                confidence = ai_result.get('confidence', 60)
                reasoning = ai_result.get('reasoning', 'Consistency check completed')
                evidence = ai_result.get('evidence', [])

                if is_contradiction:
                    logger.warning(f"Strategy contradiction detected: {reasoning}")
                    return {
                        'type': 'PROSPECTUS',
                        'severity': 'CRITICAL',
                        'slide': 'Multiple',
                        'location': 'strategy_description',
                        'rule': 'PROSP_001: Investment strategy must be presented in conformity with legal documentation',
                        'message': 'Investment strategy contradicts prospectus',
                        'evidence': ', '.join(evidence) if isinstance(evidence, list) else str(evidence),
                        'confidence': confidence,
                        'method': 'AI_SEMANTIC',
                        'ai_reasoning': reasoning
                    }
                else:
                    logger.info(f"Strategy is consistent: {reasoning}")
                    return None

        # Fallback: rule-based keyword contradiction detection
        doc_lower = doc_text.lower()
        prosp_lower = prospectus_strategy.lower()

        # Check for obvious contradictions
        contradictions = []

        # Geographic contradictions
        if 'europe' in doc_lower and 'us' in prosp_lower and 'exclusively' in prosp_lower:
            contradictions.append("Geographic focus mismatch")
        if 'us' in doc_lower and 'europe' in prosp_lower and 'exclusively' in prosp_lower:
            contradictions.append("Geographic focus mismatch")

        # Asset class contradictions
        if 'equity' in doc_lower and 'bond' in prosp_lower and 'only' in prosp_lower:
            contradictions.append("Asset class mismatch")
        if 'bond' in doc_lower and 'equity' in prosp_lower and 'only' in prosp_lower:
            contradictions.append("Asset class mismatch")

        if contradictions:
            logger.warning(f"Rule-based contradictions found: {contradictions}")
            return {
                'type': 'PROSPECTUS',
                'severity': 'CRITICAL',
                'slide': 'Multiple',
                'location': 'strategy_description',
                'rule': 'PROSP_001: Investment strategy must be presented in conformity with legal documentation',
                'message': 'Potential investment strategy contradiction',
                'evidence': ', '.join(contradictions),
                'confidence': 65,
                'method': 'RULE_BASED',
                'rule_hints': 'Keyword-based contradiction detection'
            }

        # No contradictions found
        logger.info("No strategy contradictions detected")
        return None

    except Exception as e:
        logger.error(f"Error checking strategy consistency: {e}")
        return None


# ============================================================================
# BENCHMARK VALIDATION TOOL
# ============================================================================

@tool
def check_benchmark_validation(document: dict, prospectus_data: dict, ai_engine=None) -> Optional[dict]:
    """
    Check if benchmark used in document matches the official prospectus benchmark.

    Validates:
    - Benchmark name matches prospectus
    - No unauthorized alternative benchmarks
    - Benchmark specifications match (e.g., dividends reinvested)

    Args:
        document: Document dictionary with benchmark references
        prospectus_data: Prospectus data with official benchmark
        ai_engine: Optional AI engine for semantic analysis

    Returns:
        Violation dictionary if benchmark doesn't match, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rules: PROSP_004, PROSP_005 - Performance benchmark must be the official one from prospectus
    """
    try:
        # Extract document text
        doc_text = extract_all_text_from_doc(document)

        # Extract benchmark from prospectus
        prospectus_benchmark = prospectus_data.get('benchmark', '')

        if not prospectus_benchmark:
            logger.info("No prospectus benchmark provided for comparison")
            return None

        # Use AI for semantic benchmark matching
        if ai_engine:
            prompt = f"""Analyze if the document uses the correct benchmark from the prospectus.

DOCUMENT TEXT (excerpt):
{doc_text[:1500]}

PROSPECTUS OFFICIAL BENCHMARK:
{prospectus_benchmark}

Check for:
1. Is the prospectus benchmark mentioned in the document?
2. Are there any OTHER benchmarks mentioned that are NOT the official one?
3. Do benchmark specifications match (e.g., "dividends reinvested", "net return")?

VIOLATIONS:
- Using a different benchmark than the official one
- Using multiple benchmarks when only one is authorized
- Incorrect benchmark specifications

NOT VIOLATIONS:
- Mentioning the official benchmark correctly
- Not mentioning any benchmark at all

Respond in JSON format:
{{
    "uses_correct_benchmark": true/false,
    "uses_unauthorized_benchmark": true/false,
    "confidence": 0-100,
    "reasoning": "explanation",
    "evidence": ["specific benchmark mentions"]
}}
"""

            ai_result = call_ai_for_semantic_check(prompt, ai_engine)

            if ai_result:
                uses_unauthorized = ai_result.get('uses_unauthorized_benchmark', False)
                confidence = ai_result.get('confidence', 60)
                reasoning = ai_result.get('reasoning', 'Benchmark check completed')
                evidence = ai_result.get('evidence', [])

                if uses_unauthorized:
                    logger.warning(f"Unauthorized benchmark detected: {reasoning}")
                    return {
                        'type': 'PROSPECTUS',
                        'severity': 'CRITICAL',
                        'slide': 'Performance',
                        'location': 'benchmark_reference',
                        'rule': 'PROSP_004: Performance benchmark must be the official one from prospectus',
                        'message': 'Unauthorized or incorrect benchmark used',
                        'evidence': ', '.join(evidence) if isinstance(evidence, list) else str(evidence),
                        'confidence': confidence,
                        'method': 'AI_SEMANTIC',
                        'ai_reasoning': reasoning
                    }
                else:
                    logger.info(f"Benchmark usage is correct: {reasoning}")
                    return None

        # Fallback: rule-based benchmark detection
        doc_lower = doc_text.lower()
        prosp_benchmark_lower = prospectus_benchmark.lower()

        # Common benchmark names
        common_benchmarks = [
            'msci world', 'msci europe', 'msci usa', 's&p 500', 'stoxx 600',
            'cac 40', 'dax', 'ftse 100', 'nikkei', 'hang seng',
            'bloomberg', 'barclays', 'ice bofa'
        ]

        # Check if prospectus benchmark is mentioned
        prospectus_benchmark_found = any(
            term in doc_lower
            for term in prosp_benchmark_lower.split()
            if len(term) > 3
        )

        # Check for other benchmarks
        other_benchmarks_found = []
        for benchmark in common_benchmarks:
            if benchmark in doc_lower and benchmark not in prosp_benchmark_lower:
                other_benchmarks_found.append(benchmark)

        if other_benchmarks_found and not prospectus_benchmark_found:
            logger.warning(f"Unauthorized benchmarks found: {other_benchmarks_found}")
            return {
                'type': 'PROSPECTUS',
                'severity': 'CRITICAL',
                'slide': 'Performance',
                'location': 'benchmark_reference',
                'rule': 'PROSP_004: Performance benchmark must be the official one from prospectus',
                'message': f'Potential unauthorized benchmark(s): {", ".join(other_benchmarks_found)}',
                'evidence': f'Found: {", ".join(other_benchmarks_found)}, Expected: {prospectus_benchmark}',
                'confidence': 65,
                'method': 'RULE_BASED',
                'rule_hints': 'Keyword-based benchmark detection'
            }

        # No violations found
        logger.info("Benchmark usage appears correct")
        return None

    except Exception as e:
        logger.error(f"Error checking benchmark validation: {e}")
        return None


# ============================================================================
# INVESTMENT OBJECTIVE TOOL
# ============================================================================

@tool
def check_investment_objective(document: dict, prospectus_data: dict, ai_engine=None) -> Optional[dict]:
    """
    Check if investment objective in document matches prospectus.

    Validates that the fund's stated investment objective aligns with
    the official prospectus wording. Detects contradictions in:
    - Investment goals (growth vs income vs balanced)
    - Target returns or performance objectives
    - Risk profile statements

    Args:
        document: Document dictionary with investment objective
        prospectus_data: Prospectus data with official objective
        ai_engine: Optional AI engine for semantic analysis

    Returns:
        Violation dictionary if objective doesn't match, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: PROSP_011 - Investment objective must match prospectus exactly
    """
    try:
        # Extract document text
        doc_text = extract_all_text_from_doc(document)

        # Extract investment objective from prospectus
        prospectus_objective = prospectus_data.get('investment_objective', '')

        if not prospectus_objective:
            logger.info("No prospectus investment objective provided for comparison")
            return None

        # Use AI for semantic objective matching
        if ai_engine:
            prompt = f"""Analyze if the document's investment objective CONTRADICTS the prospectus.

DOCUMENT TEXT (excerpt):
{doc_text[:1500]}

PROSPECTUS INVESTMENT OBJECTIVE:
{prospectus_objective}

Check for CONTRADICTIONS in:
1. Investment goal (growth vs income vs balanced)
2. Target returns or performance objectives
3. Risk profile (conservative vs aggressive)
4. Investment approach (active vs passive)

VIOLATIONS (contradictions):
- Document: "seeks income", Prospectus: "seeks capital growth"
- Document: "conservative approach", Prospectus: "aggressive growth strategy"
- Document: "passive index tracking", Prospectus: "active management"

NOT VIOLATIONS (simplifications):
- Document: "growth fund", Prospectus: "seeks long-term capital appreciation"
- Document: "invests in equities", Prospectus: "invests primarily in equity securities"

Respond in JSON format:
{{
    "is_contradiction": true/false,
    "confidence": 0-100,
    "reasoning": "explanation",
    "evidence": ["specific contradictory statements"]
}}

Only set is_contradiction=true for DIRECT CONTRADICTIONS, not simplifications.
"""

            ai_result = call_ai_for_semantic_check(prompt, ai_engine)

            if ai_result:
                is_contradiction = ai_result.get('is_contradiction', False)
                confidence = ai_result.get('confidence', 60)
                reasoning = ai_result.get('reasoning', 'Objective check completed')
                evidence = ai_result.get('evidence', [])

                if is_contradiction:
                    logger.warning(f"Investment objective contradiction detected: {reasoning}")
                    return {
                        'type': 'PROSPECTUS',
                        'severity': 'CRITICAL',
                        'slide': 'Multiple',
                        'location': 'investment_objective',
                        'rule': 'PROSP_011: Investment objective must match prospectus exactly',
                        'message': 'Investment objective contradicts prospectus',
                        'evidence': ', '.join(evidence) if isinstance(evidence, list) else str(evidence),
                        'confidence': confidence,
                        'method': 'AI_SEMANTIC',
                        'ai_reasoning': reasoning
                    }
                else:
                    logger.info(f"Investment objective is consistent: {reasoning}")
                    return None

        # Fallback: rule-based objective contradiction detection
        doc_lower = doc_text.lower()
        prosp_lower = prospectus_objective.lower()

        # Check for obvious contradictions
        contradictions = []

        # Growth vs Income contradictions
        if 'income' in doc_lower and 'growth' in prosp_lower and ('capital appreciation' in prosp_lower or 'capital growth' in prosp_lower):
            contradictions.append("Objective mismatch: income vs growth")
        if 'growth' in doc_lower and 'income' in prosp_lower and ('dividend' in prosp_lower or 'yield' in prosp_lower):
            contradictions.append("Objective mismatch: growth vs income")

        # Active vs Passive contradictions
        if 'passive' in doc_lower and 'active' in prosp_lower:
            contradictions.append("Management style mismatch: passive vs active")
        if 'index' in doc_lower and 'active management' in prosp_lower:
            contradictions.append("Management style mismatch: index vs active")

        if contradictions:
            logger.warning(f"Rule-based objective contradictions found: {contradictions}")
            return {
                'type': 'PROSPECTUS',
                'severity': 'CRITICAL',
                'slide': 'Multiple',
                'location': 'investment_objective',
                'rule': 'PROSP_011: Investment objective must match prospectus exactly',
                'message': 'Potential investment objective contradiction',
                'evidence': ', '.join(contradictions),
                'confidence': 65,
                'method': 'RULE_BASED',
                'rule_hints': 'Keyword-based contradiction detection'
            }

        # No contradictions found
        logger.info("No investment objective contradictions detected")
        return None

    except Exception as e:
        logger.error(f"Error checking investment objective: {e}")
        return None


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# Export all tools for agent use
PROSPECTUS_TOOLS = [
    check_fund_name_match,
    check_strategy_consistency,
    check_benchmark_validation,
    check_investment_objective
]


if __name__ == "__main__":
    # Test the tools
    logger.info("Prospectus Tools Module")
    logger.info("=" * 70)
    logger.info(f"Available tools: {len(PROSPECTUS_TOOLS)}")
    for tool in PROSPECTUS_TOOLS:
        logger.info(f"  - {tool.name}: {tool.description[:80]}...")
