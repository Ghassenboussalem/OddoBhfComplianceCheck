#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for context analysis tools

Tests:
- analyze_context
- classify_intent
- extract_subject
- is_fund_strategy_description
- is_investment_advice
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.context_tools import (
    analyze_context,
    classify_intent,
    extract_subject,
    is_fund_strategy_description,
    is_investment_advice
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_ai_engine():
    """Mock AI engine for testing"""
    class MockAIEngine:
        def call_with_cache(self, prompt, system_message):
            # Return None to trigger fallback
            return None
    
    return MockAIEngine()


# ============================================================================
# ANALYZE CONTEXT TESTS
# ============================================================================

def test_analyze_context_fund_description(mock_ai_engine):
    """Test context analysis for fund description"""
    text = "Le fonds investit dans des actions américaines"
    result = analyze_context(text, "general", mock_ai_engine)
    
    # Handle ToolResult wrapper
    analysis = result.result if hasattr(result, 'result') else result
    
    assert analysis.subject == "fund"
    assert analysis.is_fund_description is True
    assert analysis.is_client_advice is False


def test_analyze_context_client_advice(mock_ai_engine):
    """Test context analysis for client advice"""
    text = "Vous devriez investir dans ce fonds maintenant"
    result = analyze_context(text, "investment_advice", mock_ai_engine)
    
    analysis = result.result if hasattr(result, 'result') else result
    
    assert analysis.subject == "client"
    assert analysis.is_client_advice is True
    assert analysis.is_fund_description is False


def test_analyze_context_neutral(mock_ai_engine):
    """Test context analysis for neutral text"""
    text = "Le marché est volatil"
    result = analyze_context(text, "general", mock_ai_engine)
    
    analysis = result.result if hasattr(result, 'result') else result
    
    assert analysis.subject in ["general", "fund", "client"]
    assert isinstance(analysis.confidence, int)


# ============================================================================
# CLASSIFY INTENT TESTS
# ============================================================================

def test_classify_intent_advice(mock_ai_engine):
    """Test intent classification for advice"""
    text = "Vous devriez investir maintenant"
    result = classify_intent(text, mock_ai_engine)
    
    classification = result.result if hasattr(result, 'result') else result
    
    assert classification.intent_type == "ADVICE"
    assert classification.subject == "client"


def test_classify_intent_description(mock_ai_engine):
    """Test intent classification for description"""
    text = "Le fonds investit dans des actions"
    result = classify_intent(text, mock_ai_engine)
    
    classification = result.result if hasattr(result, 'result') else classification
    
    assert classification.intent_type == "DESCRIPTION"
    assert classification.subject == "fund"


def test_classify_intent_fact(mock_ai_engine):
    """Test intent classification for fact"""
    text = "Le fonds a généré 5% en 2023"
    result = classify_intent(text, mock_ai_engine)
    
    classification = result.result if hasattr(result, 'result') else result
    
    assert classification.intent_type in ["FACT", "DESCRIPTION"]


# ============================================================================
# EXTRACT SUBJECT TESTS
# ============================================================================

def test_extract_subject_fund(mock_ai_engine):
    """Test subject extraction for fund"""
    text = "Le fonds investit dans des actions"
    result = extract_subject(text, mock_ai_engine)
    
    subject = result.result if hasattr(result, 'result') else result
    
    assert subject == "fund"


def test_extract_subject_client(mock_ai_engine):
    """Test subject extraction for client"""
    text = "Vous devriez investir"
    result = extract_subject(text, mock_ai_engine)
    
    subject = result.result if hasattr(result, 'result') else result
    
    assert subject == "client"


def test_extract_subject_general(mock_ai_engine):
    """Test subject extraction for general text"""
    text = "Le marché est volatil"
    result = extract_subject(text, mock_ai_engine)
    
    subject = result.result if hasattr(result, 'result') else result
    
    assert subject in ["general", "fund", "client"]


# ============================================================================
# IS FUND STRATEGY DESCRIPTION TESTS
# ============================================================================

def test_is_fund_strategy_description_true(mock_ai_engine):
    """Test fund strategy description detection - positive"""
    texts = [
        "Le fonds investit dans des actions américaines",
        "La stratégie vise à tirer parti du momentum",
        "The fund seeks to invest in growth stocks"
    ]
    
    for text in texts:
        result = is_fund_strategy_description(text, mock_ai_engine)
        is_fund_desc = result.result if hasattr(result, 'result') else result
        assert is_fund_desc is True, f"Failed for: {text}"


def test_is_fund_strategy_description_false(mock_ai_engine):
    """Test fund strategy description detection - negative"""
    texts = [
        "Vous devriez investir maintenant",
        "Nous recommandons d'acheter",
        "You should invest in this fund"
    ]
    
    for text in texts:
        result = is_fund_strategy_description(text, mock_ai_engine)
        is_fund_desc = result.result if hasattr(result, 'result') else result
        assert is_fund_desc is False, f"Failed for: {text}"


# ============================================================================
# IS INVESTMENT ADVICE TESTS
# ============================================================================

def test_is_investment_advice_true(mock_ai_engine):
    """Test investment advice detection - positive"""
    texts = [
        "Vous devriez investir dans ce fonds",
        "Nous recommandons d'acheter maintenant",
        "You should invest now",
        "Il faut investir"
    ]
    
    for text in texts:
        result = is_investment_advice(text, mock_ai_engine)
        is_advice = result.result if hasattr(result, 'result') else result
        assert is_advice is True, f"Failed for: {text}"


def test_is_investment_advice_false(mock_ai_engine):
    """Test investment advice detection - negative"""
    texts = [
        "Le fonds investit dans des actions",
        "La stratégie vise à générer des rendements",
        "The fund invests in equities"
    ]
    
    for text in texts:
        result = is_investment_advice(text, mock_ai_engine)
        is_advice = result.result if hasattr(result, 'result') else result
        assert is_advice is False, f"Failed for: {text}"


# ============================================================================
# EDGE CASES
# ============================================================================

def test_empty_text(mock_ai_engine):
    """Test with empty text"""
    result = analyze_context("", "general", mock_ai_engine)
    analysis = result.result if hasattr(result, 'result') else result
    
    assert isinstance(analysis.subject, str)
    assert isinstance(analysis.confidence, int)


def test_very_long_text(mock_ai_engine):
    """Test with very long text"""
    text = "Le fonds investit " * 100
    result = analyze_context(text, "general", mock_ai_engine)
    analysis = result.result if hasattr(result, 'result') else result
    
    assert analysis.subject == "fund"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
