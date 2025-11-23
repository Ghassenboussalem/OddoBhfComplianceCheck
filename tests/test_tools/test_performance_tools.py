#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for performance checking tools

Tests:
- check_performance_disclaimers
- check_document_starts_with_performance
- check_benchmark_comparison
- check_fund_age_restrictions
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.performance_tools import (
    check_performance_disclaimers,
    check_document_starts_with_performance,
    check_benchmark_comparison,
    check_fund_age_restrictions
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Test configuration"""
    return {'ai_enabled': False}  # Use fallback for testing


@pytest.fixture
def document_with_performance_and_disclaimer():
    """Document with performance data and disclaimer"""
    return {
        'slide_2': {
            'content': 'Performance: +15.5% in 2023. Les performances passées ne préjugent pas des performances futures.'
        },
        'pages_suivantes': []
    }


@pytest.fixture
def document_with_performance_no_disclaimer():
    """Document with performance data but no disclaimer"""
    return {
        'slide_2': {
            'content': 'Performance: +15.5% in 2023. Great returns!'
        },
        'pages_suivantes': []
    }


# ============================================================================
# PERFORMANCE DISCLAIMERS TESTS
# ============================================================================

def test_performance_disclaimers_present(document_with_performance_and_disclaimer, config):
    """Test when performance has disclaimer"""
    result = check_performance_disclaimers.func(document=document_with_performance_and_disclaimer, config=config)
    assert isinstance(result, list)
    assert len(result) == 0  # No violations


def test_performance_disclaimers_missing(document_with_performance_no_disclaimer, config):
    """Test when performance lacks disclaimer"""
    result = check_performance_disclaimers.func(document=document_with_performance_no_disclaimer, config=config)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]['type'] == 'PERFORMANCE'
    assert 'disclaimer' in result[0]['message'].lower()


def test_performance_disclaimers_no_performance(config):
    """Test when no performance data present"""
    doc = {'slide_2': {'content': 'Investment strategy description'}}
    result = check_performance_disclaimers.func(document=doc, config=config)
    
    assert isinstance(result, list)
    assert len(result) == 0  # No violations


def test_performance_disclaimers_multiple_slides(config):
    """Test performance disclaimers across multiple slides"""
    doc = {
        'slide_2': {'content': 'Performance: +10%'},
        'pages_suivantes': [
            {'slide_number': 3, 'content': 'Returns: +15.5%'},
            {'slide_number': 4, 'content': 'Risk profile'}
        ]
    }
    result = check_performance_disclaimers.func(document=doc, config=config)
    
    assert isinstance(result, list)
    # Should find violations on slides without disclaimers


# ============================================================================
# DOCUMENT STARTS WITH PERFORMANCE TESTS
# ============================================================================

def test_document_starts_with_performance_no(config):
    """Test when document doesn't start with performance"""
    doc = {
        'page_de_garde': {
            'title': 'ODDO BHF Fund',
            'subtitle': 'Investment strategy'
        }
    }
    result = check_document_starts_with_performance.func(document=doc, config=config)
    
    assert isinstance(result, list)
    assert len(result) == 0


def test_document_starts_with_performance_yes(config):
    """Test when document starts with performance"""
    doc = {
        'page_de_garde': {
            'title': 'ODDO BHF Fund',
            'performance': '+15.5% in 2023'
        }
    }
    result = check_document_starts_with_performance.func(document=doc, config=config)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert 'start' in result[0]['message'].lower()


def test_document_starts_with_performance_no_cover(config):
    """Test when no cover page exists"""
    doc = {'slide_2': {'content': 'Content'}}
    result = check_document_starts_with_performance.func(document=doc, config=config)
    
    assert isinstance(result, list)
    assert len(result) == 0


# ============================================================================
# BENCHMARK COMPARISON TESTS
# ============================================================================

def test_benchmark_comparison_present(config):
    """Test when benchmark comparison is present"""
    doc = {
        'slide_2': {
            'content': 'Performance vs S&P 500 benchmark. Chart showing comparison.'
        }
    }
    result = check_benchmark_comparison.func(document=doc, config=config)
    assert result is None


def test_benchmark_comparison_missing(config):
    """Test when benchmark comparison is missing"""
    doc = {
        'slide_2': {
            'content': 'Performance: +15.5% in 2023'
        }
    }
    result = check_benchmark_comparison.func(document=doc, config=config)
    
    assert result is not None
    assert result['type'] == 'PERFORMANCE'
    assert 'benchmark' in result['message'].lower()


def test_benchmark_comparison_no_performance(config):
    """Test when no performance mentioned"""
    doc = {
        'slide_2': {
            'content': 'Investment strategy and risk profile'
        }
    }
    result = check_benchmark_comparison.func(document=doc, config=config)
    assert result is None


# ============================================================================
# FUND AGE RESTRICTIONS TESTS
# ============================================================================

def test_fund_age_restrictions_old_fund(config):
    """Test fund older than 3 years"""
    doc = {
        'slide_2': {'content': 'Performance: +15.5%'}
    }
    metadata = {'fund_age_years': 4.5}
    result = check_fund_age_restrictions.func(document=doc, metadata=metadata)
    assert result is None


def test_fund_age_restrictions_young_fund_with_performance(config):
    """Test fund younger than 1 year with performance"""
    doc = {
        'slide_2': {'content': 'Performance: +15.5%'}
    }
    metadata = {'fund_age_years': 0.5}
    result = check_fund_age_restrictions.func(document=doc, metadata=metadata)
    
    assert result is not None
    assert result['type'] == 'PERFORMANCE'
    assert '< 1 year' in result['message']


def test_fund_age_restrictions_young_fund_no_performance(config):
    """Test fund younger than 1 year without performance"""
    doc = {
        'slide_2': {'content': 'Investment strategy'}
    }
    metadata = {'fund_age_years': 0.5}
    result = check_fund_age_restrictions.func(document=doc, metadata=metadata)
    assert result is None


def test_fund_age_restrictions_cumulative_performance(config):
    """Test fund younger than 3 years with cumulative performance"""
    doc = {
        'slide_2': {'content': 'Cumulative performance: +25%'}
    }
    metadata = {'fund_age_years': 2.0}
    result = check_fund_age_restrictions.func(document=doc, metadata=metadata)
    
    assert result is not None
    assert '< 3 years' in result['message']


def test_fund_age_restrictions_ytd_allowed(config):
    """Test YTD performance allowed for young funds"""
    doc = {
        'slide_2': {'content': 'YTD performance: +10%'}
    }
    metadata = {'fund_age_years': 2.0}
    result = check_fund_age_restrictions.func(document=doc, metadata=metadata)
    assert result is None


def test_fund_age_restrictions_no_age_info(config):
    """Test when fund age not available"""
    doc = {
        'slide_2': {'content': 'Performance: +15.5%'}
    }
    metadata = {}
    result = check_fund_age_restrictions.func(document=doc, metadata=metadata)
    assert result is None  # Skip check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
