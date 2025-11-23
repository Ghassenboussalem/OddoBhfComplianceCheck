#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for preprocessing tools

Tests:
- extract_metadata
- build_whitelist
- normalize_document
- validate_document
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.preprocessing_tools import (
    extract_metadata,
    build_whitelist,
    normalize_document,
    validate_document
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def valid_document():
    """Valid test document"""
    return {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_esg_classification': 'article_8',
            'country_code': 'FR',
            'fund_age_years': 4.5
        },
        'page_de_garde': {
            'title': 'Fund Presentation'
        },
        'slide_2': {
            'content': 'Investment strategy'
        },
        'pages_suivantes': [
            {'slide_number': 3, 'content': 'Performance'},
            {'slide_number': 4, 'content': 'Risk'}
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management'
        }
    }


@pytest.fixture
def minimal_document():
    """Minimal document with only metadata"""
    return {
        'document_metadata': {
            'fund_isin': 'FR0010135103'
        }
    }


# ============================================================================
# EXTRACT METADATA TESTS
# ============================================================================

def test_extract_metadata_valid_document(valid_document):
    """Test metadata extraction from valid document"""
    result = extract_metadata.func(document=valid_document)
    
    assert result['fund_isin'] == 'FR0010135103'
    assert result['fund_name'] == 'ODDO BHF Algo Trend US Fund'
    assert result['client_type'] == 'retail'
    assert result['document_type'] == 'fund_presentation'
    assert result['esg_classification'] == 'article_8'
    assert result['country_code'] == 'FR'
    assert result['fund_age_years'] == 4.5


def test_extract_metadata_missing_fields():
    """Test metadata extraction with missing fields"""
    doc = {'document_metadata': {}}
    result = extract_metadata.func(document=doc)
    
    assert result['fund_isin'] is None
    assert result['client_type'] == 'retail'  # default
    assert result['document_type'] == 'fund_presentation'  # default
    assert result['esg_classification'] == 'other'  # default


def test_extract_metadata_no_metadata():
    """Test metadata extraction with no metadata section"""
    doc = {}
    result = extract_metadata.func(document=doc)
    
    assert result['fund_isin'] is None
    assert result['client_type'] == 'retail'


# ============================================================================
# BUILD WHITELIST TESTS
# ============================================================================

def test_build_whitelist_valid_document(valid_document):
    """Test whitelist building from valid document"""
    metadata = extract_metadata.func(document=valid_document)
    result = build_whitelist.func(document=valid_document, metadata=metadata)
    
    assert isinstance(result, set)
    assert len(result) > 0


def test_build_whitelist_empty_document():
    """Test whitelist building from empty document"""
    doc = {'document_metadata': {}}
    metadata = {}
    result = build_whitelist.func(document=doc, metadata=metadata)
    
    assert isinstance(result, set)


# ============================================================================
# NORMALIZE DOCUMENT TESTS
# ============================================================================

def test_normalize_document_valid(valid_document):
    """Test document normalization with valid document"""
    result = normalize_document.func(document=valid_document)
    
    assert 'page_de_garde' in result
    assert 'slide_2' in result
    assert 'pages_suivantes' in result
    assert 'page_de_fin' in result
    assert 'document_metadata' in result
    assert isinstance(result['pages_suivantes'], list)
    assert len(result['pages_suivantes']) == 2


def test_normalize_document_missing_sections():
    """Test normalization with missing sections"""
    doc = {'document_metadata': {'fund_isin': 'FR001'}}
    result = normalize_document.func(document=doc)
    
    assert 'page_de_garde' in result
    assert result['page_de_garde'] == {}
    assert 'pages_suivantes' in result
    assert result['pages_suivantes'] == []


def test_normalize_document_adds_slide_numbers():
    """Test that normalization adds slide numbers"""
    doc = {
        'pages_suivantes': [
            {'content': 'Page 1'},
            {'content': 'Page 2'}
        ]
    }
    result = normalize_document.func(document=doc)
    
    assert result['pages_suivantes'][0]['slide_number'] == 3
    assert result['pages_suivantes'][1]['slide_number'] == 4


# ============================================================================
# VALIDATE DOCUMENT TESTS
# ============================================================================

def test_validate_document_valid(valid_document):
    """Test validation of valid document"""
    result = validate_document.func(document=valid_document)
    
    assert result['valid'] is True
    assert result['total_errors'] == 0
    assert 'sections_present' in result


def test_validate_document_missing_metadata():
    """Test validation with missing metadata"""
    doc = {'page_de_garde': {}}
    result = validate_document.func(document=doc)
    
    assert result['valid'] is False
    assert result['total_errors'] > 0
    assert 'Missing required section: document_metadata' in result['errors']


def test_validate_document_no_content():
    """Test validation with no content"""
    doc = {'document_metadata': {}}
    result = validate_document.func(document=doc)
    
    assert result['valid'] is False
    assert any('no content' in err.lower() for err in result['errors'])


def test_validate_document_invalid_pages_suivantes():
    """Test validation with invalid pages_suivantes"""
    doc = {
        'document_metadata': {},
        'pages_suivantes': 'not a list'
    }
    result = validate_document.func(document=doc)
    
    assert result['valid'] is False
    assert any('must be a list' in err for err in result['errors'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
