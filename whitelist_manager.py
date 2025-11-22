#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WhitelistManager - Manages terms that are allowed to repeat
Prevents false positives for fund names, strategy terms, regulatory terms, etc.
"""

import re
import logging
from typing import Dict, Set, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WhitelistReason:
    """Reason why a term is whitelisted"""
    term: str
    category: str  # "fund_name", "strategy", "regulatory", "benchmark", "generic"
    reason: str


class WhitelistManager:
    """
    Manages whitelists of terms that are allowed to repeat in documents
    
    Prevents false positives by identifying:
    - Fund name components (e.g., "ODDO", "BHF")
    - Strategy terminology (e.g., "momentum", "quantitative")
    - Regulatory terms (e.g., "SRI", "SRRI", "SFDR")
    - Benchmark names (e.g., "S&P", "500", "MSCI")
    - Generic financial terms (e.g., "fund", "investment")
    """
    
    # Default whitelists (class-level constants)
    STRATEGY_TERMS = {
        'momentum', 'quantitative', 'quantitatif', 'systematic', 'systématique',
        'algorithmic', 'algorithmique', 'smart', 'trend', 'behavioral',
        'comportementale', 'value', 'growth', 'blend', 'core', 'satellite',
        'active', 'passive', 'index', 'enhanced', 'tactical', 'strategic',
        'dynamic', 'flexible', 'absolute', 'return', 'alpha', 'beta'
    }
    
    REGULATORY_TERMS = {
        'sri', 'srri', 'sfdr', 'ucits', 'ucit', 'mifid', 'amf', 'esma',
        'kiid', 'kid', 'priips', 'esg', 'article', 'regulation', 'directive',
        'prospectus', 'dici', 'opcvm', 'opc', 'sicav', 'fcp'
    }
    
    BENCHMARK_TERMS = {
        's&p', 'sp', '500', 'msci', 'stoxx', 'eurostoxx', 'cac', 'dax',
        'ftse', 'russell', 'dow', 'jones', 'nasdaq', 'index', 'indice',
        'benchmark', 'indicateur', 'référence', 'composite'
    }
    
    GENERIC_FINANCIAL_TERMS = {
        'actions', 'equities', 'equity', 'bonds', 'obligations', 'fund', 'fonds',
        'portfolio', 'portefeuille', 'investment', 'investissement', 'investor',
        'investisseur', 'asset', 'actif', 'allocation', 'diversification',
        'performance', 'rendement', 'risk', 'risque', 'volatility', 'volatilité',
        'market', 'marché', 'share', 'part', 'class', 'classe', 'currency',
        'devise', 'hedge', 'couverture', 'exposure', 'exposition'
    }
    
    MANAGEMENT_COMPANY_TERMS = {
        'oddo', 'bhf', 'asset', 'management', 'gestion', 'am', 'sas', 'gmbh',
        'sa', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
        'group', 'groupe', 'holding', 'partners', 'partenaires', 'capital'
    }
    
    def __init__(self):
        """Initialize WhitelistManager with empty custom whitelists"""
        self.fund_name_terms: Set[str] = set()
        self.custom_terms: Set[str] = set()
        self._whitelist_reasons: Dict[str, WhitelistReason] = {}
        
        logger.info("WhitelistManager initialized")
    
    def build_whitelist(self, doc: Dict) -> Set[str]:
        """
        Build comprehensive whitelist from document metadata
        
        Args:
            doc: Document dictionary with metadata
        
        Returns:
            Set of all whitelisted terms (lowercase)
        """
        # Reset fund name terms
        self.fund_name_terms = set()
        
        # Extract fund name from metadata
        doc_metadata = doc.get('document_metadata', {})
        fund_name = doc_metadata.get('fund_name', '')
        
        if fund_name:
            # Split fund name into components and add to whitelist
            fund_components = self._extract_fund_name_components(fund_name)
            self.fund_name_terms.update(fund_components)
            
            # Record reasons
            for component in fund_components:
                self._whitelist_reasons[component] = WhitelistReason(
                    term=component,
                    category='fund_name',
                    reason=f'Part of fund name: "{fund_name}"'
                )
            
            logger.info(f"Extracted {len(fund_components)} components from fund name: {fund_name}")
        
        # Combine all whitelists
        all_terms = (
            self.fund_name_terms |
            self.STRATEGY_TERMS |
            self.REGULATORY_TERMS |
            self.BENCHMARK_TERMS |
            self.GENERIC_FINANCIAL_TERMS |
            self.MANAGEMENT_COMPANY_TERMS |
            self.custom_terms
        )
        
        logger.info(f"Built whitelist with {len(all_terms)} total terms")
        
        return all_terms
    
    def _extract_fund_name_components(self, fund_name: str) -> Set[str]:
        """
        Extract individual components from fund name
        
        Args:
            fund_name: Full fund name (e.g., "ODDO BHF Algo Trend US")
        
        Returns:
            Set of lowercase components
        """
        components = set()
        
        # Remove common suffixes/prefixes
        cleaned = fund_name
        for suffix in [' Fund', ' SICAV', ' FCP', ' Sub-fund', ' Class', ' Share']:
            cleaned = cleaned.replace(suffix, '')
        
        # Split by spaces and hyphens
        words = re.split(r'[\s\-]+', cleaned)
        
        for word in words:
            # Clean word
            word = word.strip()
            
            # Skip empty, very short, or share class identifiers
            if len(word) < 2:
                continue
            if re.match(r'^[A-Z]-[A-Z]{3}$', word):  # Skip "I-EUR", "R-USD"
                continue
            
            # Add lowercase version
            components.add(word.lower())
            
            # Also add acronyms as-is (e.g., "ODDO", "BHF")
            if word.isupper() and len(word) <= 5:
                components.add(word.lower())
        
        return components
    
    def is_whitelisted(self, term: str) -> bool:
        """
        Check if a term is in any whitelist
        
        Args:
            term: Term to check (case-insensitive)
        
        Returns:
            True if term is whitelisted
        """
        term_lower = term.lower().strip()
        
        # Check all whitelists
        if term_lower in self.fund_name_terms:
            return True
        if term_lower in self.STRATEGY_TERMS:
            return True
        if term_lower in self.REGULATORY_TERMS:
            return True
        if term_lower in self.BENCHMARK_TERMS:
            return True
        if term_lower in self.GENERIC_FINANCIAL_TERMS:
            return True
        if term_lower in self.MANAGEMENT_COMPANY_TERMS:
            return True
        if term_lower in self.custom_terms:
            return True
        
        return False
    
    def get_whitelist_reason(self, term: str) -> Optional[str]:
        """
        Get explanation for why a term is whitelisted
        
        Args:
            term: Term to explain
        
        Returns:
            Reason string or None if not whitelisted
        """
        term_lower = term.lower().strip()
        
        # Check if we have a recorded reason
        if term_lower in self._whitelist_reasons:
            reason = self._whitelist_reasons[term_lower]
            return f"{reason.category}: {reason.reason}"
        
        # Determine category
        if term_lower in self.fund_name_terms:
            return "fund_name: Part of the fund's name"
        if term_lower in self.STRATEGY_TERMS:
            return "strategy: Investment strategy terminology"
        if term_lower in self.REGULATORY_TERMS:
            return "regulatory: Required regulatory term"
        if term_lower in self.BENCHMARK_TERMS:
            return "benchmark: Benchmark/index name"
        if term_lower in self.GENERIC_FINANCIAL_TERMS:
            return "generic: Common financial terminology"
        if term_lower in self.MANAGEMENT_COMPANY_TERMS:
            return "management_company: Management company name"
        if term_lower in self.custom_terms:
            return "custom: Custom whitelisted term"
        
        return None
    
    def add_custom_term(self, term: str, reason: str = "Custom whitelist"):
        """
        Add a custom term to whitelist
        
        Args:
            term: Term to whitelist
            reason: Reason for whitelisting
        """
        term_lower = term.lower().strip()
        self.custom_terms.add(term_lower)
        
        self._whitelist_reasons[term_lower] = WhitelistReason(
            term=term_lower,
            category='custom',
            reason=reason
        )
        
        logger.info(f"Added custom whitelist term: {term_lower} ({reason})")
    
    def get_whitelist_stats(self) -> Dict[str, int]:
        """
        Get statistics about whitelist composition
        
        Returns:
            Dict with counts by category
        """
        return {
            'fund_name_terms': len(self.fund_name_terms),
            'strategy_terms': len(self.STRATEGY_TERMS),
            'regulatory_terms': len(self.REGULATORY_TERMS),
            'benchmark_terms': len(self.BENCHMARK_TERMS),
            'generic_terms': len(self.GENERIC_FINANCIAL_TERMS),
            'management_company_terms': len(self.MANAGEMENT_COMPANY_TERMS),
            'custom_terms': len(self.custom_terms),
            'total': len(self.build_whitelist({}))
        }
    
    def get_all_whitelisted_terms(self) -> Set[str]:
        """
        Get all whitelisted terms across all categories
        
        Returns:
            Set of all whitelisted terms
        """
        return (
            self.fund_name_terms |
            self.STRATEGY_TERMS |
            self.REGULATORY_TERMS |
            self.BENCHMARK_TERMS |
            self.GENERIC_FINANCIAL_TERMS |
            self.MANAGEMENT_COMPANY_TERMS |
            self.custom_terms
        )


# Convenience function for quick whitelist building
def create_whitelist_from_doc(doc: Dict) -> Set[str]:
    """
    Quick function to create whitelist from document
    
    Args:
        doc: Document dictionary
    
    Returns:
        Set of whitelisted terms
    """
    manager = WhitelistManager()
    return manager.build_whitelist(doc)


# Example usage
if __name__ == "__main__":
    # Test with example document
    test_doc = {
        'document_metadata': {
            'fund_name': 'ODDO BHF Algo Trend US Fund'
        }
    }
    
    manager = WhitelistManager()
    whitelist = manager.build_whitelist(test_doc)
    
    print(f"\nWhitelist Statistics:")
    stats = manager.get_whitelist_stats()
    for category, count in stats.items():
        print(f"  {category}: {count}")
    
    print(f"\nFund name components extracted:")
    for term in sorted(manager.fund_name_terms):
        print(f"  - {term}")
    
    print(f"\nTesting whitelist:")
    test_terms = ['ODDO', 'BHF', 'momentum', 'SRI', 'Apple', 'Microsoft']
    for term in test_terms:
        is_whitelisted = manager.is_whitelisted(term)
        reason = manager.get_whitelist_reason(term) if is_whitelisted else "Not whitelisted"
        print(f"  {term}: {'✓' if is_whitelisted else '✗'} - {reason}")
