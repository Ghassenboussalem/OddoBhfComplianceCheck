#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Enhanced Compliance Check Functions
Each check uses AI with rule-based validation
"""

import json
import re
from typing import Dict, List, Optional

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_all_text_from_doc(doc):
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


def call_llm(prompt):
    """
    Call LLM with the given prompt
    This is a placeholder that should be replaced with actual AI engine call
    """
    try:
        from ai_engine import create_ai_engine_from_env
        ai_engine = create_ai_engine_from_env()
        
        if ai_engine:
            # Use the AI engine to call LLM
            response = ai_engine.call_with_cache(prompt)
            if response and not response.error:
                # Check if already parsed
                if response.parsed_json:
                    return response.parsed_json
                # Try to parse JSON response
                import json
                try:
                    result = json.loads(response.content)
                    return result
                except json.JSONDecodeError:
                    # If not JSON, return as dict with content
                    return {'content': response.content}
            else:
                return None
        else:
            # Return None if AI not available
            return None
    except Exception as e:
        # Silently fail and return None for graceful degradation
        return None


# ============================================================================
# STRUCTURE CHECKS - AI ENHANCED
# ============================================================================

def check_promotional_mention_ai(doc):
    """Check for promotional document mention using AI + Rules"""
    
    page_de_garde = doc.get('page_de_garde', {})
    cover_text = json.dumps(page_de_garde, ensure_ascii=False).lower()
    
    # Rule-based check
    def rule_check(text):
        promotional_keywords = [
            'document promotionnel', 'promotional document',
            'document à caractère promotionnel', 'promotional material',
            'marketing document', 'document marketing',
            'matériel promotionnel', 'marketing material'
        ]
        
        found = [kw for kw in promotional_keywords if kw in text]
        
        return {
            'violation': len(found) == 0,
            'confidence': 90 if len(found) == 0 else 0,
            'slide': 'Cover Page',
            'location': 'page_de_garde',
            'rule': 'STRUCT_003: Must indicate "promotional document"',
            'message': 'Missing promotional document mention',
            'evidence': f'Keywords checked: {", ".join(promotional_keywords[:3])}...',
            'hints': f'Found keywords: {found}' if found else 'No keywords found'
        }
    
    # AI prompt
    ai_prompt = f"""Analyze this cover page for promotional document indication.

COVER PAGE TEXT:
{cover_text[:2000]}

REGULATORY REQUIREMENT (French/EU):
Documents must clearly indicate they are "promotional" or "marketing" materials on the cover page.
This can be stated as:
- "Document promotionnel"
- "Promotional document"
- "Document à caractère promotionnel"
- "Marketing material"
- Similar variations in French or English

TASK:
1. Is there a CLEAR, EXPLICIT indication this is promotional/marketing material?
2. What exact phrase is used (if any)?
3. Is it prominently placed on the cover page?
4. Consider variations, abbreviations, or indirect mentions

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "found_phrases": ["exact phrases found"],
  "location": "where on cover page",
  "slide": "Cover Page",
  "rule": "STRUCT_003: Must indicate promotional document",
  "message": "brief message",
  "evidence": "what you found or didn't find",
  "reasoning": "your detailed analysis"
}}

Return ONLY valid JSON, no markdown:"""

    return check_with_ai_and_rules(
        "Promotional Mention",
        cover_text,
        ai_prompt,
        rule_check,
        severity='CRITICAL'
    )


def check_target_audience_ai(doc):
    """Check for target audience specification using AI + Rules"""
    
    page_de_garde = doc.get('page_de_garde', {})
    cover_text = json.dumps(page_de_garde, ensure_ascii=False).lower()
    
    # Rule-based check
    def rule_check(text):
        audience_keywords = [
            'investisseurs non professionnels', 'investisseurs professionnels',
            'non-professional investors', 'professional investors',
            'retail investors', 'qualified investors',
            'document destiné aux', 'intended for',
            'clients professionnels', 'clients non professionnels'
        ]
        
        found = [kw for kw in audience_keywords if kw in text]
        
        return {
            'violation': len(found) == 0,
            'confidence': 85 if len(found) == 0 else 0,
            'slide': 'Cover Page',
            'location': 'page_de_garde',
            'rule': 'STRUCT_004: Must indicate target audience',
            'message': 'Target audience not specified',
            'evidence': f'Checked for: retail/professional investor mentions',
            'hints': f'Found: {found}' if found else 'No audience specification found'
        }
    
    # AI prompt
    ai_prompt = f"""Analyze this cover page for target audience specification.

COVER PAGE TEXT:
{cover_text[:2000]}

REGULATORY REQUIREMENT:
Documents must clearly specify their target audience:
- "Retail investors" / "Investisseurs non professionnels"
- "Professional investors" / "Investisseurs professionnels"
- "Qualified investors" / "Clients professionnels"

TASK:
1. Is the target audience clearly specified?
2. What exact wording is used?
3. Is it unambiguous?

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "found_audience": "retail/professional/both/none",
  "exact_phrase": "phrase used",
  "slide": "Cover Page",
  "location": "page_de_garde",
  "rule": "STRUCT_004: Must indicate target audience",
  "message": "brief message",
  "evidence": "what you found",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""

    return check_with_ai_and_rules(
        "Target Audience",
        cover_text,
        ai_prompt,
        rule_check,
        severity='CRITICAL'
    )


def check_disclaimers_slide2_ai(doc):
    """Check for required disclaimers on slide 2 using AI + Rules"""
    
    slide_2 = doc.get('slide_2', {})
    slide_2_text = json.dumps(slide_2, ensure_ascii=False).lower()
    
    # Rule-based check
    def rule_check(text):
        required_disclaimers = [
            ('capital risk', ['capital', 'risque de perte', 'risk of loss', 'perte en capital']),
            ('past performance', ['performances passées', 'past performance', 'ne préjugent pas'])
        ]
        
        missing = []
        for disc_name, keywords in required_disclaimers:
            if not any(kw in text for kw in keywords):
                missing.append(disc_name)
        
        return {
            'violation': len(missing) > 0,
            'confidence': 90 if missing else 0,
            'slide': 'Slide 2',
            'location': 'slide_2',
            'rule': 'STRUCT_008: Standard disclaimer must be present',
            'message': f'Missing disclaimers: {", ".join(missing)}' if missing else 'Disclaimers present',
            'evidence': f'Required: capital risk + past performance warnings',
            'hints': f'Missing: {missing}' if missing else 'All present'
        }
    
    # AI prompt
    ai_prompt = f"""Analyze slide 2 for required regulatory disclaimers.

SLIDE 2 TEXT:
{slide_2_text[:3000]}

REGULATORY REQUIREMENTS (MiFID II / AMF):
Slide 2 must contain these disclaimers:
1. Capital risk warning: "Risque de perte en capital" / "Risk of capital loss"
2. Past performance warning: "Les performances passées ne préjugent pas des performances futures" / "Past performance is not indicative of future results"

TASK:
1. Are BOTH disclaimers present?
2. Are they clear and prominent?
3. What exact wording is used?

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "disclaimers_found": ["capital_risk", "past_performance"],
  "disclaimers_missing": ["list if any"],
  "exact_phrases": {{"capital_risk": "phrase", "past_performance": "phrase"}},
  "slide": "Slide 2",
  "location": "slide_2",
  "rule": "STRUCT_008: Standard disclaimers required",
  "message": "brief message",
  "evidence": "what you found",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""

    return check_with_ai_and_rules(
        "Slide 2 Disclaimers",
        slide_2_text,
        ai_prompt,
        rule_check,
        severity='CRITICAL'
    )


def check_management_company_ai(doc):
    """Check for management company legal mention using AI + Rules"""
    
    page_de_fin = doc.get('page_de_fin', {})
    legal_text = json.dumps(page_de_fin, ensure_ascii=False).lower()
    
    # Rule-based check
    def rule_check(text):
        company_keywords = [
            'oddo bhf asset management',
            'oddo bhf am',
            'société de gestion',
            'management company',
            'asset management sas'
        ]
        
        found = [kw for kw in company_keywords if kw in text]
        
        return {
            'violation': len(found) == 0,
            'confidence': 80 if len(found) == 0 else 0,
            'slide': 'Back Page',
            'location': 'page_de_fin',
            'rule': 'STRUCT_011: Legal mention of management company',
            'message': 'Management company legal mention missing',
            'evidence': 'Must include full legal name of management company',
            'hints': f'Found: {found}' if found else 'No company mention'
        }
    
    # AI prompt
    ai_prompt = f"""Analyze the back page for management company legal mention.

BACK PAGE TEXT:
{legal_text[:2000]}

REGULATORY REQUIREMENT:
Documents must include full legal mention of the management company, typically including:
- Company name: "ODDO BHF Asset Management"
- Legal form: "SAS" or similar
- Registration details
- Address

TASK:
1. Is the management company clearly identified?
2. Is the full legal name present?
3. Are registration details included?

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "company_name_found": "exact name",
  "legal_details_present": true/false,
  "slide": "Back Page",
  "location": "page_de_fin",
  "rule": "STRUCT_011: Management company legal mention",
  "message": "brief message",
  "evidence": "what you found or didn't find",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""

    return check_with_ai_and_rules(
        "Management Company",
        legal_text,
        ai_prompt,
        rule_check,
        severity='CRITICAL'
    )


# ============================================================================
# PERFORMANCE CHECKS - AI ENHANCED
# ============================================================================

def check_performance_disclaimers_ai(doc):
    """
    Check that ACTUAL performance data has disclaimers (data-aware version)
    
    This version eliminates false positives by:
    - Only flagging when ACTUAL performance numbers are present (e.g., "15%", "+20%")
    - Ignoring descriptive keywords like "attractive performance", "performance objective"
    - Using semantic matching for disclaimer detection
    - Verifying disclaimer is on SAME slide as performance data
    
    Eliminates 3 false positives from keyword-based approach.
    """
    violations = []
    
    # Initialize EvidenceExtractor
    try:
        from evidence_extractor import EvidenceExtractor
        from ai_engine import create_ai_engine_from_env
        
        ai_engine = create_ai_engine_from_env()
        evidence_extractor = EvidenceExtractor(ai_engine)
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize EvidenceExtractor: {e}")
        print("   Falling back to keyword-based checking")
        # Fallback to old implementation if EvidenceExtractor not available
        return _check_performance_disclaimers_fallback(doc)
    
    # Check each slide for actual performance data
    slides_to_check = []
    if 'slide_2' in doc:
        slides_to_check.append(('Slide 2', doc['slide_2']))
    
    if 'pages_suivantes' in doc:
        for i, page in enumerate(doc['pages_suivantes'], start=3):
            slides_to_check.append((f'Slide {page.get("slide_number", i)}', page))
    
    for slide_name, slide_data in slides_to_check:
        slide_text = json.dumps(slide_data, ensure_ascii=False)
        
        # Use EvidenceExtractor to find ACTUAL performance data (numbers with %)
        perf_data = evidence_extractor.find_performance_data(slide_text)
        
        # Only check if ACTUAL performance data is present
        if not perf_data:
            # No actual performance numbers found - skip this slide
            # This eliminates false positives from descriptive text like:
            # - "attractive performance"
            # - "performance objective"
            # - "strong performance potential"
            continue
        
        # Actual performance data found - check for disclaimer on SAME slide
        required_disclaimer = "performances passées ne préjugent pas"
        disclaimer_match = evidence_extractor.find_disclaimer(slide_text, required_disclaimer)
        
        if not disclaimer_match or not disclaimer_match.found:
            # Performance data without disclaimer - this is a violation
            # Extract evidence from performance data
            perf_values = [pd.value for pd in perf_data[:3]]
            perf_contexts = [pd.context[:80] + "..." for pd in perf_data[:3]]
            
            violations.append({
                'type': 'PERFORMANCE',
                'severity': 'CRITICAL',
                'slide': slide_name,
                'location': slide_data.get('title', 'Performance section'),
                'rule': 'PERF_001: Performance data must have disclaimer',
                'message': f'Actual performance data without required disclaimer',
                'evidence': f'Found performance data: {", ".join(perf_values)}. Context: {perf_contexts[0] if perf_contexts else ""}',
                'confidence': 95,
                'method': 'AI_EVIDENCE_EXTRACTOR',
                'ai_reasoning': f'Detected {len(perf_data)} actual performance data points with numerical values. No disclaimer found on same slide.',
                'rule_hints': f'Performance values: {perf_values}'
            })
    
    return violations


def _check_performance_disclaimers_fallback(doc):
    """
    Fallback implementation using keyword matching
    Used when EvidenceExtractor is not available
    """
    violations = []
    
    slides_to_check = []
    if 'slide_2' in doc:
        slides_to_check.append(('Slide 2', doc['slide_2']))
    
    if 'pages_suivantes' in doc:
        for i, page in enumerate(doc['pages_suivantes'], start=3):
            slides_to_check.append((f'Slide {page.get("slide_number", i)}', page))
    
    for slide_name, slide_data in slides_to_check:
        slide_text = json.dumps(slide_data, ensure_ascii=False).lower()
        
        # Simple pattern matching for performance numbers
        perf_patterns = [
            r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
            r'[+\-]?\d+\s*%',  # +15%, -3%
        ]
        
        has_perf_numbers = any(re.search(pattern, slide_text) for pattern in perf_patterns)
        
        if not has_perf_numbers:
            continue
        
        # Check for disclaimer
        disclaimer_keywords = [
            'performances passées',
            'past performance',
            'ne préjugent pas',
            'not indicative'
        ]
        
        has_disclaimer = any(kw in slide_text for kw in disclaimer_keywords)
        
        if not has_disclaimer:
            violations.append({
                'type': 'PERFORMANCE',
                'severity': 'CRITICAL',
                'slide': slide_name,
                'location': slide_data.get('title', 'Performance section'),
                'rule': 'PERF_001: Performance data must have disclaimer',
                'message': 'Performance data without disclaimer',
                'evidence': 'Performance numbers found without accompanying disclaimer',
                'confidence': 85,
                'method': 'RULE_BASED_FALLBACK'
            })
    
    return violations


def check_document_starts_with_performance_ai(doc):
    """
    Check if document starts with ACTUAL performance data on cover page
    
    This version eliminates false positives by:
    - Only checking the cover page (page_de_garde)
    - Only flagging when ACTUAL performance numbers are present (e.g., "15%", "+20%")
    - Ignoring descriptive keywords like "attractive performance", "performance objective"
    - Using EvidenceExtractor to detect real performance data
    
    Requirements: 3.1, 3.2, 4.1, 4.2, 4.3
    Impact: Part of 3 false positive elimination
    """
    violations = []
    
    # Initialize EvidenceExtractor
    try:
        from evidence_extractor import EvidenceExtractor
        from ai_engine import create_ai_engine_from_env
        
        ai_engine = create_ai_engine_from_env()
        evidence_extractor = EvidenceExtractor(ai_engine)
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize EvidenceExtractor: {e}")
        print("   Falling back to keyword-based checking")
        # Fallback to old implementation if EvidenceExtractor not available
        return _check_document_starts_with_performance_fallback(doc)
    
    # Only check the cover page (page_de_garde)
    if 'page_de_garde' not in doc:
        # No cover page found - cannot violate this rule
        return violations
    
    cover_page = doc['page_de_garde']
    cover_text = json.dumps(cover_page, ensure_ascii=False)
    
    # Use EvidenceExtractor to find ACTUAL performance data (numbers with %)
    perf_data = evidence_extractor.find_performance_data(cover_text)
    
    # Only flag if ACTUAL performance numbers are on cover page
    if perf_data:
        # Filter out low-confidence detections (likely descriptive text)
        high_confidence_perf = [pd for pd in perf_data if pd.confidence >= 60]
        
        if high_confidence_perf:
            # Actual performance data found on cover page - this is a violation
            perf_values = [pd.value for pd in high_confidence_perf[:3]]
            perf_contexts = [pd.context[:80] + "..." for pd in high_confidence_perf[:3]]
            
            violations.append({
                'type': 'PERFORMANCE',
                'severity': 'MAJOR',
                'slide': 'Cover Page',
                'location': 'Beginning of document',
                'rule': 'PERF_001: Document cannot start with performance data',
                'message': f'Document starts with performance data ({high_confidence_perf[0].value})',
                'evidence': f'Found performance data on cover: {", ".join(perf_values)}. Context: {perf_contexts[0] if perf_contexts else ""}. Performance must be preceded by fund presentation.',
                'confidence': high_confidence_perf[0].confidence,
                'method': 'AI_EVIDENCE_EXTRACTOR',
                'ai_reasoning': f'Detected {len(high_confidence_perf)} actual performance data points with numerical values on cover page. Documents should not start with performance data.',
                'rule_hints': f'Performance values on cover: {perf_values}'
            })
    
    # If no actual performance data found, no violation
    # This eliminates false positives from:
    # - "attractive performance" (descriptive, no numbers)
    # - "performance objective" (goal statement, no actual data)
    # - "strong performance potential" (forward-looking, no historical data)
    
    return violations


def _check_document_starts_with_performance_fallback(doc):
    """
    Fallback implementation using keyword matching
    Used when EvidenceExtractor is not available
    """
    violations = []
    
    if 'page_de_garde' not in doc:
        return violations
    
    cover_page = doc['page_de_garde']
    cover_text = json.dumps(cover_page, ensure_ascii=False).lower()
    
    # Simple pattern matching for performance numbers
    perf_patterns = [
        r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
        r'[+\-]?\d+\s*%',  # +15%, -3%
    ]
    
    has_perf_numbers = any(re.search(pattern, cover_text) for pattern in perf_patterns)
    
    if has_perf_numbers:
        violations.append({
            'type': 'PERFORMANCE',
            'severity': 'MAJOR',
            'slide': 'Cover Page',
            'location': 'Beginning of document',
            'rule': 'PERF_001: Document cannot start with performance data',
            'message': 'Document starts with performance data',
            'evidence': 'Performance numbers found on cover page. Performance must be preceded by fund presentation.',
            'confidence': 85,
            'method': 'RULE_BASED_FALLBACK'
        })
    
    return violations


def check_benchmark_comparison_ai(doc):
    """Check that performance is compared to benchmark using AI + Rules"""
    
    all_text = extract_all_text_from_doc(doc).lower()
    
    # Rule-based check
    def rule_check(text):
        has_perf = any(word in text for word in ['performance', 'rendement', 'surperform'])
        has_benchmark = 's&p 500' in text or 'benchmark' in text or 'indicateur de référence' in text
        has_chart = 'chart' in text or 'tableau' in text or 'graphique' in text
        
        return {
            'violation': has_perf and not (has_benchmark and has_chart),
            'confidence': 85,
            'slide': 'Multiple slides',
            'location': 'Performance sections',
            'rule': 'PERF_008: Performance must compare to benchmark',
            'message': 'Performance without benchmark comparison',
            'evidence': 'Performance claims without clear benchmark chart',
            'hints': f'Perf: {has_perf}, Benchmark: {has_benchmark}, Chart: {has_chart}'
        }
    
    # AI prompt
    ai_prompt = f"""Analyze this document for performance vs benchmark comparison.

DOCUMENT EXCERPT:
{all_text[:4000]}

REGULATORY REQUIREMENT:
When performance is presented, it must be compared to the official benchmark index.
This requires:
1. Clear identification of the benchmark
2. Visual comparison (chart/table)
3. Side-by-side data

TASK:
1. Is performance data presented?
2. Is there a clear benchmark comparison?
3. Is there a chart or table showing both?

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "has_performance_data": true/false,
  "has_benchmark_comparison": true/false,
  "benchmark_identified": "benchmark name if found",
  "comparison_method": "chart/table/none",
  "slide": "slide location",
  "location": "where found",
  "rule": "PERF_008: Benchmark comparison required",
  "message": "brief message",
  "evidence": "what you found",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""

    return check_with_ai_and_rules(
        "Benchmark Comparison",
        all_text,
        ai_prompt,
        rule_check,
        severity='MAJOR'
    )


# ============================================================================
# PROSPECTUS COMPLIANCE - AI ENHANCED
# ============================================================================

def check_prospectus_fund_match_ai(doc, prospectus_data):
    """Check fund name matches prospectus using semantic AI"""
    
    if not prospectus_data or not prospectus_data.get('fund_name'):
        return {
            'type': 'PROSPECTUS',
            'severity': 'WARNING',
            'slide': 'Document-wide',
            'location': 'Fund identification',
            'rule': 'PROSP_001: Fund must match prospectus',
            'message': 'Cannot verify - prospectus not parsed',
            'evidence': 'Prospectus fund name not available',
            'confidence': 50,
            'method': 'UNABLE_TO_VERIFY'
        }
    
    doc_metadata = doc.get('document_metadata', {})
    doc_fund_name = doc_metadata.get('fund_name', '')
    prospectus_fund = prospectus_data.get('fund_name', '')
    
    # Rule-based check (simple string matching)
    def rule_check(text):
        doc_normalized = doc_fund_name.lower().strip()
        prosp_normalized = prospectus_fund.lower().strip()
        
        # Simple substring check
        matches = prosp_normalized in doc_normalized or doc_normalized in prosp_normalized
        
        return {
            'violation': not matches,
            'confidence': 70 if not matches else 0,
            'slide': 'Document-wide',
            'location': 'Fund identification',
            'rule': 'PROSP_001: Fund must match prospectus',
            'message': 'Fund name mismatch',
            'evidence': f'Doc: "{doc_fund_name}" vs Prospectus: "{prospectus_fund}"',
            'hints': f'String match: {matches}'
        }
    
    # AI prompt for semantic comparison
    ai_prompt = f"""Compare these two fund names semantically to determine if they refer to the same fund.

DOCUMENT FUND NAME: "{doc_fund_name}"
PROSPECTUS FUND NAME: "{prospectus_fund}"

TASK:
Determine if these names refer to the SAME FUND, considering:
1. Abbreviations: "ODDO BHF" vs "Oddo Bank"
2. Word order: "Algo Trend US" vs "US Algo Trend"
3. Extra/missing words: "Fund", "SICAV", "Sub-fund"
4. Share class: "I-EUR", "R-USD" should be ignored
5. Different conventions but same fund

Examples of MATCHES:
- "ODDO BHF Algo Trend US" ≈ "Algo Trend US Fund"
- "Global Equity Fund Class A" ≈ "Global Equity SICAV"

Examples of NON-MATCHES:
- "European Equity" ≠ "Asian Equity"
- "Growth Fund" ≠ "Value Fund"

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "semantic_similarity": 0-100,
  "is_same_fund": true/false,
  "differences_noted": ["list of differences"],
  "slide": "Document-wide",
  "location": "Fund identification",
  "rule": "PROSP_001: Fund must match prospectus",
  "message": "brief message",
  "evidence": "your analysis",
  "reasoning": "detailed explanation"
}}

Return ONLY valid JSON:"""

    return check_with_ai_and_rules(
        "Prospectus Fund Match",
        f"{doc_fund_name} || {prospectus_fund}",
        ai_prompt,
        rule_check,
        severity='CRITICAL'
    )


# ============================================================================
# GENERAL CHECKS - AI ENHANCED
# ============================================================================

def check_glossary_requirement_ai(doc, client_type):
    """Check if technical terms require a glossary using AI"""
    
    if client_type.lower() != 'retail':
        return None  # Professional clients don't need glossary
    
    all_text = extract_all_text_from_doc(doc).lower()
    
    # Rule-based check
    def rule_check(text):
        technical_terms = [
            'momentum', 'quantitative', 'quantitatif', 'volatility', 'volatilité',
            's&p 500', 'sri', 'smart momentum', 'systematic', 'systématique',
            'behavioral finance', 'finance comportementale', 'derivatives',
            'hedge ratio', 'alpha', 'beta', 'sharpe ratio'
        ]
        
        has_technical = any(term in text for term in technical_terms)
        has_glossary = 'glossaire' in text or 'glossary' in text
        
        terms_found = [t for t in technical_terms if t in text]
        
        return {
            'violation': has_technical and not has_glossary,
            'confidence': 90 if (has_technical and not has_glossary) else 0,
            'slide': 'End of document',
            'location': 'Missing glossary',
            'rule': 'GEN_006: Retail docs with technical terms need glossary',
            'message': f'Technical terms without glossary: {len(terms_found)} terms',
            'evidence': f'Found: {", ".join(terms_found[:5])}{"..." if len(terms_found) > 5 else ""}',
            'hints': f'Technical terms: {len(terms_found)}, Glossary: {has_glossary}'
        }
    
    # AI prompt
    ai_prompt = f"""Analyze this RETAIL investor document for technical terms and glossary.

DOCUMENT EXCERPT:
{all_text[:4000]}

REGULATORY REQUIREMENT (AMF):
Documents for retail (non-professional) investors must include a glossary if they contain technical/specialized financial terms.

TASK:
1. Identify technical terms used (financial jargon, specialized concepts)
2. Is there a glossary section?
3. Are the technical terms explained?

Technical terms include:
- Investment strategies: "momentum", "quantitative", "systematic"
- Metrics: "volatility", "Sharpe ratio", "alpha", "beta"
- Instruments: "derivatives", "futures", "swaps"
- Indices: "S&P 500", "MSCI World"

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "technical_terms_found": ["list of terms"],
  "has_glossary": true/false,
  "terms_explained": true/false,
  "slide": "End of document",
  "location": "Glossary section",
  "rule": "GEN_006: Glossary required for retail",
  "message": "brief message",
  "evidence": "what you found",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""

    return check_with_ai_and_rules(
        "Glossary Requirement",
        all_text,
        ai_prompt,
        rule_check,
        severity='MAJOR'
    )


# ============================================================================
# REGISTRATION CHECK - AI ENHANCED
# ============================================================================

def check_registration_countries_ai(doc, fund_isin, authorized_countries):
    """
    Enhanced registration compliance checking with AI-powered country extraction
    
    Implements:
    - AI-powered country extraction from documents
    - Semantic matching for country name variations
    - Validation logic against authorized countries database
    - Confidence scoring for country identification
    """
    
    doc_text = extract_all_text_from_doc(doc)
    
    # Rule-based check
    def rule_check(text):
        # Simple keyword matching for countries
        found_countries = []
        for country in authorized_countries:
            if country.lower() in text.lower():
                found_countries.append(country)
        
        return {
            'violation': False,  # Rules can't determine violations, only hints
            'confidence': 60,
            'found_countries': found_countries,
            'hints': f'Found {len(found_countries)} authorized countries in text'
        }
    
    # AI prompt for country extraction
    ai_prompt = f"""Extract ALL countries mentioned in this document and determine if they are distribution authorization claims.

DOCUMENT TEXT (excerpt):
{doc_text[:3000]}

AUTHORIZED COUNTRIES FOR THIS FUND:
{', '.join(authorized_countries[:20])}{'...' if len(authorized_countries) > 20 else ''}

TASK:
1. Extract ALL country names mentioned in the document
2. For each country, determine if it's mentioned as:
   - Distribution authorization ("Authorized in", "Distributed in", "Available in")
   - Example/illustration ("For example, in France...")
   - General reference ("European markets including...")
3. Identify any countries mentioned for distribution that are NOT in the authorized list
4. Consider variations: "USA" = "United States", "UK" = "United Kingdom", etc.

Look for phrases like:
- "Autorisé à la distribution en:", "Authorized in:", "Distributed in:"
- "Pays d'autorisation:", "Countries of authorization:"
- "Available to investors in:"

Respond with JSON:
{{
  "countries_found": [
    {{
      "country_name": "Country name",
      "context_type": "distribution_claim|example|general_reference",
      "exact_phrase": "phrase where mentioned",
      "is_authorized": true/false
    }}
  ],
  "unauthorized_distribution_claims": ["Country1", "Country2"],
  "confidence": 0-100,
  "location": "where found in document",
  "slide": "slide identifier",
  "rule": "Countries must match registration database",
  "message": "brief message",
  "evidence": "specific evidence",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""
    
    result = call_llm(ai_prompt)
    
    violations = []
    
    if result and result.get('unauthorized_distribution_claims'):
        unauthorized = result.get('unauthorized_distribution_claims', [])
        
        for country in unauthorized:
            violations.append({
                'type': 'REGISTRATION',
                'severity': 'CRITICAL',
                'slide': result.get('slide', 'Back page'),
                'location': result.get('location', 'Authorization list'),
                'rule': 'REG_001: Countries must match registration database',
                'message': f"Unauthorized distribution claim: {country}",
                'evidence': result.get('evidence', f"Fund {fund_isin} not authorized in {country}"),
                'confidence': result.get('confidence', 80),
                'method': 'AI_DETECTED',
                'ai_reasoning': result.get('reasoning', ''),
                'rule_hints': f"Authorized in: {', '.join(authorized_countries[:10])}"
            })
    
    return violations


# ============================================================================
# STRUCTURE CHECKS - AI ENHANCED (Additional)
# ============================================================================

def check_structure_semantic_ai(doc, client_type='retail'):
    """
    Enhanced structure validation with semantic understanding
    
    Implements:
    - Target audience detection with context understanding
    - Management company mention validation
    - Semantic date and fund name validation
    - Flexible structure validation with AI reasoning
    """
    violations = []
    
    # Check date format and consistency
    all_text = extract_all_text_from_doc(doc)
    
    # Rule-based check for dates
    def rule_check_dates(text):
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, text))
        
        return {
            'violation': len(dates_found) == 0,
            'confidence': 70,
            'dates_found': dates_found,
            'hints': f'Found {len(dates_found)} dates'
        }
    
    # AI prompt for date validation
    ai_prompt = f"""Analyze this document for date consistency and format.

DOCUMENT TEXT (excerpt):
{all_text[:2000]}

REGULATORY REQUIREMENT:
Documents must include:
1. Document creation/publication date
2. Data as-of date (for performance/statistics)
3. Dates must be consistent and properly formatted

TASK:
1. Identify all dates in the document
2. Check if document date is present
3. Check if dates are consistent
4. Verify date formats are appropriate

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "document_date_found": true/false,
  "document_date": "date if found",
  "dates_consistent": true/false,
  "issues_found": ["list of issues"],
  "slide": "slide location",
  "location": "where found",
  "rule": "STRUCT_012: Document must include proper dates",
  "message": "brief message",
  "evidence": "specific evidence",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""
    
    result = call_llm(ai_prompt)
    
    if result and not result.get('compliant', True):
        violations.append({
            'type': 'STRUCTURE',
            'severity': 'MAJOR',
            'slide': result.get('slide', 'Document-wide'),
            'location': result.get('location', 'Date fields'),
            'rule': result.get('rule', 'STRUCT_012: Document dates required'),
            'message': result.get('message', 'Date validation issues'),
            'evidence': result.get('evidence', 'Date inconsistencies found'),
            'confidence': result.get('confidence', 75),
            'method': 'AI_DETECTED',
            'ai_reasoning': result.get('reasoning', '')
        })
    
    return violations


# ============================================================================
# GENERAL RULES - AI ENHANCED
# ============================================================================

def check_general_semantic_ai(doc, client_type='retail'):
    """
    Enhanced general rules with AI context awareness
    
    Implements:
    - Semantic glossary term detection for technical language
    - Enhanced Morningstar date validation with context awareness
    - Intelligent technical term identification
    - Context-aware rule application logic
    """
    violations = []
    
    all_text = extract_all_text_from_doc(doc)
    
    # Check for Morningstar rating date
    # Rule-based check
    def rule_check_morningstar(text):
        morningstar_keywords = ['morningstar', 'rating', 'étoiles', 'stars']
        has_morningstar = any(kw in text.lower() for kw in morningstar_keywords)
        
        date_patterns = [r'\d{1,2}/\d{1,2}/\d{4}', r'as of', r'au \d{1,2}']
        has_date = any(re.search(pattern, text.lower()) for pattern in date_patterns)
        
        return {
            'violation': has_morningstar and not has_date,
            'confidence': 85,
            'has_morningstar': has_morningstar,
            'has_date': has_date,
            'hints': f'Morningstar: {has_morningstar}, Date: {has_date}'
        }
    
    # AI prompt for Morningstar validation
    ai_prompt = f"""Analyze this document for Morningstar rating compliance.

DOCUMENT TEXT (excerpt):
{all_text[:3000]}

REGULATORY REQUIREMENT (AMF):
If a Morningstar rating is displayed, it MUST include:
1. The date of the rating ("as of DD/MM/YYYY" or "au DD/MM/YYYY")
2. The rating must be current (not outdated)
3. Clear indication it's a Morningstar rating

TASK:
1. Is there a Morningstar rating displayed?
2. If yes, is the date clearly indicated?
3. Is the date format appropriate?
4. Is the rating properly attributed to Morningstar?

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "has_morningstar_rating": true/false,
  "rating_date_present": true/false,
  "rating_date": "date if found",
  "rating_value": "rating if found",
  "slide": "slide location",
  "location": "where found",
  "rule": "GEN_004: Morningstar rating must include date",
  "message": "brief message",
  "evidence": "specific evidence",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""
    
    result = call_llm(ai_prompt)
    
    if result and result.get('has_morningstar_rating') and not result.get('rating_date_present'):
        violations.append({
            'type': 'GENERAL',
            'severity': 'MAJOR',
            'slide': result.get('slide', 'Unknown'),
            'location': result.get('location', 'Morningstar rating'),
            'rule': result.get('rule', 'GEN_004: Morningstar rating requires date'),
            'message': result.get('message', 'Morningstar rating without date'),
            'evidence': result.get('evidence', 'Rating displayed without as-of date'),
            'confidence': result.get('confidence', 85),
            'method': 'AI_DETECTED',
            'ai_reasoning': result.get('reasoning', '')
        })
    
    # Check for technical terms requiring glossary (retail only)
    if client_type.lower() == 'retail':
        # AI prompt for technical terms
        ai_prompt_glossary = f"""Analyze this RETAIL investor document for technical terms and glossary.

DOCUMENT TEXT (excerpt):
{all_text[:3000]}

REGULATORY REQUIREMENT (AMF):
Documents for retail (non-professional) investors must include a glossary if they contain technical/specialized financial terms.

TASK:
1. Identify technical financial terms used (not common language)
2. Check if there's a glossary section
3. Verify if technical terms are explained

Technical terms include:
- Investment strategies: "momentum", "quantitative", "systematic", "alpha", "beta"
- Metrics: "volatility", "Sharpe ratio", "tracking error", "duration"
- Instruments: "derivatives", "futures", "swaps", "options"
- Indices: "S&P 500", "MSCI World", "Stoxx 600"
- Jargon: "overweight", "underweight", "hedge ratio"

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "technical_terms_found": ["term1", "term2"],
  "has_glossary": true/false,
  "terms_explained": true/false,
  "slide": "slide location",
  "location": "Glossary section",
  "rule": "GEN_006: Glossary required for retail with technical terms",
  "message": "brief message",
  "evidence": "what you found",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""
        
        result_glossary = call_llm(ai_prompt_glossary)
        
        if result_glossary and len(result_glossary.get('technical_terms_found', [])) > 0 and not result_glossary.get('has_glossary'):
            violations.append({
                'type': 'GENERAL',
                'severity': 'MAJOR',
                'slide': result_glossary.get('slide', 'End of document'),
                'location': result_glossary.get('location', 'Missing glossary'),
                'rule': result_glossary.get('rule', 'GEN_006: Glossary required for retail'),
                'message': result_glossary.get('message', f"Technical terms without glossary: {len(result_glossary.get('technical_terms_found', []))} terms"),
                'evidence': result_glossary.get('evidence', f"Found: {', '.join(result_glossary.get('technical_terms_found', [])[:5])}"),
                'confidence': result_glossary.get('confidence', 90),
                'method': 'AI_DETECTED',
                'ai_reasoning': result_glossary.get('reasoning', '')
            })
    
    return violations


# ============================================================================
# VALUES/SECURITIES CHECKS - AI ENHANCED
# ============================================================================

def check_values_semantic_ai(doc):
    """
    Enhanced values/securities checks with semantic analysis
    
    Implements:
    - Context understanding for company mentions
    - Distinguish between examples and recommendations using AI
    - Semantic disclaimer detection for securities content
    - Intent analysis for investment advice detection
    """
    violations = []
    
    all_text = extract_all_text_from_doc(doc)
    
    # Rule-based check for company mentions
    def rule_check_companies(text):
        # Common company indicators
        company_indicators = [
            'apple', 'microsoft', 'google', 'amazon', 'tesla',
            'lvmh', 'total', 'bnp', 'société générale',
            'stock', 'share', 'equity', 'company'
        ]
        
        found_indicators = [ind for ind in company_indicators if ind in text.lower()]
        
        return {
            'violation': len(found_indicators) > 0,
            'confidence': 60,
            'found_indicators': found_indicators,
            'hints': f'Found {len(found_indicators)} company/stock indicators'
        }
    
    # AI prompt for company mention analysis
    ai_prompt = f"""Analyze this document for company/securities mentions and investment advice.

DOCUMENT TEXT (excerpt):
{all_text[:3000]}

REGULATORY REQUIREMENT (MiFID II / AMF):
Documents must NOT:
1. Recommend specific securities/companies without proper disclaimers
2. Provide investment advice on individual stocks
3. Suggest buying/selling specific securities

Acceptable mentions:
- Examples for illustration: "For example, a company like Apple..."
- Portfolio holdings disclosure: "The fund holds positions in..."
- Sector examples: "Technology companies such as Microsoft..."

NOT acceptable:
- Recommendations: "We recommend investing in Apple"
- Advice: "Buy Tesla stock now"
- Predictions: "Amazon will outperform"

TASK:
1. Identify all company/security mentions
2. Determine if they are:
   - Examples/illustrations (OK)
   - Portfolio holdings (OK with disclosure)
   - Recommendations/advice (NOT OK)
3. Check for required disclaimers
4. Assess intent (informational vs. advisory)

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "company_mentions": [
    {{
      "company_name": "name",
      "context_type": "example|holding|recommendation|advice",
      "exact_phrase": "phrase used",
      "requires_disclaimer": true/false
    }}
  ],
  "investment_advice_detected": true/false,
  "disclaimers_present": true/false,
  "slide": "slide location",
  "location": "where found",
  "rule": "VAL_001: No investment advice on specific securities",
  "message": "brief message",
  "evidence": "specific evidence",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""
    
    result = call_llm(ai_prompt)
    
    if result and result.get('investment_advice_detected') and not result.get('disclaimers_present'):
        violations.append({
            'type': 'VALUES',
            'severity': 'CRITICAL',
            'slide': result.get('slide', 'Unknown'),
            'location': result.get('location', 'Company mentions'),
            'rule': result.get('rule', 'VAL_001: No investment advice on specific securities'),
            'message': result.get('message', 'Investment advice without disclaimers'),
            'evidence': result.get('evidence', 'Specific securities mentioned as recommendations'),
            'confidence': result.get('confidence', 85),
            'method': 'AI_DETECTED',
            'ai_reasoning': result.get('reasoning', '')
        })
    
    return violations


# ============================================================================
# CROSS-SLIDE VALIDATION - AI ENHANCED
# ============================================================================

def check_risk_profile_consistency(doc):
    """
    Check risk profile consistency across slides (cross-slide validation)
    
    Validates that Slide 2 risk profile includes all risks mentioned elsewhere in document.
    Slide 2 should have >= risks from other pages (comprehensive risk disclosure).
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    Impact: Catches 1 missed violation
    
    Returns:
        list: Violations if Slide 2 has fewer risks than other pages
    """
    violations = []
    
    # Extract Slide 2 content
    slide_2 = doc.get('slide_2', {})
    if not slide_2:
        # No Slide 2 found - cannot perform check
        return violations
    
    slide_2_text = json.dumps(slide_2, ensure_ascii=False)
    
    # Extract all other pages content (pages_suivantes + page_de_fin)
    other_pages_text = ""
    
    # Add pages_suivantes
    if 'pages_suivantes' in doc:
        for page in doc['pages_suivantes']:
            other_pages_text += json.dumps(page, ensure_ascii=False) + "\n"
    
    # Add final page
    if 'page_de_fin' in doc:
        other_pages_text += json.dumps(doc['page_de_fin'], ensure_ascii=False)
    
    # Extract risks from Slide 2 using regex patterns
    # Look for risk keywords in French and English
    risk_patterns_slide2 = [
        r'risque de perte en capital',
        r'risque lié aux actions',
        r'risque de change',
        r'risque lié à la gestion discrétionnaire',
        r'risque de taux d\'intérêt',
        r'risque de crédit',
        r'risque de volatilité',
        r'risque de contrepartie',
        r'risque de liquidité',
        r'risque lié aux engagements',
        r'risques liés à la conversion monétaire',
        r'risque lié aux marchés émergents',
        r'risque lié à la durabilité',
        r'capital risk',
        r'equity risk',
        r'currency risk',
        r'discretionary management risk',
        r'interest rate risk',
        r'credit risk',
        r'volatility risk',
        r'counterparty risk',
        r'liquidity risk',
        r'emerging markets risk',
        r'sustainability risk'
    ]
    
    # Find risks on Slide 2
    slide_2_risks = set()
    slide_2_text_lower = slide_2_text.lower()
    
    for pattern in risk_patterns_slide2:
        if re.search(pattern, slide_2_text_lower):
            # Normalize risk name for comparison
            risk_name = pattern.replace(r'\\', '').replace(r'\'', "'")
            slide_2_risks.add(risk_name)
    
    # Extract risks from other pages
    other_pages_risks = set()
    other_pages_text_lower = other_pages_text.lower()
    
    for pattern in risk_patterns_slide2:
        if re.search(pattern, other_pages_text_lower):
            # Normalize risk name for comparison
            risk_name = pattern.replace(r'\\', '').replace(r'\'', "'")
            other_pages_risks.add(risk_name)
    
    # Compare risk counts
    slide_2_count = len(slide_2_risks)
    other_pages_count = len(other_pages_risks)
    
    # Slide 2 should have >= risks from other pages
    if slide_2_count < other_pages_count:
        # Find missing risks
        missing_risks = other_pages_risks - slide_2_risks
        
        # Format missing risks for evidence
        missing_risks_list = sorted(list(missing_risks))
        missing_risks_str = '\n   - '.join(missing_risks_list[:10])  # Show first 10
        if len(missing_risks_list) > 10:
            missing_risks_str += f'\n   ... and {len(missing_risks_list) - 10} more'
        
        violations.append({
            'type': 'STRUCTURE',
            'severity': 'MAJOR',
            'slide': 'Slide 2',
            'location': 'Risk profile section',
            'rule': 'STRUCT_009: Slide 2 risk profile must be comprehensive',
            'message': f'Incomplete risk profile on Slide 2: {slide_2_count} risks vs {other_pages_count} elsewhere in document',
            'evidence': f'Slide 2 mentions {slide_2_count} risk(s), but other pages mention {other_pages_count} risk(s).\n\nMissing risks on Slide 2:\n   - {missing_risks_str}\n\nSlide 2 should include all major risks disclosed elsewhere in the document.',
            'confidence': 95,
            'method': 'CROSS_SLIDE_VALIDATION',
            'ai_reasoning': f'Cross-slide analysis detected inconsistency: Slide 2 risk disclosure is incomplete compared to other pages. Regulatory requirement: comprehensive risk disclosure must appear early in document (Slide 2).',
            'rule_hints': f'Slide 2 risks: {sorted(list(slide_2_risks))[:5]}... | Other pages risks: {sorted(list(other_pages_risks))[:5]}...'
        })
    
    return violations


def check_anglicisms_retail(doc, client_type):
    """
    Check for English terms (anglicisms) in retail documents without glossary
    
    Only applies to retail documents (skip professional).
    Detects common English terms used in French financial documents.
    Flags as MINOR violation if terms found but no glossary present.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    Impact: Catches 1 missed violation
    
    Args:
        doc: Document dictionary
        client_type: 'retail' or 'professional'
    
    Returns:
        list: Violations if anglicisms found without glossary in retail docs
    """
    violations = []
    
    # Only apply to retail documents
    if client_type.lower() != 'retail':
        # Professional clients don't need glossary for anglicisms
        return violations
    
    # Extract all text from document
    all_text = extract_all_text_from_doc(doc)
    all_text_lower = all_text.lower()
    
    # Define list of common English terms in French financial documents
    # These are technical terms that should be explained in a glossary for retail investors
    common_anglicisms = [
        'momentum',
        'smart',
        'trend',
        'tracking error',
        'hedge',
        'alpha',
        'beta',
        'benchmark',
        'rating',
        'overweight',
        'underweight',
        'overlay',
        'swap',
        'futures',
        'options',
        'hedge ratio',
        'sharpe ratio',
        'value at risk',
        'var',
        'stress test',
        'backtesting',
        'quantitative',
        'systematic'
    ]
    
    # Check which terms are used in the document
    terms_found = []
    for term in common_anglicisms:
        # Use word boundary matching to avoid partial matches
        # e.g., "momentum" should match but not "momentums" in middle of French word
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, all_text_lower):
            terms_found.append(term)
    
    # If no anglicisms found, no violation
    if not terms_found:
        return violations
    
    # Check if glossary exists
    # Look for "glossaire" or "glossary" in the document
    has_glossary = bool(re.search(r'\b(glossaire|glossary)\b', all_text_lower))
    
    # If terms found and no glossary, flag as violation
    if terms_found and not has_glossary:
        # Format terms for evidence
        terms_str = ', '.join(terms_found[:10])  # Show first 10 terms
        if len(terms_found) > 10:
            terms_str += f' ... and {len(terms_found) - 10} more'
        
        violations.append({
            'type': 'GENERAL',
            'severity': 'MINOR',
            'slide': 'Document-wide',
            'location': 'Missing glossary',
            'rule': 'GEN_005: Retail documents with anglicisms must include glossary',
            'message': f'English terms used without glossary: {len(terms_found)} term(s)',
            'evidence': f'Found English terms: {terms_str}. Retail documents should include a glossary ("glossaire") to explain technical English terms for non-professional investors.',
            'confidence': 90,
            'method': 'RULE_BASED_ANGLICISM_DETECTION',
            'ai_reasoning': f'Detected {len(terms_found)} English technical terms in retail document. AMF guidelines require glossary for technical terms to ensure retail investor comprehension.',
            'rule_hints': f'Terms found: {terms_found[:5]}... | Glossary present: {has_glossary}'
        })
    
    return violations


print("All AI-enhanced check functions loaded (including remaining checks)")
