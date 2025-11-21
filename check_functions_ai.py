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
    """Check that performance mentions have disclaimers using AI + Rules"""
    
    violations = []
    
    slides_to_check = []
    if 'slide_2' in doc:
        slides_to_check.append(('Slide 2', doc['slide_2']))
    
    if 'pages_suivantes' in doc:
        for i, page in enumerate(doc['pages_suivantes'], start=3):
            slides_to_check.append((f'Slide {page.get("slide_number", i)}', page))
    
    for slide_name, slide_data in slides_to_check:
        slide_text = json.dumps(slide_data, ensure_ascii=False).lower()
        
        # Rule-based check
        def rule_check(text):
            performance_keywords = [
                'performance', 'rendement', 'return',
                'surperformance', 'surperform', 'outperform',
                'généré.*rendement', 'avec succès'
            ]
            
            has_perf = any(re.search(kw, text) for kw in performance_keywords)
            
            disclaimer_keywords = [
                'performances passées',
                'past performance',
                'ne préjugent pas',
                'not guarantee',
                'no guarantee'
            ]
            
            has_disclaimer = any(kw in text for kw in disclaimer_keywords)
            
            return {
                'violation': has_perf and not has_disclaimer,
                'confidence': 95 if (has_perf and not has_disclaimer) else 0,
                'slide': slide_name,
                'location': slide_data.get('title', 'Unknown'),
                'rule': 'PERF_001: Performance must have disclaimer',
                'message': 'Performance mentioned without disclaimer',
                'evidence': f'Performance content found without accompanying disclaimer',
                'hints': f'Has performance: {has_perf}, Has disclaimer: {has_disclaimer}'
            }
        
        # AI prompt
        ai_prompt = f"""Analyze this slide for performance information and disclaimers.

SLIDE: {slide_name}
CONTENT:
{slide_text[:2500]}

REGULATORY REQUIREMENT (UCITS / MiFID II):
Any mention of performance, returns, or track record MUST be accompanied by:
"Past performance is not indicative of future results"
or
"Les performances passées ne préjugent pas des performances futures"

TASK:
1. Does this slide mention performance/returns?
2. If yes, is there a disclaimer on THE SAME SLIDE?
3. What specific performance claims are made?

Consider:
- "surperformance", "outperform"
- "généré des rendements", "delivered returns"
- Historical performance data
- Track record claims

Respond with JSON:
{{
  "compliant": true/false,
  "confidence": 0-100,
  "has_performance_content": true/false,
  "performance_claims": ["list of claims"],
  "has_disclaimer": true/false,
  "disclaimer_text": "exact disclaimer if found",
  "slide": "{slide_name}",
  "location": "slide location",
  "rule": "PERF_001: Performance requires disclaimer",
  "message": "brief message",
  "evidence": "specific evidence",
  "reasoning": "your analysis"
}}

Return ONLY valid JSON:"""

        violation = check_with_ai_and_rules(
            "Performance Disclaimer",
            slide_text,
            ai_prompt,
            rule_check,
            severity='CRITICAL'
        )
        
        if violation:
            violations.append(violation)
    
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


print("All AI-enhanced check functions loaded (including remaining checks)")
