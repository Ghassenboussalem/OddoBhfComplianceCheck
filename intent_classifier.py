#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Classifier - AI-Powered Intent Detection
Determines whether text is advice, description, fact, or example to distinguish
between prohibited client advice and allowed fund descriptions.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any
from data_models import IntentClassification, IntentType, SubjectType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies text intent using AI to distinguish between different types of content.
    
    Key capabilities:
    - Classify intent as ADVICE, DESCRIPTION, FACT, or EXAMPLE
    - Detect client advice (prohibited in marketing materials)
    - Identify fund descriptions (allowed in marketing materials)
    - Handle edge cases and ambiguous text
    - Provide explainable reasoning
    """
    
    def __init__(self, ai_engine):
        """
        Initialize Intent Classifier
        
        Args:
            ai_engine: AIEngine instance for LLM calls
        """
        self.ai_engine = ai_engine
        self._init_prompt_templates()
        logger.info("IntentClassifier initialized")
    
    def _init_prompt_templates(self):
        """Initialize AI prompt templates for intent classification"""
        
        # Main intent classification template
        self.intent_classification_prompt = """Analyze this text and classify its intent.

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

        # Client advice detection template
        self.client_advice_prompt = """Analyze this text to determine if it contains advice to clients.

TEXT: {text}

CLIENT ADVICE INDICATORS:
- Direct recommendations: "vous devriez", "you should", "nous recommandons", "we recommend"
- Imperative statements: "investissez", "invest", "achetez", "buy"
- Timing advice: "bon moment", "good time", "maintenant", "now"
- Action suggestions: "il faut", "must", "devez", "should"

NOT CLIENT ADVICE:
- Fund descriptions: "le fonds investit", "the fund invests"
- Strategy goals: "vise à", "aims to", "cherche à", "seeks to"
- Factual statements: "a généré", "generated", "est composé", "consists of"

TASK:
Determine if this text advises clients what to do (PROHIBITED) or describes the fund (ALLOWED).

Respond with JSON only, no additional text:
{{
  "is_client_advice": true/false,
  "confidence": 0-100,
  "subject": "fund|client|general",
  "reasoning": "explanation",
  "evidence": "supporting phrases"
}}"""

        # Fund description detection template
        self.fund_description_prompt = """Analyze this text to determine if it describes the fund's characteristics or strategy.

TEXT: {text}

FUND DESCRIPTION INDICATORS:
- Investment approach: "le fonds investit", "the fund invests", "la stratégie", "the strategy"
- Strategy goals: "vise à", "aims to", "cherche à", "seeks to", "tirer parti", "take advantage"
- Portfolio composition: "composé de", "consists of", "allocation", "répartition"
- Fund characteristics: "domicilié", "domiciled", "géré par", "managed by"

NOT FUND DESCRIPTION:
- Client advice: "vous devriez", "you should", "nous recommandons", "we recommend"
- Client actions: "investissez", "invest", "achetez", "buy"

TASK:
Determine if this text describes what the fund does (ALLOWED) or advises clients (PROHIBITED).

Respond with JSON only, no additional text:
{{
  "is_fund_description": true/false,
  "confidence": 0-100,
  "subject": "fund|client|general",
  "reasoning": "explanation",
  "evidence": "supporting phrases"
}}"""
    
    def classify_intent(self, text: str) -> IntentClassification:
        """
        Classify text intent using AI
        
        Args:
            text: Text to classify
            
        Returns:
            IntentClassification with intent_type, confidence, subject, reasoning, and evidence
            
        Examples:
            >>> classifier.classify_intent("Le fonds investit dans des actions")
            IntentClassification(intent_type="DESCRIPTION", confidence=95, subject="fund", ...)
            
            >>> classifier.classify_intent("Vous devriez investir maintenant")
            IntentClassification(intent_type="ADVICE", confidence=95, subject="client", ...)
        """
        try:
            # Try AI classification first
            prompt = self.intent_classification_prompt.format(text=text)
            system_message = "You are a financial compliance expert. Classify text intent accurately. Return only valid JSON with no additional text."
            
            response = self.ai_engine.call_with_cache(prompt, system_message)
            
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
                return self._fallback_rule_based_classification(text)
                
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return self._fallback_rule_based_classification(text)
    
    def is_client_advice(self, text: str) -> bool:
        """
        Check if text advises clients what to do (PROHIBITED)
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is client advice
            
        Examples:
            >>> classifier.is_client_advice("Vous devriez investir")
            True
            
            >>> classifier.is_client_advice("Le fonds investit")
            False
        """
        try:
            # Try AI analysis first
            prompt = self.client_advice_prompt.format(text=text)
            system_message = "You are a financial compliance expert. Detect client advice accurately. Return only valid JSON with no additional text."
            
            response = self.ai_engine.call_with_cache(prompt, system_message)
            
            if response and response.parsed_json:
                result = response.parsed_json
                is_advice = result.get("is_client_advice", False)
                confidence = result.get("confidence", 50)
                
                # High confidence advice detection
                if is_advice and confidence >= 70:
                    return True
                
                # Also check intent classification for additional validation
                classification = self.classify_intent(text)
                if classification.intent_type == "ADVICE" and classification.confidence >= 70:
                    return True
                
                return False
            else:
                # AI failed, use fallback
                logger.warning("AI client advice detection failed, using fallback")
                return self._is_client_advice_fallback(text)
                
        except Exception as e:
            logger.error(f"Client advice detection error: {e}")
            return self._is_client_advice_fallback(text)
    
    def is_fund_description(self, text: str) -> bool:
        """
        Check if text describes the fund's characteristics or strategy (ALLOWED)
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is fund description
            
        Examples:
            >>> classifier.is_fund_description("Le fonds investit dans des actions")
            True
            
            >>> classifier.is_fund_description("Vous devriez acheter ce fonds")
            False
        """
        try:
            # Try AI analysis first
            prompt = self.fund_description_prompt.format(text=text)
            system_message = "You are a financial compliance expert. Detect fund descriptions accurately. Return only valid JSON with no additional text."
            
            response = self.ai_engine.call_with_cache(prompt, system_message)
            
            if response and response.parsed_json:
                result = response.parsed_json
                is_description = result.get("is_fund_description", False)
                confidence = result.get("confidence", 50)
                
                # High confidence description detection
                if is_description and confidence >= 70:
                    return True
                
                # Also check intent classification for additional validation
                classification = self.classify_intent(text)
                if classification.intent_type == "DESCRIPTION" and classification.confidence >= 70:
                    return True
                
                return False
            else:
                # AI failed, use fallback
                logger.warning("AI fund description detection failed, using fallback")
                return self._is_fund_description_fallback(text)
                
        except Exception as e:
            logger.error(f"Fund description detection error: {e}")
            return self._is_fund_description_fallback(text)
    
    def _fallback_rule_based_classification(self, text: str) -> IntentClassification:
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
    
    def _is_client_advice_fallback(self, text: str) -> bool:
        """Fallback rule-based client advice detection"""
        text_lower = text.lower()
        
        advice_patterns = [
            r'\bvous devriez\b', r'\byou should\b', r'\bnous recommandons\b',
            r'\bwe recommend\b', r'\binvestissez\b', r'\binvest now\b',
            r'\bachetez\b', r'\bbuy\b', r'\bil faut\b', r'\bmust\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in advice_patterns)
    
    def _is_fund_description_fallback(self, text: str) -> bool:
        """Fallback rule-based fund description detection"""
        text_lower = text.lower()
        
        description_patterns = [
            r'\ble fonds\b', r'\bthe fund\b', r'\bla stratégie\b',
            r'\bthe strategy\b', r'\bfonds investit\b', r'\bfund invests\b',
            r'\bstratégie vise\b', r'\bstrategy aims\b', r'\btirer parti\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in description_patterns)


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Intent Classifier - Standalone Test")
    print("="*70)
    
    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    if not ai_engine:
        print("\n✗ Failed to initialize AI Engine")
        print("  Using fallback rule-based classification only")
        print("\nNote: Set GEMINI_API_KEY or TOKENFACTORY_API_KEY in .env for full AI classification")
    
    # Create classifier
    classifier = IntentClassifier(ai_engine) if ai_engine else None
    
    if classifier:
        print("\n✓ Intent Classifier initialized")
        
        # Test cases
        test_cases = [
            {
                "text": "Tirer parti du momentum des marchés américains grâce à une stratégie quantitative",
                "expected_intent": "DESCRIPTION",
                "expected_advice": False,
                "expected_description": True,
                "description": "Fund strategy description (French)"
            },
            {
                "text": "Le fonds investit dans des actions américaines à forte capitalisation",
                "expected_intent": "DESCRIPTION",
                "expected_advice": False,
                "expected_description": True,
                "description": "Fund investment description"
            },
            {
                "text": "Vous devriez investir dans ce fonds maintenant",
                "expected_intent": "ADVICE",
                "expected_advice": True,
                "expected_description": False,
                "description": "Client investment advice"
            },
            {
                "text": "Nous recommandons d'acheter des actions technologiques",
                "expected_intent": "ADVICE",
                "expected_advice": True,
                "expected_description": False,
                "description": "Recommendation to client"
            },
            {
                "text": "Le fonds a généré un rendement de 15% en 2023",
                "expected_intent": "FACT",
                "expected_advice": False,
                "expected_description": False,
                "description": "Factual performance statement"
            },
            {
                "text": "Par exemple, un investissement de 1000€ aurait généré 150€",
                "expected_intent": "EXAMPLE",
                "expected_advice": False,
                "expected_description": False,
                "description": "Illustrative example"
            },
            {
                "text": "La stratégie vise à générer des rendements supérieurs au marché",
                "expected_intent": "DESCRIPTION",
                "expected_advice": False,
                "expected_description": True,
                "description": "Strategy goal description"
            },
            {
                "text": "Il faut investir maintenant pour profiter de cette opportunité",
                "expected_intent": "ADVICE",
                "expected_advice": True,
                "expected_description": False,
                "description": "Urgent investment advice"
            }
        ]
        
        print("\n" + "="*70)
        print("Running Test Cases")
        print("="*70)
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[Test {i}] {test['description']}")
            print(f"Text: \"{test['text']}\"")
            print(f"Expected Intent: {test['expected_intent']}")
            
            # Classify intent
            classification = classifier.classify_intent(test['text'])
            is_advice = classifier.is_client_advice(test['text'])
            is_description = classifier.is_fund_description(test['text'])
            
            print(f"\nResults:")
            print(f"  Intent Type: {classification.intent_type}")
            print(f"  Subject: {classification.subject}")
            print(f"  Confidence: {classification.confidence}%")
            print(f"  Is Client Advice: {is_advice}")
            print(f"  Is Fund Description: {is_description}")
            print(f"  Reasoning: {classification.reasoning}")
            if classification.evidence:
                print(f"  Evidence: {classification.evidence}")
            
            # Check if results match expected
            intent_match = classification.intent_type == test['expected_intent']
            advice_match = is_advice == test['expected_advice']
            description_match = is_description == test['expected_description']
            
            all_match = intent_match and advice_match and description_match
            
            if all_match:
                print(f"\n  ✓ PASS - All checks match expected results")
                passed += 1
            else:
                print(f"\n  ✗ FAIL - Mismatch detected:")
                if not intent_match:
                    print(f"    - Intent: expected {test['expected_intent']}, got {classification.intent_type}")
                if not advice_match:
                    print(f"    - Advice: expected {test['expected_advice']}, got {is_advice}")
                if not description_match:
                    print(f"    - Description: expected {test['expected_description']}, got {is_description}")
                failed += 1
        
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        print(f"Total Tests: {len(test_cases)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
        print("="*70)
    else:
        print("\n✗ Cannot run tests without AI Engine")
        print("  Set API keys in .env file to enable testing")
