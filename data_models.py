#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models for False Positive Elimination
Dataclasses for context analysis, intent classification, and validation results
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class IntentType(Enum):
    """Types of text intent"""
    ADVICE = "ADVICE"  # Tells clients what they should do
    DESCRIPTION = "DESCRIPTION"  # Describes what the fund does
    FACT = "FACT"  # States objective information
    EXAMPLE = "EXAMPLE"  # Illustrative scenario


class SubjectType(Enum):
    """Who performs the action"""
    FUND = "fund"
    CLIENT = "client"
    GENERAL = "general"


class ValidationMethod(Enum):
    """Method used for validation"""
    AI_ONLY = "AI_ONLY"
    RULES_ONLY = "RULES_ONLY"
    AI_AND_RULES = "AI_AND_RULES"


@dataclass
class ContextAnalysis:
    """Result of context analysis"""
    subject: str  # "fund", "client", "general"
    intent: str  # "describe", "advise", "state_fact"
    confidence: int  # 0-100
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    is_fund_description: bool = False
    is_client_advice: bool = False


@dataclass
class IntentClassification:
    """Result of intent classification"""
    intent_type: str  # "ADVICE", "DESCRIPTION", "FACT", "EXAMPLE"
    confidence: int  # 0-100
    subject: str  # "fund", "client", "general"
    reasoning: str
    evidence: str = ""


@dataclass
class ValidationResult:
    """Result of semantic validation"""
    is_violation: bool
    confidence: int  # 0-100
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    method: str = "AI_ONLY"  # "AI_ONLY", "RULES_ONLY", "AI_AND_RULES"
    rule_hints: Optional[str] = None


@dataclass
class Evidence:
    """Evidence supporting a finding"""
    quotes: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)  # ["Slide 2", "Cover Page"]
    context: str = ""
    confidence: int = 0


@dataclass
class PerformanceData:
    """Performance data found in text"""
    value: str  # "15%", "+20%", etc.
    context: str  # Surrounding text
    location: str  # Slide/section
    confidence: int = 0


@dataclass
class DisclaimerMatch:
    """Disclaimer matching result"""
    found: bool
    text: str = ""
    location: str = ""
    similarity_score: int = 0  # 0-100
    confidence: int = 0
