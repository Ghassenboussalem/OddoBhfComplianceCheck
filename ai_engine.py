#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Engine - LLM Abstraction Layer
Unified interface for Token Factory and Gemini APIs with caching and prompt templates
"""

import json
import hashlib
import logging
import time
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    TOKEN_FACTORY = "token_factory"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: LLMProvider
    api_key: str
    model_name: str
    base_url: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.9
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class AIResponse:
    """Standardized AI response structure"""
    content: str
    parsed_json: Optional[Dict] = None
    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def call(self, prompt: str, system_message: str = "", **kwargs) -> AIResponse:
        """Make a call to the LLM"""
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get the provider type"""
        pass


class TokenFactoryClient(LLMClient):
    """Token Factory (Llama) LLM client"""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Token Factory client
        
        Args:
            config: LLMConfig with Token Factory settings
        """
        self.config = config
        
        try:
            import httpx
            from openai import OpenAI
            
            # Disable SSL verification as per Token Factory documentation
            http_client = httpx.Client(verify=False)
            
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url or "https://tokenfactory.esprit.tn/api",
                http_client=http_client,
                timeout=config.timeout
            )
            
            logger.info(f"Token Factory client initialized: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Token Factory client: {e}")
            raise
    
    def call(self, prompt: str, system_message: str = "", **kwargs) -> AIResponse:
        """
        Call Token Factory API
        
        Args:
            prompt: User prompt
            system_message: System message for context
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with result
        """
        start_time = time.time()
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            parsed_json = None
            try:
                cleaned = self._clean_json_response(content)
                parsed_json = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
            return AIResponse(
                content=content,
                parsed_json=parsed_json,
                provider=LLMProvider.TOKEN_FACTORY,
                model=self.config.model_name,
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None,
                latency_ms=latency_ms,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Token Factory API call failed: {e}")
            return AIResponse(
                content="",
                error=str(e),
                provider=LLMProvider.TOKEN_FACTORY,
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        return LLMProvider.TOKEN_FACTORY
    
    @staticmethod
    def _clean_json_response(text: str) -> str:
        """Clean response text to extract valid JSON"""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        
        return text


class GeminiClient(LLMClient):
    """Google Gemini LLM client"""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Gemini client
        
        Args:
            config: LLMConfig with Gemini settings
        """
        self.config = config
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=config.api_key)
            self.model = genai.GenerativeModel(config.model_name)
            
            logger.info(f"Gemini client initialized: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def call(self, prompt: str, system_message: str = "", **kwargs) -> AIResponse:
        """
        Call Gemini API
        
        Args:
            prompt: User prompt
            system_message: System message (prepended to prompt for Gemini)
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with result
        """
        start_time = time.time()
        
        try:
            # Gemini doesn't have separate system messages, so prepend to prompt
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            response = self.model.generate_content(full_prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response and hasattr(response, 'text'):
                content = response.text.strip()
                
                # Try to parse as JSON
                parsed_json = None
                try:
                    cleaned = TokenFactoryClient._clean_json_response(content)
                    parsed_json = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
                
                return AIResponse(
                    content=content,
                    parsed_json=parsed_json,
                    provider=LLMProvider.GEMINI,
                    model=self.config.model_name,
                    latency_ms=latency_ms,
                    cached=False
                )
            else:
                return AIResponse(
                    content="",
                    error="No response text from Gemini",
                    provider=LLMProvider.GEMINI,
                    latency_ms=latency_ms
                )
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return AIResponse(
                content="",
                error=str(e),
                provider=LLMProvider.GEMINI,
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        return LLMProvider.GEMINI


class ResponseCache:
    """Intelligent in-memory cache for AI responses with LRU eviction and monitoring"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
        """
        self.cache: Dict[str, AIResponse] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_count: Dict[str, int] = {}
        self.last_access_time: Dict[str, float] = {}
        self.creation_time: Dict[str, float] = {}
        
        # Monitoring metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        
        logger.info(f"Response cache initialized (max_size={max_size}, ttl={ttl_seconds}s)")
    
    def get(self, key: str) -> Optional[AIResponse]:
        """
        Get cached response
        
        Args:
            key: Cache key
            
        Returns:
            Cached AIResponse or None
        """
        if key in self.cache:
            # Check if entry has expired
            if self._is_expired(key):
                logger.debug(f"Cache entry expired: {key[:16]}...")
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access tracking
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access_time[key] = time.time()
            
            response = self.cache[key]
            response.cached = True
            
            self.hits += 1
            logger.debug(f"Cache hit: {key[:16]}...")
            return response
        
        self.misses += 1
        logger.debug(f"Cache miss: {key[:16]}...")
        return None
    
    def set(self, key: str, response: AIResponse):
        """
        Store response in cache
        
        Args:
            key: Cache key
            response: AIResponse to cache
        """
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        current_time = time.time()
        self.cache[key] = response
        self.access_count[key] = 1
        self.last_access_time[key] = current_time
        self.creation_time[key] = current_time
        
        logger.debug(f"Cached response: {key[:16]}...")
    
    def invalidate(self, pattern: Optional[str] = None, check_type: Optional[str] = None):
        """
        Invalidate cache entries matching pattern or check type
        
        Args:
            pattern: String pattern to match in cache keys
            check_type: Specific check type to invalidate
        """
        keys_to_remove = []
        
        if pattern:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        elif check_type:
            # Invalidate all entries for a specific check type
            keys_to_remove = [k for k in self.cache.keys() if check_type in k]
        else:
            # Invalidate all
            keys_to_remove = list(self.cache.keys())
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self.invalidations += 1
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def clear(self):
        """Clear all cached responses"""
        count = len(self.cache)
        self.cache.clear()
        self.access_count.clear()
        self.last_access_time.clear()
        self.creation_time.clear()
        self.invalidations += count
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "total_requests": total_requests,
            "total_accesses": sum(self.access_count.values()),
            "ttl_seconds": self.ttl_seconds
        }
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate
        
        Returns:
            Hit rate as percentage (0-100)
        """
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def reset_metrics(self):
        """Reset monitoring metrics"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        logger.info("Cache metrics reset")
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired"""
        if not self.ttl_seconds:
            return False
        
        if key not in self.creation_time:
            return True
        
        age = time.time() - self.creation_time[key]
        return age > self.ttl_seconds
    
    def _remove_entry(self, key: str):
        """Remove a cache entry and its metadata"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_count:
            del self.access_count[key]
        if key in self.last_access_time:
            del self.last_access_time[key]
        if key in self.creation_time:
            del self.creation_time[key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.last_access_time:
            return
        
        # Find key with oldest last access time
        lru_key = min(self.last_access_time, key=self.last_access_time.get)
        self._remove_entry(lru_key)
        self.evictions += 1
        logger.debug(f"Evicted LRU item: {lru_key[:16]}...")
    
    @staticmethod
    def generate_key(prompt: str, system_message: str = "", **kwargs) -> str:
        """
        Generate cache key from prompt and parameters
        
        Args:
            prompt: User prompt
            system_message: System message
            **kwargs: Additional parameters
            
        Returns:
            Cache key (hash)
        """
        # Create deterministic string from inputs
        key_data = f"{system_message}|{prompt}|{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()


class PromptTemplate:
    """Template for generating prompts for different check types"""
    
    def __init__(self, template: str, required_vars: List[str] = None):
        """
        Initialize prompt template
        
        Args:
            template: Template string with {variable} placeholders
            required_vars: List of required variable names
        """
        self.template = template
        self.required_vars = required_vars or []
    
    def render(self, **kwargs) -> str:
        """
        Render template with provided variables
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Rendered prompt string
        """
        # Check required variables
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")


class PromptTemplateLibrary:
    """Library of prompt templates for different compliance checks"""
    
    def __init__(self):
        """Initialize template library with default templates"""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates for compliance checks"""
        
        # Promotional document detection
        self.templates["promotional_mention"] = PromptTemplate(
            template="""Analyze this cover page for promotional document indication:

COVER PAGE TEXT: {cover_text}

TASK:
1. Is there a clear indication this is a promotional/marketing document?
2. What specific phrases indicate this?
3. Are there any variations or indirect mentions?
4. Consider: "document promotionnel", "à caractère promotionnel", 
   "marketing material", "promotional", etc.

REGULATORY CONTEXT:
- French/EU regulations require explicit promotional document labeling
- Must be visible and unambiguous on cover page

{rule_hints}

Respond with JSON:
{{
  "violation": true/false,
  "confidence": 0-100,
  "found_phrases": ["phrase1", "phrase2"],
  "reasoning": "explanation",
  "location": "where on page",
  "slide": "Cover Page",
  "rule": "Promotional document mention required",
  "message": "description of issue",
  "evidence": "specific text found"
}}""",
            required_vars=["cover_text"]
        )
        
        # Performance claims analysis
        self.templates["performance_claims"] = PromptTemplate(
            template="""Analyze this text for performance claims and context:

TEXT: {slide_text}

DISTINGUISH:
1. Historical fact: "The fund generated 5% returns in 2023"
2. Predictive claim: "The fund will deliver strong performance"  
3. Capability statement: "The fund aims to generate returns"
4. Example/illustration: "For example, a 5% return would mean..."

TASK:
- Identify the type of performance statement
- Determine if disclaimers are required
- Check if appropriate disclaimers are present
- Assess compliance risk level

{rule_hints}

Respond with JSON:
{{
  "violation": true/false,
  "performance_type": "historical|predictive|capability|example",
  "requires_disclaimer": true/false,
  "disclaimer_present": true/false,
  "disclaimer_location": "same_slide|different_slide|not_found",
  "confidence": 0-100,
  "evidence": "specific text found",
  "reasoning": "explanation of decision",
  "slide": "slide identifier",
  "rule": "rule description",
  "message": "violation description"
}}""",
            required_vars=["slide_text"]
        )
        
        # Fund name semantic matching
        self.templates["fund_name_match"] = PromptTemplate(
            template="""Compare these two fund names semantically:

PROSPECTUS FUND: {prospectus_fund_name}
DOCUMENT FUND: {doc_fund_name}

Do they refer to the same fund? Consider:
- Abbreviations (ODDO BHF vs Oddo Bank)
- Word order (Algo Trend US vs US Algo Trend)
- Missing/extra words (Fund, SICAV, etc.)
- Different naming conventions
- Legal entity variations

{rule_hints}

Respond with JSON:
{{
  "violation": true/false,
  "is_same_fund": true/false,
  "confidence": 0-100,
  "similarity_score": 0-100,
  "reasoning": "explanation",
  "differences_noted": ["list of differences"],
  "match_factors": ["what makes them similar"],
  "slide": "slide identifier",
  "rule": "Fund name must match prospectus",
  "message": "violation description",
  "evidence": "comparison details"
}}""",
            required_vars=["prospectus_fund_name", "doc_fund_name"]
        )
        
        # Country authorization
        self.templates["country_authorization"] = PromptTemplate(
            template="""Extract all country mentions from this document:

DOCUMENT TEXT: {document_text}

AUTHORIZED COUNTRIES: {authorized_countries}

TASK:
1. Identify all country names mentioned in the document
2. Check if each mentioned country is in the authorized list
3. Flag any unauthorized countries
4. Consider variations in country names (e.g., "USA" vs "United States")

{rule_hints}

Respond with JSON:
{{
  "violation": true/false,
  "countries_found": ["country1", "country2"],
  "unauthorized_countries": ["country1"],
  "confidence": 0-100,
  "reasoning": "explanation",
  "evidence": "where countries were mentioned",
  "slide": "slide identifier",
  "rule": "Fund only authorized in specific countries",
  "message": "violation description"
}}""",
            required_vars=["document_text", "authorized_countries"]
        )
        
        # Generic compliance check
        self.templates["generic_check"] = PromptTemplate(
            template="""Analyze this document for compliance:

DOCUMENT TEXT: {document_text}

CHECK TYPE: {check_type}
REQUIREMENTS: {requirements}

{rule_hints}

Respond with JSON:
{{
  "violation": true/false,
  "confidence": 0-100,
  "reasoning": "explanation",
  "evidence": "specific findings",
  "slide": "slide identifier",
  "rule": "rule description",
  "message": "violation description"
}}""",
            required_vars=["document_text", "check_type", "requirements"]
        )
        
        logger.info(f"Loaded {len(self.templates)} default prompt templates")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get template by name
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate or None
        """
        return self.templates.get(name)
    
    def add_template(self, name: str, template: PromptTemplate):
        """
        Add or update template
        
        Args:
            name: Template name
            template: PromptTemplate instance
        """
        self.templates[name] = template
        logger.info(f"Added template: {name}")
    
    def list_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())


class AIEngine:
    """
    Main AI Engine for compliance checking
    Provides unified interface for LLM calls with caching and prompt templates
    """
    
    def __init__(self, primary_client: LLMClient, fallback_client: Optional[LLMClient] = None,
                 cache_enabled: bool = True, cache_size: int = 1000, error_handler=None,
                 performance_monitor=None):
        """
        Initialize AI Engine
        
        Args:
            primary_client: Primary LLM client
            fallback_client: Optional fallback LLM client
            cache_enabled: Enable response caching
            cache_size: Maximum cache size
            error_handler: Optional error handler for retry and fallback logic
            performance_monitor: Optional performance monitor for tracking API usage
        """
        self.primary_client = primary_client
        self.fallback_client = fallback_client
        self.cache_enabled = cache_enabled
        self.cache = ResponseCache(max_size=cache_size) if cache_enabled else None
        self.prompt_library = PromptTemplateLibrary()
        self.error_handler = error_handler
        self.performance_monitor = performance_monitor
        
        logger.info(f"AIEngine initialized with {primary_client.get_provider().value}")
        if fallback_client:
            logger.info(f"Fallback client: {fallback_client.get_provider().value}")
        if error_handler:
            logger.info("Error handler integrated")
        if performance_monitor:
            logger.info("Performance monitoring integrated")
    
    def analyze(self, document: Dict, check_type: str, rule_hints: Dict, **kwargs) -> Optional[Dict]:
        """
        Analyze document for compliance using AI
        
        Args:
            document: Document data
            check_type: Type of compliance check
            rule_hints: Hints from rule-based analysis
            **kwargs: Additional parameters for specific checks
            
        Returns:
            Dict with analysis results or None on failure
        """
        try:
            # Get appropriate prompt template
            template = self.prompt_library.get_template(check_type)
            if not template:
                logger.warning(f"No template found for check type: {check_type}")
                template = self.prompt_library.get_template("generic_check")
                kwargs["check_type"] = check_type
                kwargs["requirements"] = "Analyze for compliance violations"
            
            # Add rule hints to kwargs
            rule_hints_text = self._format_rule_hints(rule_hints)
            kwargs["rule_hints"] = rule_hints_text
            
            # Render prompt
            prompt = template.render(**kwargs)
            
            # System message
            system_message = "You are a financial compliance expert. Analyze documents for regulatory compliance and return only valid JSON with no additional text."
            
            # Call LLM with caching
            response = self.call_with_cache(prompt, system_message)
            
            if response and response.parsed_json:
                return response.parsed_json
            elif response and response.content:
                # Try to extract JSON from content
                return {"response": response.content, "raw": response.content}
            
            return None
            
        except Exception as e:
            logger.error(f"AI analysis failed for {check_type}: {e}")
            return None
    
    def call_with_cache(self, prompt: str, system_message: str = "", **kwargs) -> Optional[AIResponse]:
        """
        Call LLM with caching support
        
        Args:
            prompt: User prompt
            system_message: System message
            **kwargs: Additional parameters
            
        Returns:
            AIResponse or None
        """
        # Check cache first
        cache_hit = False
        if self.cache_enabled and self.cache:
            cache_key = ResponseCache.generate_key(prompt, system_message, **kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                cache_hit = True
                # Record cached API call in performance monitor
                if self.performance_monitor and cached_response.provider:
                    self.performance_monitor.record_api_call(
                        provider=cached_response.provider.value,
                        tokens=cached_response.tokens_used or 0,
                        latency_ms=0,  # Cached calls have no latency
                        cached=True,
                        success=True
                    )
                return cached_response
        
        # Call primary client
        response = self._call_with_retry(self.primary_client, prompt, system_message, **kwargs)
        
        # Try fallback if primary fails
        if (not response or response.error) and self.fallback_client:
            logger.warning("Primary client failed, trying fallback")
            response = self._call_with_retry(self.fallback_client, prompt, system_message, **kwargs)
        
        # Record API call in performance monitor
        if self.performance_monitor and response:
            self.performance_monitor.record_api_call(
                provider=response.provider.value if response.provider else 'unknown',
                tokens=response.tokens_used or 500,  # Default estimate if not provided
                latency_ms=response.latency_ms or 0,
                cached=False,
                success=(not response.error)
            )
        
        # Cache successful response
        if response and not response.error and self.cache_enabled and self.cache:
            self.cache.set(cache_key, response)
        
        return response
    
    def _call_with_retry(self, client: LLMClient, prompt: str, system_message: str = "",
                        max_retries: int = 3, **kwargs) -> Optional[AIResponse]:
        """
        Call LLM client with retry logic
        
        Args:
            client: LLM client to use
            prompt: User prompt
            system_message: System message
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters
            
        Returns:
            AIResponse or None
        """
        for attempt in range(max_retries):
            try:
                response = client.call(prompt, system_message, **kwargs)
                
                if response and not response.error:
                    return response
                
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 1.0  # Exponential backoff
                    logger.warning(f"Retry attempt {attempt + 1} after {delay}s")
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 1.0)
        
        return None
    
    def _format_rule_hints(self, rule_hints: Dict) -> str:
        """
        Format rule hints for inclusion in prompt
        
        Args:
            rule_hints: Dict with rule-based analysis results
            
        Returns:
            Formatted string
        """
        if not rule_hints or not rule_hints.get("found"):
            return "RULE-BASED HINTS: No specific patterns detected by rules."
        
        hints = []
        if rule_hints.get("keywords"):
            hints.append(f"Keywords found: {', '.join(rule_hints['keywords'])}")
        if rule_hints.get("patterns"):
            hints.append(f"Patterns matched: {', '.join(rule_hints['patterns'])}")
        if rule_hints.get("confidence"):
            hints.append(f"Rule confidence: {rule_hints['confidence']}%")
        
        return "RULE-BASED HINTS: " + "; ".join(hints)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        if self.cache:
            return self.cache.get_hit_rate()
        return 0.0
    
    def clear_cache(self):
        """Clear response cache"""
        if self.cache:
            self.cache.clear()
    
    def invalidate_cache(self, pattern: Optional[str] = None, check_type: Optional[str] = None):
        """
        Invalidate cache entries
        
        Args:
            pattern: String pattern to match in cache keys
            check_type: Specific check type to invalidate
        """
        if self.cache:
            self.cache.invalidate(pattern=pattern, check_type=check_type)
    
    def reset_cache_metrics(self):
        """Reset cache monitoring metrics"""
        if self.cache:
            self.cache.reset_metrics()
    
    def add_prompt_template(self, name: str, template: PromptTemplate):
        """Add custom prompt template"""
        self.prompt_library.add_template(name, template)
    
    def list_templates(self) -> List[str]:
        """List available prompt templates"""
        return self.prompt_library.list_templates()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_ai_engine_from_env() -> Optional[AIEngine]:
    """
    Create AIEngine from environment variables
    
    Returns:
        AIEngine instance or None if no API keys available
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    tokenfactory_key = os.getenv('TOKENFACTORY_API_KEY')
    
    primary_client = None
    fallback_client = None
    
    # Try Token Factory first (preferred)
    if tokenfactory_key:
        try:
            config = LLMConfig(
                provider=LLMProvider.TOKEN_FACTORY,
                api_key=tokenfactory_key,
                model_name="hosted_vllm/Llama-3.1-70B-Instruct",
                base_url="https://tokenfactory.esprit.tn/api"
            )
            primary_client = TokenFactoryClient(config)
            logger.info("Primary client: Token Factory")
        except Exception as e:
            logger.warning(f"Failed to initialize Token Factory: {e}")
    
    # Try Gemini as primary or fallback
    if gemini_key:
        try:
            config = LLMConfig(
                provider=LLMProvider.GEMINI,
                api_key=gemini_key,
                model_name="gemini-2.5-flash"
            )
            gemini_client = GeminiClient(config)
            
            if primary_client:
                fallback_client = gemini_client
                logger.info("Fallback client: Gemini")
            else:
                primary_client = gemini_client
                logger.info("Primary client: Gemini")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
    
    if primary_client:
        return AIEngine(primary_client, fallback_client, cache_enabled=True, cache_size=1000)
    
    logger.error("No LLM clients available - check API keys in .env")
    return None


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("AI Engine - LLM Abstraction Layer")
    print("="*70)
    
    # Create engine from environment
    engine = create_ai_engine_from_env()
    
    if engine:
        print(f"\n✓ AI Engine initialized successfully")
        print(f"  Available templates: {', '.join(engine.list_templates())}")
        print(f"  Cache enabled: {engine.cache_enabled}")
        
        # Test call
        print("\n🧪 Testing AI call...")
        response = engine.call_with_cache(
            prompt="What is 2+2? Respond with JSON: {\"answer\": <number>}",
            system_message="You are a helpful assistant."
        )
        
        if response:
            print(f"  ✓ Response received ({response.latency_ms:.0f}ms)")
            print(f"  Provider: {response.provider.value if response.provider else 'Unknown'}")
            print(f"  Cached: {response.cached}")
            if response.parsed_json:
                print(f"  Parsed JSON: {response.parsed_json}")
        else:
            print("  ✗ No response")
        
        print(f"\n📊 Cache stats: {engine.get_cache_stats()}")
    else:
        print("\n✗ Failed to initialize AI Engine")
        print("  Check API keys in .env file")
    
    print("\n" + "="*70)
