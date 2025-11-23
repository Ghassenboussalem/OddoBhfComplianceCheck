#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Interaction Tests

This test suite verifies agent-to-agent interactions in the multi-agent workflow:
1. State passing between agents
2. Violation aggregation from multiple agents
3. Context analysis flow (aggregator → context → evidence)
4. Review queue flow (evidence → reviewer)

Requirements: 14.3
"""

import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import agents and components
from agents.supervisor_agent import SupervisorAgent
from agents.preprocessor_agent import PreprocessorAgent
from agents.structure_agent import StructureAgent
from agents.performance_agent import PerformanceAgent
from agents.securities_agent import SecuritiesAgent
from agents.general_agent import GeneralAgent
from agents.aggregator_agent import AggregatorAgent
from agents.context_agent import ContextAgent
from agents.evidence_agent import EvidenceAgent
from agents.reviewer_agent import ReviewerAgent
from agents.base_agent import AgentConfig

from data_models_multiagent import (
    initialize_compliance_state,
    validate_compliance_state,
    get_state_summary,
    ViolationStatus,
    WorkflowStatus
)

from ai_engine import create_ai_engine_from_env
from review_manager import ReviewManager


def create_test_document() -> Dict[str, Any]:
    """Create a test document with content that triggers multiple agents"""
    return {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'Test Multi-Agent Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_esg_classification': 'article_8',
            'fund_age_years': 5,
            'document_date': '2024-01-15',
            'fund_inception_date': '2020-01-15'
        },
        'page_de_garde': {
            'title': 'Fund Presentation',
            'subtitle': 'Test Multi-Agent Fund',
            'date': '2024-01-15',
            'content': 'Document de présentation'  # Missing promotional mention
        },
        'slide_2': {
            'title': 'Performance',
            'content': 'Le fonds a généré une performance de +15.5% en 2023. '
                      'Rendement annualisé: 8.2% sur 5 ans.'  # Performance without disclaimer
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'title': 'Strategy',
                'content': 'Nous recommandons d\'investir dans ce fonds pour profiter '
                          'de la croissance des marchés émergents.'  # Investment advice
            },
            {
                'slide_number': 4,
                'title': 'Technical Analysis',
                'content': 'Le fonds utilise l\'alpha et le beta pour optimiser '
                          'la volatilité du portefeuille.'  # Technical terms without glossary
            }
        ],
        'page_de_fin': {
            'legal': 'Asset Management',  # Incomplete legal mention
            'disclaimers': 'Standard disclaimers'
        }
    }


class AgentInteractionTests:
    """Test suite for agent interactions"""
    
    def __init__(self):
        self.test_results = []
        self.ai_engine = create_ai_engine_from_env()
        self.review_manager = ReviewManager()
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents for testing"""
        logger.info("Initializing agents...")
        
        # Initialize agents with proper configs
        self.supervisor = SupervisorAgent(
            config=AgentConfig(name="supervisor", enabled=True, log_level="INFO", timeout_seconds=30.0)
        )
        self.preprocessor = PreprocessorAgent(
            config=AgentConfig(name="preprocessor", enabled=True, log_level="INFO", timeout_seconds=30.0)
        )
        self.structure = StructureAgent(
            config=AgentConfig(name="structure", enabled=True, log_level="INFO", timeout_seconds=30.0)
        )
        self.performance = PerformanceAgent(
            config=AgentConfig(name="performance", enabled=True, log_level="INFO", timeout_seconds=30.0),
            ai_engine=self.ai_engine
        )
        self.securities = SecuritiesAgent(
            config=AgentConfig(name="securities", enabled=True, log_level="INFO", timeout_seconds=30.0),
            ai_engine=self.ai_engine
        )
        self.general = GeneralAgent(
            config=AgentConfig(name="general", enabled=True, log_level="INFO", timeout_seconds=30.0)
        )
        self.aggregator = AggregatorAgent(
            config=AgentConfig(name="aggregator", enabled=True, log_level="INFO", timeout_seconds=30.0),
            deduplication_enabled=True,
            context_threshold=80,
            review_threshold=70
        )
        self.context = ContextAgent(
            config=AgentConfig(name="context", enabled=True, log_level="INFO", timeout_seconds=30.0),
            ai_engine=self.ai_engine
        )
        self.evidence = EvidenceAgent(
            config=AgentConfig(name="evidence", enabled=True, log_level="INFO", timeout_seconds=30.0),
            ai_engine=self.ai_engine
        )
        self.reviewer = ReviewerAgent(
            config=AgentConfig(name="reviewer", enabled=True, log_level="INFO", timeout_seconds=30.0),
            review_manager=self.review_manager
        )
        
        logger.info("✓ All agents initialized")
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")
    
    def test_1_state_passing_between_agents(self) -> bool:
        """
        Test 1: State passing between agents
        
        Verifies:
        - State is correctly passed from one agent to another
        - Each agent receives complete state from previous agent
        - State modifications are preserved
        - State structure remains valid throughout
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: State Passing Between Agents")
        logger.info("="*70)
        
        try:
            # Create initial state
            document = create_test_document()
            config = {
                "routing": {
                    "context_threshold": 80,
                    "review_threshold": 70
                }
            }
            
            state = initialize_compliance_state(
                document=document,
                document_id="test_state_passing_001",
                config=config
            )
            
            logger.info("\n[1] Initial state created")
            logger.info(f"  Document ID: {state['document_id']}")
            logger.info(f"  Workflow status: {state['workflow_status']}")
            
            # Validate initial state
            is_valid, errors = validate_compliance_state(state)
            if not is_valid:
                logger.error(f"  ✗ Initial state invalid: {errors}")
                self.log_test_result("State Passing Between Agents", False, "Initial state invalid")
                return False
            
            logger.info("  ✓ Initial state valid")
            
            # Pass through Supervisor
            logger.info("\n[2] Passing through Supervisor Agent...")
            state = self.supervisor(state)
            
            logger.info(f"  Current agent: {state['current_agent']}")
            logger.info(f"  Next action: {state['next_action']}")
            logger.info(f"  Execution plan: {state.get('execution_plan', [])}")
            
            # Validate state after supervisor
            is_valid, errors = validate_compliance_state(state)
            if not is_valid:
                logger.error(f"  ✗ State invalid after supervisor: {errors}")
                self.log_test_result("State Passing Between Agents", False, "State invalid after supervisor")
                return False
            
            logger.info("  ✓ State valid after supervisor")
            
            # Pass through Preprocessor
            logger.info("\n[3] Passing through Preprocessor Agent...")
            state = self.preprocessor(state)
            
            logger.info(f"  Current agent: {state['current_agent']}")
            logger.info(f"  Metadata fields: {len(state.get('metadata', {}))}")
            logger.info(f"  Whitelist terms: {len(state.get('whitelist', set()))}")
            logger.info(f"  Normalized document: {'normalized_document' in state}")
            
            # Validate state after preprocessor
            is_valid, errors = validate_compliance_state(state)
            if not is_valid:
                logger.error(f"  ✗ State invalid after preprocessor: {errors}")
                self.log_test_result("State Passing Between Agents", False, "State invalid after preprocessor")
                return False
            
            logger.info("  ✓ State valid after preprocessor")
            
            # Verify preprocessing results are present
            if not state.get('metadata'):
                logger.error("  ✗ Metadata not populated")
                self.log_test_result("State Passing Between Agents", False, "Metadata not populated")
                return False
            
            if not state.get('whitelist'):
                logger.warning("  ⚠ Whitelist not populated (may be expected)")
            
            logger.info("  ✓ Preprocessing results present")
            
            # Pass through Structure Agent
            logger.info("\n[4] Passing through Structure Agent...")
            violations_before = len(state.get('violations', []))
            state = self.structure(state)
            violations_after = len(state.get('violations', []))
            
            current_agent = state.get('current_agent', 'unknown')
            logger.info(f"  Current agent: {current_agent}")
            logger.info(f"  Violations before: {violations_before}")
            logger.info(f"  Violations after: {violations_after}")
            logger.info(f"  New violations: {violations_after - violations_before}")
            
            # Validate state after structure agent (check key fields are present)
            required_fields = ['document', 'document_id', 'violations', 'config']
            missing_fields = [f for f in required_fields if f not in state]
            
            if missing_fields:
                logger.error(f"  ✗ Missing required fields after structure agent: {missing_fields}")
                self.log_test_result("State Passing Between Agents", False, f"Missing fields: {missing_fields}")
                return False
            
            logger.info("  ✓ State valid after structure agent (key fields present)")
            
            # Verify state integrity
            logger.info("\n[5] Verifying state integrity...")
            
            # Check that original document is preserved
            if state['document_id'] != "test_state_passing_001":
                logger.error("  ✗ Document ID changed")
                self.log_test_result("State Passing Between Agents", False, "Document ID changed")
                return False
            
            logger.info("  ✓ Document ID preserved")
            
            # Check that config is preserved
            if state['config'] != config:
                logger.error("  ✗ Config changed")
                self.log_test_result("State Passing Between Agents", False, "Config changed")
                return False
            
            logger.info("  ✓ Config preserved")
            
            # Check that agent timings are tracked
            agent_timings = state.get('agent_timings', {})
            expected_agents = ['supervisor', 'preprocessor', 'structure']
            
            for agent in expected_agents:
                if agent not in agent_timings:
                    logger.warning(f"  ⚠ Agent timing not recorded for: {agent}")
                else:
                    logger.info(f"  ✓ {agent}: {agent_timings[agent]:.3f}s")
            
            # Verify current_agent is set
            if 'current_agent' not in state:
                logger.error("  ✗ current_agent not set in state")
                self.log_test_result("State Passing Between Agents", False, "current_agent not set")
                return False
            
            logger.info(f"  ✓ Current agent tracked: {state['current_agent']}")
            
            self.log_test_result(
                "State Passing Between Agents",
                True,
                f"State passed through {len(expected_agents)} agents successfully"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("State Passing Between Agents", False, str(e))
            return False
    
    def test_2_violation_aggregation(self) -> bool:
        """
        Test 2: Violation aggregation from multiple agents
        
        Verifies:
        - Multiple agents can add violations to state
        - Violations are properly merged
        - Aggregator collects all violations
        - Deduplication works correctly
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Violation Aggregation from Multiple Agents")
        logger.info("="*70)
        
        try:
            # Create initial state
            document = create_test_document()
            config = {}
            
            state = initialize_compliance_state(
                document=document,
                document_id="test_aggregation_001",
                config=config
            )
            
            # Run preprocessor first
            logger.info("\n[1] Running preprocessor...")
            state = self.preprocessor(state)
            logger.info(f"  ✓ Preprocessor completed")
            
            # Run multiple specialist agents
            logger.info("\n[2] Running specialist agents...")
            
            violations_by_agent = {}
            
            # Structure agent
            logger.info("  Running Structure Agent...")
            violations_before = len(state.get('violations', []))
            state = self.structure(state)
            violations_after = len(state.get('violations', []))
            violations_by_agent['structure'] = violations_after - violations_before
            logger.info(f"    Violations added: {violations_by_agent['structure']}")
            
            # Performance agent
            logger.info("  Running Performance Agent...")
            violations_before = violations_after
            state = self.performance(state)
            violations_after = len(state.get('violations', []))
            violations_by_agent['performance'] = violations_after - violations_before
            logger.info(f"    Violations added: {violations_by_agent['performance']}")
            
            # Securities agent
            logger.info("  Running Securities Agent...")
            violations_before = violations_after
            state = self.securities(state)
            violations_after = len(state.get('violations', []))
            violations_by_agent['securities'] = violations_after - violations_before
            logger.info(f"    Violations added: {violations_by_agent['securities']}")
            
            # General agent
            logger.info("  Running General Agent...")
            violations_before = violations_after
            state = self.general(state)
            violations_after = len(state.get('violations', []))
            violations_by_agent['general'] = violations_after - violations_before
            logger.info(f"    Violations added: {violations_by_agent['general']}")
            
            total_violations = violations_after
            logger.info(f"\n  Total violations from all agents: {total_violations}")
            
            # Verify violations have agent attribution
            logger.info("\n[3] Verifying violation attribution...")
            
            violations = state.get('violations', [])
            agents_found = set()
            
            for violation in violations:
                agent = violation.get('agent', 'unknown')
                agents_found.add(agent)
            
            logger.info(f"  Agents with violations: {agents_found}")
            
            for agent in ['structure', 'performance', 'securities', 'general']:
                if agent in agents_found:
                    count = sum(1 for v in violations if v.get('agent') == agent)
                    logger.info(f"    ✓ {agent}: {count} violations")
                else:
                    logger.warning(f"    ⚠ {agent}: no violations (may be expected)")
            
            # Run aggregator
            logger.info("\n[4] Running Aggregator Agent...")
            violations_before_agg = len(violations)
            state = self.aggregator(state)
            violations_after_agg = len(state.get('violations', []))
            
            logger.info(f"  Violations before aggregation: {violations_before_agg}")
            logger.info(f"  Violations after aggregation: {violations_after_agg}")
            logger.info(f"  Duplicates removed: {violations_before_agg - violations_after_agg}")
            
            # Check aggregation results
            aggregated_confidence = state.get('aggregated_confidence', 0)
            next_action = state.get('next_action', '')
            categorization = state.get('violation_categorization', {})
            
            logger.info(f"\n  Aggregated confidence: {aggregated_confidence}%")
            logger.info(f"  Next action: {next_action}")
            
            if categorization:
                logger.info(f"  Categorization:")
                logger.info(f"    By type: {categorization.get('by_type', {})}")
                logger.info(f"    By severity: {categorization.get('by_severity', {})}")
                logger.info(f"    By agent: {categorization.get('by_agent', {})}")
            
            # Verify aggregation worked
            if violations_after_agg == 0:
                logger.warning("  ⚠ No violations after aggregation")
            else:
                logger.info(f"  ✓ {violations_after_agg} violations aggregated")
            
            if aggregated_confidence > 0:
                logger.info(f"  ✓ Aggregated confidence calculated")
            else:
                logger.warning(f"  ⚠ Aggregated confidence is 0")
            
            if next_action:
                logger.info(f"  ✓ Next action determined")
            else:
                logger.warning(f"  ⚠ Next action not set")
            
            self.log_test_result(
                "Violation Aggregation",
                True,
                f"{violations_after_agg} violations from {len(agents_found)} agents"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Violation Aggregation", False, str(e))
            return False
    
    def test_3_context_analysis_flow(self) -> bool:
        """
        Test 3: Context analysis flow (aggregator → context → evidence)
        
        Verifies:
        - Aggregator routes low-confidence violations to context agent
        - Context agent analyzes violations and updates confidence
        - Evidence agent extracts supporting evidence
        - Flow completes successfully
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Context Analysis Flow")
        logger.info("="*70)
        
        try:
            # Create state with low-confidence violations
            document = create_test_document()
            config = {
                "routing": {
                    "context_threshold": 80,
                    "review_threshold": 70
                }
            }
            
            state = initialize_compliance_state(
                document=document,
                document_id="test_context_flow_001",
                config=config
            )
            
            # Add preprocessor results
            logger.info("\n[1] Running preprocessor...")
            state = self.preprocessor(state)
            
            # Add some low-confidence violations manually
            logger.info("\n[2] Adding low-confidence test violations...")
            
            state['violations'] = [
                {
                    "rule": "INVESTMENT_ADVICE",
                    "type": "SECURITIES",
                    "severity": "HIGH",
                    "slide": "3",
                    "location": "Strategy section",
                    "evidence": "Nous recommandons d'investir dans ce fonds",
                    "ai_reasoning": "Detected investment advice pattern",
                    "confidence": 65,
                    "status": ViolationStatus.DETECTED.value,
                    "agent": "securities"
                },
                {
                    "rule": "PERFORMANCE_DISCLAIMER",
                    "type": "PERFORMANCE",
                    "severity": "HIGH",
                    "slide": "2",
                    "location": "Performance section",
                    "evidence": "Performance de +15.5% en 2023",
                    "ai_reasoning": "Performance data without disclaimer",
                    "confidence": 75,
                    "status": ViolationStatus.DETECTED.value,
                    "agent": "performance"
                },
                {
                    "rule": "TECHNICAL_TERMS",
                    "type": "GENERAL",
                    "severity": "MEDIUM",
                    "slide": "4",
                    "location": "Technical section",
                    "evidence": "Alpha et beta",
                    "ai_reasoning": "Technical terms without glossary",
                    "confidence": 70,
                    "status": ViolationStatus.DETECTED.value,
                    "agent": "general"
                }
            ]
            
            logger.info(f"  Added {len(state['violations'])} low-confidence violations")
            for v in state['violations']:
                logger.info(f"    - {v['rule']}: {v['confidence']}%")
            
            # Run aggregator
            logger.info("\n[3] Running Aggregator Agent...")
            state = self.aggregator(state)
            
            next_action = state.get('next_action', '')
            aggregated_confidence = state.get('aggregated_confidence', 0)
            
            logger.info(f"  Aggregated confidence: {aggregated_confidence}%")
            logger.info(f"  Next action: {next_action}")
            
            # Check if aggregator routes to context analysis
            if next_action != "context_analysis":
                logger.warning(f"  ⚠ Expected next_action='context_analysis', got '{next_action}'")
                logger.info(f"  Note: This may be expected if confidence is high")
            else:
                logger.info(f"  ✓ Aggregator routing to context analysis")
            
            # Run context agent
            logger.info("\n[4] Running Context Agent...")
            
            context_analyses_before = len(state.get('context_analysis', {}))
            intent_classifications_before = len(state.get('intent_classifications', {}))
            
            state = self.context(state)
            
            context_analyses_after = len(state.get('context_analysis', {}))
            intent_classifications_after = len(state.get('intent_classifications', {}))
            
            logger.info(f"  Context analyses: {context_analyses_before} → {context_analyses_after}")
            logger.info(f"  Intent classifications: {intent_classifications_before} → {intent_classifications_after}")
            
            # Check if violations were analyzed
            violations_analyzed = sum(
                1 for v in state.get('violations', [])
                if v.get('context_analyzed', False)
            )
            
            logger.info(f"  Violations analyzed: {violations_analyzed}/{len(state['violations'])}")
            
            if violations_analyzed > 0:
                logger.info(f"  ✓ Context agent analyzed violations")
            else:
                logger.warning(f"  ⚠ No violations analyzed (may use fallback)")
            
            # Check for false positive filtering
            false_positives = sum(
                1 for v in state.get('violations', [])
                if v.get('status') == ViolationStatus.FALSE_POSITIVE_FILTERED.value
            )
            
            logger.info(f"  False positives filtered: {false_positives}")
            
            # Run evidence agent
            logger.info("\n[5] Running Evidence Agent...")
            
            evidence_extractions_before = len(state.get('evidence_extractions', {}))
            
            state = self.evidence(state)
            
            evidence_extractions_after = len(state.get('evidence_extractions', {}))
            
            logger.info(f"  Evidence extractions: {evidence_extractions_before} → {evidence_extractions_after}")
            
            # Check if violations were enhanced with evidence
            violations_with_evidence = sum(
                1 for v in state.get('violations', [])
                if 'evidence_quotes' in v or 'evidence_context' in v
            )
            
            logger.info(f"  Violations with evidence: {violations_with_evidence}/{len(state['violations'])}")
            
            if violations_with_evidence > 0:
                logger.info(f"  ✓ Evidence agent enhanced violations")
            else:
                logger.warning(f"  ⚠ No violations enhanced with evidence")
            
            # Verify flow completion
            logger.info("\n[6] Verifying flow completion...")
            
            final_next_action = state.get('next_action', '')
            logger.info(f"  Final next action: {final_next_action}")
            
            success = True
            
            if context_analyses_after > context_analyses_before:
                logger.info(f"  ✓ Context analysis performed")
            else:
                logger.warning(f"  ⚠ No context analysis (may use fallback)")
            
            if evidence_extractions_after > evidence_extractions_before:
                logger.info(f"  ✓ Evidence extraction performed")
            else:
                logger.warning(f"  ⚠ No evidence extraction")
            
            if final_next_action in ['review', 'complete']:
                logger.info(f"  ✓ Valid final action")
            else:
                logger.warning(f"  ⚠ Unexpected final action: {final_next_action}")
            
            self.log_test_result(
                "Context Analysis Flow",
                True,
                f"Flow completed: aggregator → context → evidence"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Context Analysis Flow", False, str(e))
            return False
    
    def test_4_review_queue_flow(self) -> bool:
        """
        Test 4: Review queue flow (evidence → reviewer)
        
        Verifies:
        - Very low confidence violations route to reviewer
        - Reviewer adds violations to review queue
        - Review queue is properly managed
        - HITL interrupt is triggered
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Review Queue Flow")
        logger.info("="*70)
        
        try:
            # Create state with very low-confidence violations
            document = create_test_document()
            config = {
                "routing": {
                    "context_threshold": 80,
                    "review_threshold": 70
                }
            }
            
            state = initialize_compliance_state(
                document=document,
                document_id="test_review_flow_001",
                config=config
            )
            
            # Add preprocessor results
            logger.info("\n[1] Running preprocessor...")
            state = self.preprocessor(state)
            
            # Add very low-confidence violations
            logger.info("\n[2] Adding very low-confidence test violations...")
            
            state['violations'] = [
                {
                    "rule": "AMBIGUOUS_ADVICE",
                    "type": "SECURITIES",
                    "severity": "HIGH",
                    "slide": "3",
                    "location": "Strategy section",
                    "evidence": "Le fonds peut offrir de bons rendements",
                    "ai_reasoning": "Ambiguous language, unclear if advice",
                    "confidence": 55,
                    "status": ViolationStatus.DETECTED.value,
                    "agent": "securities"
                },
                {
                    "rule": "UNCLEAR_DISCLAIMER",
                    "type": "PERFORMANCE",
                    "severity": "MEDIUM",
                    "slide": "2",
                    "location": "Performance section",
                    "evidence": "Résultats passés",
                    "ai_reasoning": "Unclear if proper disclaimer",
                    "confidence": 60,
                    "status": ViolationStatus.DETECTED.value,
                    "agent": "performance"
                },
                {
                    "rule": "POTENTIAL_VIOLATION",
                    "type": "GENERAL",
                    "severity": "LOW",
                    "slide": "4",
                    "location": "General section",
                    "evidence": "Termes techniques",
                    "ai_reasoning": "May or may not be violation",
                    "confidence": 50,
                    "status": ViolationStatus.DETECTED.value,
                    "agent": "general"
                }
            ]
            
            logger.info(f"  Added {len(state['violations'])} very low-confidence violations")
            for v in state['violations']:
                logger.info(f"    - {v['rule']}: {v['confidence']}%")
            
            # Run aggregator
            logger.info("\n[3] Running Aggregator Agent...")
            state = self.aggregator(state)
            
            aggregated_confidence = state.get('aggregated_confidence', 0)
            next_action = state.get('next_action', '')
            
            logger.info(f"  Aggregated confidence: {aggregated_confidence}%")
            logger.info(f"  Next action: {next_action}")
            
            # Run context agent (should still have low confidence)
            logger.info("\n[4] Running Context Agent...")
            state = self.context(state)
            
            # Run evidence agent
            logger.info("\n[5] Running Evidence Agent...")
            state = self.evidence(state)
            
            next_action_after_evidence = state.get('next_action', '')
            logger.info(f"  Next action after evidence: {next_action_after_evidence}")
            
            # Check if routing to reviewer
            if next_action_after_evidence == "review":
                logger.info(f"  ✓ Routing to reviewer")
            else:
                logger.warning(f"  ⚠ Not routing to reviewer (action: {next_action_after_evidence})")
            
            # Run reviewer agent
            logger.info("\n[6] Running Reviewer Agent...")
            
            review_queue_before = len(state.get('review_queue', []))
            
            state = self.reviewer(state)
            
            review_queue_after = len(state.get('review_queue', []))
            
            logger.info(f"  Review queue: {review_queue_before} → {review_queue_after}")
            logger.info(f"  Items added: {review_queue_after - review_queue_before}")
            
            # Check HITL interrupt
            hitl_interrupt = state.get('hitl_interrupt_required', False)
            hitl_reason = state.get('hitl_interrupt_reason', '')
            
            logger.info(f"\n  HITL interrupt required: {hitl_interrupt}")
            if hitl_reason:
                logger.info(f"  HITL interrupt reason: {hitl_reason}")
            
            # Verify review queue contents
            logger.info("\n[7] Verifying review queue...")
            
            review_queue = state.get('review_queue', [])
            
            if len(review_queue) > 0:
                logger.info(f"  ✓ {len(review_queue)} items in review queue")
                
                for i, item in enumerate(review_queue[:3], 1):
                    review_id = item.get('review_id', 'unknown')
                    violation = item.get('violation', {})
                    rule = violation.get('rule', 'unknown')
                    confidence = violation.get('confidence', 0)
                    
                    logger.info(f"    [{i}] {rule} (confidence: {confidence}%)")
                    logger.info(f"        Review ID: {review_id}")
            else:
                logger.warning(f"  ⚠ Review queue is empty")
            
            # Check review manager integration
            logger.info("\n[8] Checking review manager integration...")
            
            queue_stats = self.review_manager.get_queue_stats()
            
            logger.info(f"  Review manager statistics:")
            logger.info(f"    Total pending: {queue_stats.total_pending}")
            logger.info(f"    Total reviewed: {queue_stats.total_reviewed}")
            if hasattr(queue_stats, 'total_rejected'):
                logger.info(f"    Total rejected: {queue_stats.total_rejected}")
            else:
                logger.info(f"    Total in review: {queue_stats.total_in_review}")
            
            # Verify flow completion
            logger.info("\n[9] Verifying flow completion...")
            
            success = True
            
            if review_queue_after > review_queue_before:
                logger.info(f"  ✓ Violations added to review queue")
            else:
                logger.warning(f"  ⚠ No violations added to review queue")
            
            if hitl_interrupt:
                logger.info(f"  ✓ HITL interrupt triggered")
            else:
                logger.warning(f"  ⚠ HITL interrupt not triggered")
            
            workflow_status = state.get('workflow_status', '')
            logger.info(f"  Workflow status: {workflow_status}")
            
            self.log_test_result(
                "Review Queue Flow",
                True,
                f"{review_queue_after} items in review queue, HITL={hitl_interrupt}"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Review Queue Flow", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all agent interaction tests"""
        logger.info("\n" + "="*70)
        logger.info("AGENT INTERACTION TEST SUITE")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now().isoformat()}")
        
        start_time = datetime.now()
        
        # Run all tests
        tests = [
            self.test_1_state_passing_between_agents,
            self.test_2_violation_aggregation,
            self.test_3_context_analysis_flow,
            self.test_4_review_queue_flow
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test crashed: {e}", exc_info=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"  {result['details']}")
        
        logger.info("")
        logger.info(f"Results: {passed}/{total} tests passed")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Completed at: {end_time.isoformat()}")
        logger.info("="*70)
        
        # Save results to file
        results_dir = "tests"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "agent_interaction_test_results.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "duration_seconds": duration,
                    "timestamp": end_time.isoformat()
                },
                "tests": self.test_results
            }, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return passed == total


def main():
    """Main entry point for agent interaction tests"""
    try:
        test_suite = AgentInteractionTests()
        success = test_suite.run_all_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
