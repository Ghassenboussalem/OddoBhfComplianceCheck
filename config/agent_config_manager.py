#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Configuration Manager - Multi-Agent System Configuration
Manages agent-specific configurations, validation, and runtime updates
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent types in the multi-agent system"""
    SUPERVISOR = "supervisor"
    PREPROCESSOR = "preprocessor"
    STRUCTURE = "structure"
    PERFORMANCE = "performance"
    SECURITIES = "securities"
    GENERAL = "general"
    PROSPECTUS = "prospectus"
    REGISTRATION = "registration"
    ESG = "esg"
    AGGREGATOR = "aggregator"
    CONTEXT = "context"
    EVIDENCE = "evidence"
    REVIEWER = "reviewer"
    FEEDBACK = "feedback"


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    enabled: bool = True
    timeout_seconds: int = 30
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    parallel_tool_execution: bool = False
    conditional: bool = False
    condition: Optional[str] = None
    confidence_threshold: Optional[int] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system"""
    enabled: bool = False
    parallel_execution: bool = True
    max_parallel_agents: int = 4
    agent_timeout_seconds: int = 30
    checkpoint_interval: int = 5
    state_persistence: bool = True
    checkpoint_db_path: str = "./checkpoints/compliance_workflow.db"
    enable_workflow_visualization: bool = True
    enable_agent_logging: bool = True


@dataclass
class RoutingConfig:
    """Configuration for agent routing logic"""
    context_threshold: int = 80
    review_threshold: int = 70
    skip_context_if_high_confidence: bool = True
    skip_review_if_high_confidence: bool = True
    parallel_specialist_agents: List[str] = field(default_factory=lambda: [
        "structure", "performance", "securities", "general"
    ])
    conditional_agents: List[str] = field(default_factory=lambda: [
        "prospectus", "registration", "esg"
    ])
    sequential_flow: List[str] = field(default_factory=lambda: [
        "supervisor", "preprocessor", "aggregator", "context", "evidence", "reviewer"
    ])


@dataclass
class StateManagementConfig:
    """Configuration for state management"""
    checkpoint_enabled: bool = True
    checkpoint_interval_agents: int = 5
    save_intermediate_states: bool = True
    state_history_max_size: int = 100
    enable_state_validation: bool = True
    auto_cleanup_old_checkpoints: bool = True
    checkpoint_retention_days: int = 7


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling"""
    agent_failure_strategy: str = "continue"  # continue, stop, retry
    max_agent_retries: int = 2
    retry_delay_seconds: float = 1.0
    fallback_to_rules_on_ai_failure: bool = True
    partial_results_on_failure: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability"""
    enabled: bool = True
    log_agent_invocations: bool = True
    log_level: str = "INFO"
    track_execution_times: bool = True
    track_success_rates: bool = True
    track_cache_hits: bool = True
    track_api_calls: bool = True
    metrics_export_enabled: bool = True
    metrics_export_path: str = "./monitoring/metrics/"
    dashboard_enabled: bool = False
    alert_on_failures: bool = True
    alert_threshold_failure_rate: float = 0.2


class AgentConfigManager:
    """
    Manages configuration for multi-agent compliance system
    Supports loading from file, validation, and runtime updates
    """
    
    def __init__(self, config_path: str = "hybrid_config.json"):
        """
        Initialize agent configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.multi_agent_config: Optional[MultiAgentConfig] = None
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.routing_config: Optional[RoutingConfig] = None
        self.state_management_config: Optional[StateManagementConfig] = None
        self.error_handling_config: Optional[ErrorHandlingConfig] = None
        self.monitoring_config: Optional[MonitoringConfig] = None
        
        self._load_config()
        self._validate_config()
        logger.info("AgentConfigManager initialized")
    
    def _load_config(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            self._load_defaults()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Load multi-agent config
            multi_agent_data = config_data.get('multi_agent', {})
            self.multi_agent_config = MultiAgentConfig(**multi_agent_data)
            
            # Load agent configs
            agents_data = config_data.get('agents', {})
            for agent_name, agent_data in agents_data.items():
                if isinstance(agent_data, dict):
                    # Separate known fields from custom settings
                    known_fields = {
                        'enabled', 'timeout_seconds', 'max_retries', 'retry_delay_seconds',
                        'parallel_tool_execution', 'conditional', 'condition', 'confidence_threshold'
                    }
                    
                    agent_config_data = {}
                    custom_settings = {}
                    
                    for key, value in agent_data.items():
                        if key in known_fields:
                            agent_config_data[key] = value
                        else:
                            custom_settings[key] = value
                    
                    agent_config_data['custom_settings'] = custom_settings
                    self.agent_configs[agent_name] = AgentConfig(**agent_config_data)
            
            # Load routing config
            routing_data = config_data.get('routing', {})
            self.routing_config = RoutingConfig(**routing_data)
            
            # Load state management config
            state_mgmt_data = config_data.get('state_management', {})
            self.state_management_config = StateManagementConfig(**state_mgmt_data)
            
            # Load error handling config
            error_handling_data = config_data.get('error_handling', {})
            self.error_handling_config = ErrorHandlingConfig(**error_handling_data)
            
            # Load monitoring config
            monitoring_data = config_data.get('monitoring', {})
            self.monitoring_config = MonitoringConfig(**monitoring_data)
            
            logger.info(f"Loaded agent configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.multi_agent_config = MultiAgentConfig()
        self.routing_config = RoutingConfig()
        self.state_management_config = StateManagementConfig()
        self.error_handling_config = ErrorHandlingConfig()
        self.monitoring_config = MonitoringConfig()
        
        # Load default agent configs
        for agent_type in AgentType:
            self.agent_configs[agent_type.value] = AgentConfig()
        
        logger.info("Loaded default agent configuration")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate multi-agent config
        if self.multi_agent_config:
            if self.multi_agent_config.max_parallel_agents < 1:
                errors.append(f"max_parallel_agents must be >= 1, got {self.multi_agent_config.max_parallel_agents}")
            
            if self.multi_agent_config.agent_timeout_seconds < 1:
                errors.append(f"agent_timeout_seconds must be >= 1, got {self.multi_agent_config.agent_timeout_seconds}")
            
            if self.multi_agent_config.checkpoint_interval < 1:
                errors.append(f"checkpoint_interval must be >= 1, got {self.multi_agent_config.checkpoint_interval}")
        
        # Validate agent configs
        for agent_name, agent_config in self.agent_configs.items():
            if agent_config.timeout_seconds < 1:
                errors.append(f"{agent_name}.timeout_seconds must be >= 1, got {agent_config.timeout_seconds}")
            
            if agent_config.max_retries < 0:
                errors.append(f"{agent_name}.max_retries must be >= 0, got {agent_config.max_retries}")
            
            if agent_config.retry_delay_seconds < 0:
                errors.append(f"{agent_name}.retry_delay_seconds must be >= 0, got {agent_config.retry_delay_seconds}")
            
            if agent_config.confidence_threshold is not None:
                if not 0 <= agent_config.confidence_threshold <= 100:
                    errors.append(f"{agent_name}.confidence_threshold must be 0-100, got {agent_config.confidence_threshold}")
        
        # Validate routing config
        if self.routing_config:
            if not 0 <= self.routing_config.context_threshold <= 100:
                errors.append(f"routing.context_threshold must be 0-100, got {self.routing_config.context_threshold}")
            
            if not 0 <= self.routing_config.review_threshold <= 100:
                errors.append(f"routing.review_threshold must be 0-100, got {self.routing_config.review_threshold}")
        
        # Validate state management config
        if self.state_management_config:
            if self.state_management_config.checkpoint_interval_agents < 1:
                errors.append(f"state_management.checkpoint_interval_agents must be >= 1")
            
            if self.state_management_config.state_history_max_size < 1:
                errors.append(f"state_management.state_history_max_size must be >= 1")
            
            if self.state_management_config.checkpoint_retention_days < 1:
                errors.append(f"state_management.checkpoint_retention_days must be >= 1")
        
        # Validate error handling config
        if self.error_handling_config:
            valid_strategies = ["continue", "stop", "retry"]
            if self.error_handling_config.agent_failure_strategy not in valid_strategies:
                errors.append(f"error_handling.agent_failure_strategy must be one of {valid_strategies}")
            
            if self.error_handling_config.max_agent_retries < 0:
                errors.append(f"error_handling.max_agent_retries must be >= 0")
            
            if self.error_handling_config.circuit_breaker_threshold < 1:
                errors.append(f"error_handling.circuit_breaker_threshold must be >= 1")
        
        # Validate monitoring config
        if self.monitoring_config:
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.monitoring_config.log_level not in valid_log_levels:
                errors.append(f"monitoring.log_level must be one of {valid_log_levels}")
            
            if not 0 <= self.monitoring_config.alert_threshold_failure_rate <= 1:
                errors.append(f"monitoring.alert_threshold_failure_rate must be 0-1")
        
        if errors:
            error_msg = "Agent configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Agent configuration validation passed")
    
    def is_multi_agent_enabled(self) -> bool:
        """Check if multi-agent system is enabled"""
        return self.multi_agent_config.enabled if self.multi_agent_config else False
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """
        Check if a specific agent is enabled
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            True if agent is enabled
        """
        agent_config = self.agent_configs.get(agent_name)
        return agent_config.enabled if agent_config else False
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get configuration for a specific agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentConfig or None if not found
        """
        return self.agent_configs.get(agent_name)
    
    def get_multi_agent_config(self) -> Optional[MultiAgentConfig]:
        """Get multi-agent system configuration"""
        return self.multi_agent_config
    
    def get_routing_config(self) -> Optional[RoutingConfig]:
        """Get routing configuration"""
        return self.routing_config
    
    def get_state_management_config(self) -> Optional[StateManagementConfig]:
        """Get state management configuration"""
        return self.state_management_config
    
    def get_error_handling_config(self) -> Optional[ErrorHandlingConfig]:
        """Get error handling configuration"""
        return self.error_handling_config
    
    def get_monitoring_config(self) -> Optional[MonitoringConfig]:
        """Get monitoring configuration"""
        return self.monitoring_config
    
    def update_agent_config(self, agent_name: str, **kwargs):
        """
        Update configuration for a specific agent
        
        Args:
            agent_name: Name of the agent
            **kwargs: Configuration values to update
        """
        if agent_name not in self.agent_configs:
            logger.warning(f"Agent {agent_name} not found, creating new config")
            self.agent_configs[agent_name] = AgentConfig()
        
        agent_config = self.agent_configs[agent_name]
        
        for key, value in kwargs.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)
                logger.info(f"Updated {agent_name}.{key} = {value}")
            elif key == "custom_settings":
                agent_config.custom_settings.update(value)
                logger.info(f"Updated {agent_name}.custom_settings")
            else:
                logger.warning(f"Unknown config key for agent {agent_name}: {key}")
        
        # Validate after update
        self._validate_config()
    
    def update_multi_agent_config(self, **kwargs):
        """
        Update multi-agent system configuration
        
        Args:
            **kwargs: Configuration values to update
        """
        if not self.multi_agent_config:
            self.multi_agent_config = MultiAgentConfig()
        
        for key, value in kwargs.items():
            if hasattr(self.multi_agent_config, key):
                setattr(self.multi_agent_config, key, value)
                logger.info(f"Updated multi_agent.{key} = {value}")
            else:
                logger.warning(f"Unknown multi_agent config key: {key}")
        
        # Validate after update
        self._validate_config()
    
    def update_routing_config(self, **kwargs):
        """
        Update routing configuration
        
        Args:
            **kwargs: Configuration values to update
        """
        if not self.routing_config:
            self.routing_config = RoutingConfig()
        
        for key, value in kwargs.items():
            if hasattr(self.routing_config, key):
                setattr(self.routing_config, key, value)
                logger.info(f"Updated routing.{key} = {value}")
            else:
                logger.warning(f"Unknown routing config key: {key}")
        
        # Validate after update
        self._validate_config()
    
    def should_run_agent(self, agent_name: str, state: Dict[str, Any]) -> bool:
        """
        Determine if an agent should run based on its configuration and state
        
        Args:
            agent_name: Name of the agent
            state: Current workflow state
            
        Returns:
            True if agent should run
        """
        agent_config = self.get_agent_config(agent_name)
        
        if not agent_config or not agent_config.enabled:
            return False
        
        # Check conditional execution
        if agent_config.conditional and agent_config.condition:
            return self._evaluate_condition(agent_config.condition, state)
        
        return True
    
    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """
        Evaluate a condition string against state
        
        Args:
            condition: Condition string (e.g., "prospectus_data_available")
            state: Current workflow state
            
        Returns:
            True if condition is met
        """
        # Handle common conditions
        if condition == "prospectus_data_available":
            return bool(state.get("config", {}).get("prospectus_data"))
        
        elif condition == "fund_isin_available":
            return bool(state.get("metadata", {}).get("fund_isin"))
        
        elif condition == "esg_classification_not_other":
            esg_class = state.get("metadata", {}).get("esg_classification", "other")
            return esg_class != "other"
        
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def get_parallel_agents(self) -> List[str]:
        """Get list of agents that can run in parallel"""
        if self.routing_config:
            return self.routing_config.parallel_specialist_agents
        return []
    
    def get_conditional_agents(self) -> List[str]:
        """Get list of conditional agents"""
        if self.routing_config:
            return self.routing_config.conditional_agents
        return []
    
    def get_sequential_agents(self) -> List[str]:
        """Get list of agents that run sequentially"""
        if self.routing_config:
            return self.routing_config.sequential_flow
        return []
    
    def save_config(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: Path to save to (uses self.config_path if None)
        """
        save_path = path or self.config_path
        
        try:
            # Build config dict
            config_dict = {}
            
            # Add multi-agent config
            if self.multi_agent_config:
                config_dict['multi_agent'] = asdict(self.multi_agent_config)
            
            # Add agent configs
            config_dict['agents'] = {}
            for agent_name, agent_config in self.agent_configs.items():
                config_dict['agents'][agent_name] = asdict(agent_config)
            
            # Add routing config
            if self.routing_config:
                config_dict['routing'] = asdict(self.routing_config)
            
            # Add state management config
            if self.state_management_config:
                config_dict['state_management'] = asdict(self.state_management_config)
            
            # Add error handling config
            if self.error_handling_config:
                config_dict['error_handling'] = asdict(self.error_handling_config)
            
            # Add monitoring config
            if self.monitoring_config:
                config_dict['monitoring'] = asdict(self.monitoring_config)
            
            # Load existing config and merge
            existing_config = {}
            if os.path.exists(save_path):
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
            
            # Merge configs (agent config takes precedence)
            existing_config.update(config_dict)
            
            # Save
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2)
            
            logger.info(f"Agent configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save agent configuration: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        enabled_agents = [name for name, config in self.agent_configs.items() if config.enabled]
        conditional_agents = [name for name, config in self.agent_configs.items() 
                            if config.conditional and config.enabled]
        
        return {
            'multi_agent_enabled': self.is_multi_agent_enabled(),
            'parallel_execution': self.multi_agent_config.parallel_execution if self.multi_agent_config else False,
            'max_parallel_agents': self.multi_agent_config.max_parallel_agents if self.multi_agent_config else 0,
            'state_persistence': self.multi_agent_config.state_persistence if self.multi_agent_config else False,
            'total_agents': len(self.agent_configs),
            'enabled_agents': len(enabled_agents),
            'conditional_agents': len(conditional_agents),
            'context_threshold': self.routing_config.context_threshold if self.routing_config else None,
            'review_threshold': self.routing_config.review_threshold if self.routing_config else None,
            'monitoring_enabled': self.monitoring_config.enabled if self.monitoring_config else False,
            'agent_list': enabled_agents
        }
    
    def print_summary(self):
        """Print configuration summary"""
        summary = self.get_summary()
        
        print("="*70)
        print("MULTI-AGENT SYSTEM CONFIGURATION")
        print("="*70)
        print(f"\nMulti-Agent System: {'Enabled' if summary['multi_agent_enabled'] else 'Disabled'}")
        
        if summary['multi_agent_enabled']:
            print(f"Parallel Execution: {'Yes' if summary['parallel_execution'] else 'No'}")
            print(f"Max Parallel Agents: {summary['max_parallel_agents']}")
            print(f"State Persistence: {'Yes' if summary['state_persistence'] else 'No'}")
            print(f"\nAgents: {summary['enabled_agents']}/{summary['total_agents']} enabled")
            print(f"Conditional Agents: {summary['conditional_agents']}")
            print(f"\nRouting:")
            print(f"  Context Threshold: {summary['context_threshold']}%")
            print(f"  Review Threshold: {summary['review_threshold']}%")
            print(f"\nMonitoring: {'Enabled' if summary['monitoring_enabled'] else 'Disabled'}")
            print(f"\nEnabled Agents:")
            for agent_name in summary['agent_list']:
                agent_config = self.agent_configs[agent_name]
                status = "conditional" if agent_config.conditional else "always"
                print(f"  - {agent_name} ({status})")
        
        print("="*70)


# Global instance
_agent_config_manager = None


def get_agent_config_manager(config_path: str = "hybrid_config.json") -> AgentConfigManager:
    """
    Get or create global agent configuration manager
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        AgentConfigManager instance
    """
    global _agent_config_manager
    
    if _agent_config_manager is None:
        _agent_config_manager = AgentConfigManager(config_path)
    
    return _agent_config_manager


if __name__ == "__main__":
    # Test agent configuration manager
    print("="*70)
    print("Agent Configuration Manager - Test")
    print("="*70)
    
    # Create manager
    manager = get_agent_config_manager()
    
    # Print summary
    manager.print_summary()
    
    # Test agent config retrieval
    print("\n📊 Testing Agent Config Retrieval:")
    structure_config = manager.get_agent_config("structure")
    if structure_config:
        print(f"  Structure Agent:")
        print(f"    Enabled: {structure_config.enabled}")
        print(f"    Timeout: {structure_config.timeout_seconds}s")
        print(f"    Parallel Tools: {structure_config.parallel_tool_execution}")
    
    # Test runtime updates
    print("\n📊 Testing Runtime Updates:")
    print(f"  Before: structure.timeout_seconds = {structure_config.timeout_seconds}")
    manager.update_agent_config("structure", timeout_seconds=45)
    structure_config = manager.get_agent_config("structure")
    print(f"  After: structure.timeout_seconds = {structure_config.timeout_seconds}")
    
    # Test conditional agent evaluation
    print("\n📊 Testing Conditional Agent Evaluation:")
    test_state = {
        "config": {"prospectus_data": {"fund_name": "Test Fund"}},
        "metadata": {"fund_isin": "FR0000000000", "esg_classification": "article_8"}
    }
    
    for agent_name in ["prospectus", "registration", "esg"]:
        should_run = manager.should_run_agent(agent_name, test_state)
        print(f"  {agent_name}: {'Run' if should_run else 'Skip'}")
    
    # Test parallel agents
    print("\n📊 Testing Parallel Agents:")
    parallel_agents = manager.get_parallel_agents()
    print(f"  Parallel Agents: {', '.join(parallel_agents)}")
    
    print("\n" + "="*70)
    print("✓ Agent configuration manager tests complete")
    print("="*70)
