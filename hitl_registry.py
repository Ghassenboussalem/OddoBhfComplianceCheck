#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HITL Component Registry

Global registry for HITL components to avoid serialization issues with LangGraph.
This allows workflow_builder and agents to access the same HITL component instances
without storing them in the state (which would cause serialization errors).
"""

from typing import Optional, Any

# Global registry for HITL components
_REGISTRY = {}


def register_component(name: str, component: Any) -> None:
    """
    Register a HITL component in the global registry
    
    Args:
        name: Component name (e.g., 'review_manager', 'audit_logger')
        component: Component instance
    """
    _REGISTRY[name] = component


def get_component(name: str) -> Optional[Any]:
    """
    Get a HITL component from the global registry
    
    Args:
        name: Component name
        
    Returns:
        Component instance or None if not found
    """
    return _REGISTRY.get(name)


def clear_registry() -> None:
    """Clear all registered components"""
    _REGISTRY.clear()


def is_registered(name: str) -> bool:
    """
    Check if a component is registered
    
    Args:
        name: Component name
        
    Returns:
        True if component is registered
    """
    return name in _REGISTRY


def get_all_components() -> dict:
    """
    Get all registered components
    
    Returns:
        Dictionary of all registered components
    """
    return _REGISTRY.copy()
