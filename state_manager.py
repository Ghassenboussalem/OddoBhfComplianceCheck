#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Manager for Multi-Agent Compliance System

This module provides comprehensive state persistence and management capabilities
for the LangGraph-based multi-agent compliance workflow.

Key Features:
- State persistence to disk at configurable intervals
- State history tracking for audit and debugging
- Checkpoint management and restoration
- State validation and integrity checks
- Thread-safe state operations
"""

import logging
import json
import os
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import threading

from data_models_multiagent import (
    ComplianceState,
    serialize_state,
    deserialize_state,
    validate_compliance_state,
    get_state_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages state persistence, checkpointing, and history tracking
    
    Provides thread-safe operations for:
    - Saving and loading workflow states
    - Managing checkpoint history
    - Querying state history
    - State validation and integrity checks
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 1,
        max_history_size: int = 100,
        enable_compression: bool = False
    ):
        """
        Initialize state manager
        
        Args:
            checkpoint_dir: Directory for storing checkpoint files
            checkpoint_interval: Number of agent transitions between checkpoints
            max_history_size: Maximum number of historical states to keep
            enable_compression: Whether to compress saved states
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_history_size = max_history_size
        self.enable_compression = enable_compression
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # In-memory state history (for quick access)
        self._state_history: List[Dict[str, Any]] = []
        
        # Counter for checkpoint intervals
        self._transition_counter = 0
        
        logger.info(f"StateManager initialized: dir={checkpoint_dir}, interval={checkpoint_interval}")
    
    def should_checkpoint(self) -> bool:
        """
        Determine if a checkpoint should be created based on interval
        
        Returns:
            True if checkpoint should be created
        """
        with self._lock:
            self._transition_counter += 1
            should_save = (self._transition_counter % self.checkpoint_interval) == 0
            
            if should_save:
                logger.debug(f"Checkpoint triggered at transition {self._transition_counter}")
            
            return should_save
    
    def save_state(
        self,
        state: ComplianceState,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Save workflow state to disk
        
        Args:
            state: Compliance state to save
            checkpoint_id: Optional checkpoint ID (auto-generated if None)
            metadata: Optional metadata to store with checkpoint
            
        Returns:
            Tuple of (success, checkpoint_id)
        """
        try:
            with self._lock:
                # Generate checkpoint ID if not provided
                if checkpoint_id is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    document_id = state.get("document_id", "unknown")
                    checkpoint_id = f"{document_id}_{timestamp}"
                
                # Validate state before saving
                is_valid, errors = validate_compliance_state(state)
                if not is_valid:
                    logger.warning(f"Saving state with validation errors: {errors}")
                
                # Serialize state
                state_json = serialize_state(state)
                
                # Create checkpoint record
                checkpoint_record = {
                    "checkpoint_id": checkpoint_id,
                    "timestamp": datetime.now().isoformat(),
                    "document_id": state.get("document_id", "unknown"),
                    "current_agent": state.get("current_agent", "unknown"),
                    "workflow_status": state.get("workflow_status", "unknown"),
                    "violation_count": len(state.get("violations", [])),
                    "metadata": metadata or {},
                    "state_json": state_json,
                    "is_valid": is_valid,
                    "validation_errors": errors if not is_valid else []
                }
                
                # Save to file
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_record, f, indent=2, ensure_ascii=False)
                
                # Add to history
                self._add_to_history(checkpoint_record)
                
                logger.info(f"State saved: {checkpoint_id} (agent={state.get('current_agent')})")
                return True, checkpoint_id
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)
            return False, ""
    
    def load_state(
        self,
        checkpoint_id: str
    ) -> Optional[ComplianceState]:
        """
        Load workflow state from disk
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Loaded ComplianceState or None if failed
        """
        try:
            with self._lock:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
                
                if not checkpoint_path.exists():
                    logger.error(f"Checkpoint not found: {checkpoint_id}")
                    return None
                
                # Load checkpoint record
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_record = json.load(f)
                
                # Deserialize state
                state_json = checkpoint_record.get("state_json", "")
                state = deserialize_state(state_json)
                
                # Validate loaded state
                is_valid, errors = validate_compliance_state(state)
                if not is_valid:
                    logger.warning(f"Loaded state has validation errors: {errors}")
                
                logger.info(f"State loaded: {checkpoint_id} (agent={state.get('current_agent')})")
                return state
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}", exc_info=True)
            return None
    
    def get_state_history(
        self,
        document_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get state history records
        
        Args:
            document_id: Optional filter by document ID
            limit: Optional limit on number of records
            
        Returns:
            List of checkpoint records (most recent first)
        """
        with self._lock:
            history = self._state_history.copy()
            
            # Filter by document ID if specified
            if document_id:
                history = [h for h in history if h.get("document_id") == document_id]
            
            # Apply limit
            if limit:
                history = history[:limit]
            
            return history
    
    def get_latest_checkpoint(
        self,
        document_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the most recent checkpoint ID
        
        Args:
            document_id: Optional filter by document ID
            
        Returns:
            Checkpoint ID or None if no checkpoints exist
        """
        history = self.get_state_history(document_id=document_id, limit=1)
        
        if history:
            return history[0].get("checkpoint_id")
        
        return None
    
    def list_checkpoints(
        self,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available checkpoints
        
        Args:
            document_id: Optional filter by document ID
            
        Returns:
            List of checkpoint metadata (without full state)
        """
        try:
            checkpoints = []
            
            # Scan checkpoint directory
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                    
                    # Filter by document ID if specified
                    if document_id and record.get("document_id") != document_id:
                        continue
                    
                    # Extract metadata (exclude full state)
                    metadata = {
                        "checkpoint_id": record.get("checkpoint_id"),
                        "timestamp": record.get("timestamp"),
                        "document_id": record.get("document_id"),
                        "current_agent": record.get("current_agent"),
                        "workflow_status": record.get("workflow_status"),
                        "violation_count": record.get("violation_count"),
                        "is_valid": record.get("is_valid"),
                        "file_path": str(checkpoint_file)
                    }
                    
                    checkpoints.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
            
            # Sort by timestamp (most recent first)
            checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def delete_checkpoint(
        self,
        checkpoint_id: str
    ) -> bool:
        """
        Delete a checkpoint from disk
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
                
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    
                    # Remove from history
                    self._state_history = [
                        h for h in self._state_history
                        if h.get("checkpoint_id") != checkpoint_id
                    ]
                    
                    logger.info(f"Checkpoint deleted: {checkpoint_id}")
                    return True
                else:
                    logger.warning(f"Checkpoint not found: {checkpoint_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(
        self,
        keep_count: Optional[int] = None,
        older_than_days: Optional[int] = None
    ) -> int:
        """
        Clean up old checkpoints
        
        Args:
            keep_count: Keep only this many most recent checkpoints
            older_than_days: Delete checkpoints older than this many days
            
        Returns:
            Number of checkpoints deleted
        """
        try:
            checkpoints = self.list_checkpoints()
            deleted_count = 0
            
            # Delete by count
            if keep_count is not None and len(checkpoints) > keep_count:
                to_delete = checkpoints[keep_count:]
                for checkpoint in to_delete:
                    if self.delete_checkpoint(checkpoint["checkpoint_id"]):
                        deleted_count += 1
            
            # Delete by age
            if older_than_days is not None:
                from datetime import timedelta
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                
                for checkpoint in checkpoints:
                    timestamp_str = checkpoint.get("timestamp", "")
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp < cutoff_date:
                            if self.delete_checkpoint(checkpoint["checkpoint_id"]):
                                deleted_count += 1
                    except Exception:
                        pass
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old checkpoints")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0
    
    def get_checkpoint_info(
        self,
        checkpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a checkpoint
        
        Args:
            checkpoint_id: ID of checkpoint
            
        Returns:
            Checkpoint metadata or None if not found
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            
            if not checkpoint_path.exists():
                return None
            
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            # Return metadata without full state
            return {
                "checkpoint_id": record.get("checkpoint_id"),
                "timestamp": record.get("timestamp"),
                "document_id": record.get("document_id"),
                "current_agent": record.get("current_agent"),
                "workflow_status": record.get("workflow_status"),
                "violation_count": record.get("violation_count"),
                "metadata": record.get("metadata", {}),
                "is_valid": record.get("is_valid"),
                "validation_errors": record.get("validation_errors", []),
                "file_path": str(checkpoint_path),
                "file_size": checkpoint_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return None
    
    def _add_to_history(self, checkpoint_record: Dict[str, Any]) -> None:
        """
        Add checkpoint record to in-memory history
        
        Args:
            checkpoint_record: Checkpoint record to add
        """
        # Create lightweight history entry (without full state)
        history_entry = {
            "checkpoint_id": checkpoint_record.get("checkpoint_id"),
            "timestamp": checkpoint_record.get("timestamp"),
            "document_id": checkpoint_record.get("document_id"),
            "current_agent": checkpoint_record.get("current_agent"),
            "workflow_status": checkpoint_record.get("workflow_status"),
            "violation_count": checkpoint_record.get("violation_count"),
            "is_valid": checkpoint_record.get("is_valid")
        }
        
        # Add to front of history
        self._state_history.insert(0, history_entry)
        
        # Trim history if needed
        if len(self._state_history) > self.max_history_size:
            self._state_history = self._state_history[:self.max_history_size]
    
    def export_state_history(
        self,
        output_path: str,
        document_id: Optional[str] = None
    ) -> bool:
        """
        Export state history to a JSON file
        
        Args:
            output_path: Path to save history
            document_id: Optional filter by document ID
            
        Returns:
            True if successful
        """
        try:
            history = self.get_state_history(document_id=document_id)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported state history to {output_path} ({len(history)} records)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export state history: {e}")
            return False


class CheckpointManager:
    """
    High-level checkpoint management for workflow execution
    
    Integrates with StateManager to provide automatic checkpointing
    during workflow execution.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        auto_checkpoint: bool = True
    ):
        """
        Initialize checkpoint manager
        
        Args:
            state_manager: StateManager instance
            auto_checkpoint: Whether to automatically checkpoint on transitions
        """
        self.state_manager = state_manager
        self.auto_checkpoint = auto_checkpoint
        
        logger.info(f"CheckpointManager initialized (auto={auto_checkpoint})")
    
    def on_agent_transition(
        self,
        state: ComplianceState,
        from_agent: str,
        to_agent: str
    ) -> Optional[str]:
        """
        Handle agent transition and potentially create checkpoint
        
        Args:
            state: Current workflow state
            from_agent: Agent that just completed
            to_agent: Agent about to execute
            
        Returns:
            Checkpoint ID if checkpoint was created, None otherwise
        """
        if not self.auto_checkpoint:
            return None
        
        # Check if we should checkpoint
        if self.state_manager.should_checkpoint():
            metadata = {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "transition_type": "agent_transition"
            }
            
            success, checkpoint_id = self.state_manager.save_state(
                state,
                metadata=metadata
            )
            
            if success:
                logger.debug(f"Auto-checkpoint created: {checkpoint_id}")
                return checkpoint_id
        
        return None
    
    def on_workflow_interrupt(
        self,
        state: ComplianceState,
        reason: str
    ) -> Optional[str]:
        """
        Handle workflow interrupt and create checkpoint
        
        Args:
            state: Current workflow state
            reason: Reason for interrupt (e.g., "hitl_review")
            
        Returns:
            Checkpoint ID if successful
        """
        metadata = {
            "interrupt_reason": reason,
            "transition_type": "workflow_interrupt"
        }
        
        success, checkpoint_id = self.state_manager.save_state(
            state,
            metadata=metadata
        )
        
        if success:
            logger.info(f"Interrupt checkpoint created: {checkpoint_id} (reason={reason})")
            return checkpoint_id
        
        return None
    
    def on_workflow_complete(
        self,
        state: ComplianceState
    ) -> Optional[str]:
        """
        Handle workflow completion and create final checkpoint
        
        Args:
            state: Final workflow state
            
        Returns:
            Checkpoint ID if successful
        """
        metadata = {
            "transition_type": "workflow_complete",
            "final_violation_count": len(state.get("violations", []))
        }
        
        success, checkpoint_id = self.state_manager.save_state(
            state,
            metadata=metadata
        )
        
        if success:
            logger.info(f"Final checkpoint created: {checkpoint_id}")
            return checkpoint_id
        
        return None


# Export all public symbols
__all__ = [
    "StateManager",
    "CheckpointManager"
]
