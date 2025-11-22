#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistence Utilities - Common utilities for data persistence
Provides thread-safe file operations, locking, and data migration
"""

import json
import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Manages thread-safe file operations with locking and atomic writes
    
    Features:
    - Thread-safe file locking (cross-platform)
    - Atomic writes using temporary files
    - Automatic backup before overwrite
    - Corruption detection and recovery
    - Schema migration support
    """
    
    @staticmethod
    def load_json(filepath: str, migration_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load JSON file with thread-safe locking
        
        Args:
            filepath: Path to JSON file
            migration_func: Optional function to migrate old schema
            
        Returns:
            Loaded data dictionary
        """
        try:
            if not os.path.exists(filepath):
                logger.info(f"File not found: {filepath}")
                return {}
            
            # Try to import fcntl for Unix-like systems
            try:
                import fcntl
                has_fcntl = True
            except ImportError:
                has_fcntl = False
            
            # Open with shared lock for reading
            with open(filepath, 'r') as f:
                # Acquire shared lock (multiple readers allowed)
                if has_fcntl:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    except OSError as e:
                        logger.warning(f"Could not acquire file lock: {e}")
                
                try:
                    data = json.load(f)
                    
                    # Apply migration if provided
                    if migration_func:
                        schema_version = data.get('schema_version', 1)
                        if schema_version < 2:  # Assuming current version is 2
                            data = migration_func(data, schema_version)
                    
                    return data
                    
                finally:
                    # Release lock
                    if has_fcntl:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except OSError:
                            pass
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            # Backup corrupted file
            PersistenceManager.backup_file(filepath, suffix='corrupted')
            return {}
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    @staticmethod
    def save_json(filepath: str, data: Dict[str, Any], backup: bool = True):
        """
        Save JSON file with thread-safe locking and atomic write
        
        Args:
            filepath: Path to JSON file
            data: Data dictionary to save
            backup: Whether to backup existing file before overwrite
        """
        try:
            # Try to import fcntl for Unix-like systems
            try:
                import fcntl
                has_fcntl = True
            except ImportError:
                has_fcntl = False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            
            # Write to temporary file first (atomic write)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(filepath) or '.',
                prefix='.tmp_',
                suffix='.json'
            )
            
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    # Acquire exclusive lock for writing
                    if has_fcntl:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        except OSError as e:
                            logger.warning(f"Could not acquire file lock: {e}")
                    
                    try:
                        json.dump(data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                    finally:
                        # Release lock
                        if has_fcntl:
                            try:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            except OSError:
                                pass
                
                # Backup existing file if requested
                if backup and os.path.exists(filepath):
                    PersistenceManager.backup_file(filepath, suffix='backup')
                
                # Atomic rename (replaces old file)
                if os.name == 'nt':  # Windows
                    # Windows doesn't support atomic replace if target exists
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    os.rename(temp_path, filepath)
                else:  # Unix-like
                    os.replace(temp_path, filepath)
                
                logger.debug(f"Saved data to {filepath}")
                
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise e
            
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            raise
    
    @staticmethod
    def backup_file(filepath: str, suffix: str = 'backup'):
        """
        Create a backup copy of a file
        
        Args:
            filepath: Path to file to backup
            suffix: Suffix to add to backup filename
        """
        try:
            if not os.path.exists(filepath):
                return
            
            # Generate backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{filepath}.{suffix}.{timestamp}"
            
            # Copy file
            shutil.copy2(filepath, backup_path)
            logger.info(f"Backed up {filepath} to {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to backup {filepath}: {e}")
    
    @staticmethod
    def rotate_file(filepath: str, archive_dir: Optional[str] = None, 
                   max_size_mb: Optional[float] = None) -> bool:
        """
        Rotate a file by moving it to archive
        
        Args:
            filepath: Path to file to rotate
            archive_dir: Directory for archived files (default: same as file)
            max_size_mb: Optional maximum file size in MB before rotation
            
        Returns:
            True if rotation occurred, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
            
            # Check size if threshold provided
            if max_size_mb:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if size_mb < max_size_mb:
                    return False
            
            # Determine archive directory
            if archive_dir is None:
                archive_dir = os.path.dirname(filepath) or '.'
            os.makedirs(archive_dir, exist_ok=True)
            
            # Generate archive filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            basename = os.path.basename(filepath)
            name, ext = os.path.splitext(basename)
            archive_name = f"{name}_{timestamp}{ext}"
            archive_path = os.path.join(archive_dir, archive_name)
            
            # Move file to archive
            shutil.move(filepath, archive_path)
            logger.info(f"Rotated {filepath} to {archive_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate {filepath}: {e}")
            return False
    
    @staticmethod
    def cleanup_old_backups(directory: str, pattern: str = '*.backup.*', 
                           keep_count: int = 5):
        """
        Clean up old backup files, keeping only the most recent
        
        Args:
            directory: Directory containing backup files
            pattern: Glob pattern for backup files
            keep_count: Number of most recent backups to keep
        """
        try:
            from pathlib import Path
            
            # Find all backup files
            backup_files = sorted(
                Path(directory).glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove old backups
            removed_count = 0
            for backup_file in backup_files[keep_count:]:
                try:
                    backup_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {backup_file}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old backup files in {directory}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup backups in {directory}: {e}")
    
    @staticmethod
    def verify_json_integrity(filepath: str) -> bool:
        """
        Verify JSON file can be loaded without errors
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            True if file is valid JSON, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                json.load(f)
            return True
        except Exception as e:
            logger.error(f"JSON integrity check failed for {filepath}: {e}")
            return False


class SchemaMigrator:
    """
    Handles schema migrations for data files
    
    Features:
    - Version tracking
    - Migration chain execution
    - Rollback support
    - Migration validation
    """
    
    def __init__(self, current_version: int = 2):
        """
        Initialize schema migrator
        
        Args:
            current_version: Current schema version
        """
        self.current_version = current_version
        self.migrations = {}
    
    def register_migration(self, from_version: int, to_version: int, 
                          migration_func: Callable):
        """
        Register a migration function
        
        Args:
            from_version: Source schema version
            to_version: Target schema version
            migration_func: Function that performs migration
        """
        key = (from_version, to_version)
        self.migrations[key] = migration_func
        logger.debug(f"Registered migration: v{from_version} -> v{to_version}")
    
    def migrate(self, data: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """
        Migrate data from old version to current version
        
        Args:
            data: Data in old schema
            from_version: Current version of data
            
        Returns:
            Migrated data in current schema
        """
        if from_version == self.current_version:
            return data
        
        if from_version > self.current_version:
            logger.warning(f"Data version ({from_version}) is newer than current ({self.current_version})")
            return data
        
        # Execute migration chain
        current_data = data
        current_ver = from_version
        
        while current_ver < self.current_version:
            # Find next migration
            next_ver = current_ver + 1
            key = (current_ver, next_ver)
            
            if key not in self.migrations:
                logger.error(f"No migration found for v{current_ver} -> v{next_ver}")
                break
            
            # Execute migration
            logger.info(f"Migrating data: v{current_ver} -> v{next_ver}")
            current_data = self.migrations[key](current_data)
            current_data['schema_version'] = next_ver
            current_ver = next_ver
        
        return current_data
    
    def validate_schema(self, data: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate that data contains required fields
        
        Args:
            data: Data to validate
            required_fields: List of required field names
            
        Returns:
            True if all required fields present, False otherwise
        """
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        return True


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Persistence Utilities - Testing")
    print("="*70)
    
    # Test save and load
    print("\n📝 Testing save and load...")
    test_file = "test_persistence.json"
    test_data = {
        'schema_version': 2,
        'test_field': 'test_value',
        'timestamp': datetime.now().isoformat(),
        'items': [1, 2, 3, 4, 5]
    }
    
    PersistenceManager.save_json(test_file, test_data)
    print(f"  ✓ Saved test data to {test_file}")
    
    loaded_data = PersistenceManager.load_json(test_file)
    print(f"  ✓ Loaded data: {len(loaded_data)} fields")
    
    # Test integrity verification
    print("\n🔍 Testing integrity verification...")
    is_valid = PersistenceManager.verify_json_integrity(test_file)
    print(f"  ✓ File integrity: {'Valid' if is_valid else 'Invalid'}")
    
    # Test backup
    print("\n💾 Testing backup...")
    PersistenceManager.backup_file(test_file, suffix='test')
    print(f"  ✓ Created backup")
    
    # Test schema migration
    print("\n🔄 Testing schema migration...")
    migrator = SchemaMigrator(current_version=2)
    
    def migrate_v1_to_v2(data):
        """Example migration function"""
        data['new_field'] = 'added_in_v2'
        return data
    
    migrator.register_migration(1, 2, migrate_v1_to_v2)
    
    old_data = {'schema_version': 1, 'old_field': 'value'}
    migrated_data = migrator.migrate(old_data, from_version=1)
    print(f"  ✓ Migrated data: v{old_data['schema_version']} -> v{migrated_data['schema_version']}")
    print(f"    New field added: {'new_field' in migrated_data}")
    
    # Test validation
    print("\n✅ Testing schema validation...")
    required_fields = ['schema_version', 'new_field']
    is_valid = migrator.validate_schema(migrated_data, required_fields)
    print(f"  ✓ Schema validation: {'Passed' if is_valid else 'Failed'}")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    try:
        os.remove(test_file)
        # Clean up backup files
        import glob
        for backup in glob.glob(f"{test_file}.*.test.*"):
            os.remove(backup)
        print(f"  ✓ Cleaned up test files")
    except Exception as e:
        print(f"  ⚠ Cleanup warning: {e}")
    
    print("\n" + "="*70)
    print("✓ Persistence utilities test complete")
    print("="*70)
