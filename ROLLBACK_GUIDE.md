# Multi-Agent System Rollback Guide

## Overview

The rollback scripts allow you to restore your system to a previous state before the multi-agent migration. This is useful if you encounter issues with the multi-agent system and need to revert to the original implementation.

## Available Scripts

Three rollback scripts are provided for different platforms:

- **rollback_multiagent.ps1** - PowerShell (Windows, Linux, macOS)
- **rollback_multiagent.sh** - Bash (Linux, macOS)
- **rollback_multiagent.bat** - Windows Command Prompt

## Usage

### PowerShell (Recommended for Windows)

```powershell
# Rollback to latest backup
.\rollback_multiagent.ps1

# Rollback to specific backup
.\rollback_multiagent.ps1 -BackupDir "backups\deployment_20251123_120000"

# Force rollback without prompts
.\rollback_multiagent.ps1 -Force
```

### Bash (Linux/macOS)

```bash
# Make script executable (first time only)
chmod +x rollback_multiagent.sh

# Rollback to latest backup
./rollback_multiagent.sh

# Rollback to specific backup
./rollback_multiagent.sh backups/deployment_20251123_120000

# Force rollback without prompts
./rollback_multiagent.sh --force
```

### Windows Command Prompt

```cmd
REM Rollback to latest backup
rollback_multiagent.bat

REM Rollback to specific backup
rollback_multiagent.bat backups\deployment_20251123_120000

REM Force rollback without prompts
rollback_multiagent.bat backups\deployment_20251123_120000 --force
```

## What Gets Restored

The rollback script restores the following:

1. **Configuration Files**
   - `hybrid_config.json` (with multi-agent mode disabled)
   - `.env` (environment variables)
   - `review_queue.json` (review queue state)

2. **Data Directories**
   - `checkpoints/` (workflow checkpoints)
   - `audit_logs/` (audit trail)

3. **System Files**
   - All files from the backup directory

## Rollback Process

The rollback script performs the following steps:

1. **Find Backup** - Locates the latest backup or uses specified backup directory
2. **Validate Backup** - Checks backup integrity and critical files
3. **Confirm Rollback** - Prompts for user confirmation (unless --force)
4. **Stop Processes** - Stops running Python processes that might lock files
5. **Create Pre-Rollback Backup** - Backs up current state before rollback
6. **Restore Configuration** - Restores configuration files and disables multi-agent mode
7. **Restore Files** - Restores all files from backup
8. **Verify Success** - Tests that rollback completed successfully

## Safety Features

### Pre-Rollback Backup

Before performing the rollback, the script creates a backup of your current state in:
- `backups/pre_rollback_YYYYMMDD_HHMMSS/`

This allows you to restore the multi-agent system if the rollback was performed by mistake.

### Backup Validation

The script validates the backup before proceeding:
- Checks that backup directory exists
- Verifies critical files are present
- Reads manifest file if available

### Confirmation Prompts

Unless using `--force`, the script prompts for confirmation:
- Before starting rollback
- Before stopping processes
- When backup is incomplete

## Rollback Verification

After rollback, the script verifies:
- Critical files exist (`check.py`, `hybrid_config.json`)
- Configuration is valid JSON
- Multi-agent mode is disabled
- Basic system functionality works

## Troubleshooting

### Rollback Failed

If rollback fails, check the rollback log:
- **PowerShell**: `rollback_YYYYMMDD_HHMMSS.log`
- **Bash**: `rollback_YYYYMMDD_HHMMSS.log`
- **Batch**: `rollback_YYYYMMDD_HHMMSS.log`

Common issues:
1. **Files in use** - Stop all Python processes before rollback
2. **Insufficient permissions** - Run with administrator/sudo privileges
3. **Incomplete backup** - Use a different backup or use `--force`

### Restore Multi-Agent System

If you need to restore the multi-agent system after rollback:

```bash
# PowerShell
.\deploy_multiagent.ps1

# Bash
./deploy_multiagent.sh

# Batch
deploy_multiagent.bat
```

### Restore Pre-Rollback State

If you rolled back by mistake, restore from the pre-rollback backup:

```bash
# PowerShell
.\rollback_multiagent.ps1 -BackupDir "backups\pre_rollback_YYYYMMDD_HHMMSS"

# Bash
./rollback_multiagent.sh backups/pre_rollback_YYYYMMDD_HHMMSS

# Batch
rollback_multiagent.bat backups\pre_rollback_YYYYMMDD_HHMMSS
```

## Backup Management

### List Available Backups

```bash
# PowerShell
Get-ChildItem backups -Directory | Sort-Object Name -Descending

# Bash
ls -lt backups/

# Batch
dir /b /ad /o-n backups
```

### Delete Old Backups

```bash
# PowerShell
Remove-Item "backups\deployment_20251120_*" -Recurse -Force

# Bash
rm -rf backups/deployment_20251120_*

# Batch
rmdir /s /q backups\deployment_20251120_*
```

## Best Practices

1. **Test Before Production** - Test rollback in a development environment first
2. **Keep Multiple Backups** - Don't delete backups immediately after deployment
3. **Document Changes** - Note what was changed before rollback
4. **Verify After Rollback** - Test the system thoroughly after rollback
5. **Review Logs** - Always check the rollback log for warnings

## Support

If you encounter issues with rollback:

1. Check the rollback log for detailed error messages
2. Verify backup integrity manually
3. Try rollback with `--force` flag
4. Manually restore files from backup directory
5. Contact support with rollback log

## Related Documentation

- [Deployment Guide](DEPLOYMENT_README.md)
- [Migration Guide](docs/MIGRATION_TO_MULTIAGENT.md)
- [Troubleshooting Guide](docs/MULTIAGENT_TROUBLESHOOTING.md)
