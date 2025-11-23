# Multi-Agent System Deployment Guide

This guide explains how to deploy the LangGraph-based multi-agent compliance system.

## Overview

The deployment scripts automate the complete deployment process including:
- ✓ Python installation verification
- ✓ Dependency installation
- ✓ Configuration validation
- ✓ Backup creation
- ✓ Test execution
- ✓ System deployment
- ✓ Multi-agent mode enablement

## Deployment Scripts

Three deployment scripts are provided for different platforms:

### 1. **deploy_multiagent.sh** (Linux/macOS)
Bash script for Unix-based systems.

**Usage:**
```bash
chmod +x deploy_multiagent.sh
./deploy_multiagent.sh
```

### 2. **deploy_multiagent.ps1** (Windows PowerShell)
PowerShell script with advanced features and parameters.

**Usage:**
```powershell
# Basic deployment
.\deploy_multiagent.ps1

# Skip tests
.\deploy_multiagent.ps1 -SkipTests

# Skip backup
.\deploy_multiagent.ps1 -SkipBackup

# Force deployment even if tests fail
.\deploy_multiagent.ps1 -Force

# Custom backup directory
.\deploy_multiagent.ps1 -BackupDir "my_backup"

# Custom Python command
.\deploy_multiagent.ps1 -PythonCmd "python3"
```

**Parameters:**
- `-SkipTests`: Skip running tests before deployment
- `-SkipBackup`: Skip creating backup of current system
- `-Force`: Force deployment even if tests fail
- `-BackupDir`: Custom backup directory name
- `-PythonCmd`: Custom Python command (default: "python")

### 3. **deploy_multiagent.bat** (Windows CMD)
Batch script for Windows Command Prompt.

**Usage:**
```cmd
deploy_multiagent.bat
```

## Deployment Process

All scripts follow the same deployment process:

### Step 1: Check Python Installation
- Verifies Python is installed
- Checks Python version (requires 3.8+)
- Validates Python compatibility

### Step 2: Create Backup
- Creates timestamped backup directory
- Backs up configuration files:
  - `hybrid_config.json`
  - `.env`
  - `review_queue.json`
- Backs up directories:
  - `audit_logs/`
  - `checkpoints/`
- Creates backup manifest

### Step 3: Install Dependencies
- Upgrades pip to latest version
- Installs all packages from `requirements.txt`:
  - `langgraph>=0.2.0`
  - `langchain>=0.3.0`
  - `langchain-openai>=0.2.0`
  - `langchain-community>=0.3.0`
  - `langgraph-checkpoint-sqlite>=1.0.0`
  - And all other dependencies

### Step 4: Validate Configuration
- Checks for required configuration files
- Validates `hybrid_config.json` structure
- Checks for `.env` file (creates from `.env.example` if missing)
- Validates multi-agent configuration sections

### Step 5: Create Required Directories
- `checkpoints/` - For workflow state persistence
- `audit_logs/` - For audit trail
- `monitoring/logs/` - For agent execution logs
- `monitoring/metrics/` - For performance metrics
- `monitoring/visualizations/` - For workflow diagrams

### Step 6: Run Tests (Optional)
- Runs unit tests for critical components:
  - `test_data_models_multiagent.py`
  - `test_base_agent.py`
  - `test_workflow_builder.py`
- Runs integration tests:
  - `tests/test_workflow.py`
- Runs validation test with `exemple.json`

### Step 7: Enable Multi-Agent Mode
- Updates `hybrid_config.json`
- Sets `multi_agent.enabled = true`
- Saves updated configuration

### Step 8: Display Summary
- Shows deployment status
- Lists backup location
- Provides usage instructions
- Shows next steps

## Prerequisites

### Required Files
- `requirements.txt` - Python dependencies
- `hybrid_config.json` - System configuration
- `.env.example` - Environment template
- `check_multiagent.py` - Multi-agent entry point
- `workflow_builder.py` - Workflow construction
- `data_models_multiagent.py` - State definitions
- `agents/base_agent.py` - Base agent class
- `agents/supervisor_agent.py` - Supervisor agent

### Required Python Version
- Python 3.8 or higher

### Required Environment Variables
Create `.env` file with:
```env
# AI Service Configuration
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Specific model selection
AI_PROVIDER=auto
AI_MODEL=auto
```

## Post-Deployment

### 1. Verify Installation
```bash
python check_multiagent.py exemple.json
```

### 2. Check Configuration
Review `hybrid_config.json` and ensure:
- `multi_agent.enabled = true`
- Agent configurations are correct
- Routing thresholds are appropriate

### 3. Test Features
```bash
# Test with metrics
python check_multiagent.py exemple.json --show-metrics

# Test with review mode
python check_multiagent.py exemple.json --review-mode

# Test with custom confidence threshold
python check_multiagent.py exemple.json --ai-confidence=80
```

### 4. Monitor System
```bash
# Start monitoring dashboard
python monitoring/dashboard.py

# View agent logs
cat monitoring/logs/agent_execution.log

# View metrics
cat monitoring/metrics/metrics_*.json
```

## Rollback

If deployment fails or you need to rollback:

### Using Backup
```bash
# Linux/macOS
cp -r backups/deployment_YYYYMMDD_HHMMSS/* .

# Windows PowerShell
Copy-Item -Path backups\deployment_YYYYMMDD_HHMMSS\* -Destination . -Recurse -Force

# Windows CMD
xcopy /e /y backups\deployment_YYYYMMDD_HHMMSS\* .
```

### Using Rollback Script (if available)
```bash
# Linux/macOS
./rollback_multiagent.sh

# Windows PowerShell
.\rollback_multiagent.ps1 -BackupDir "backup_YYYYMMDD_HHMMSS"

# Windows CMD
rollback_multiagent.bat
```

## Troubleshooting

### Python Not Found
**Error:** `Python not found`

**Solution:**
- Install Python 3.8 or higher from python.org
- Add Python to system PATH
- Use custom Python command: `.\deploy_multiagent.ps1 -PythonCmd "python3"`

### Dependency Installation Failed
**Error:** `Failed to install dependencies`

**Solution:**
- Check internet connection
- Upgrade pip: `python -m pip install --upgrade pip`
- Install dependencies manually: `pip install -r requirements.txt`
- Check for conflicting packages

### Configuration Validation Failed
**Error:** `Invalid JSON in configuration file`

**Solution:**
- Validate JSON syntax in `hybrid_config.json`
- Use JSON validator: https://jsonlint.com/
- Restore from backup if corrupted

### Tests Failed
**Error:** `Tests failed`

**Solution:**
- Review test output in deployment log
- Fix failing tests or use `-Force` flag to deploy anyway
- Skip tests: `.\deploy_multiagent.ps1 -SkipTests`

### Missing .env File
**Warning:** `.env file not found`

**Solution:**
- Copy `.env.example` to `.env`
- Configure API keys in `.env`
- Ensure at least one AI provider is configured

### Permission Denied (Linux/macOS)
**Error:** `Permission denied`

**Solution:**
```bash
chmod +x deploy_multiagent.sh
./deploy_multiagent.sh
```

### Execution Policy Error (Windows PowerShell)
**Error:** `Execution policy does not allow running scripts`

**Solution:**
```powershell
# Temporarily allow script execution
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\deploy_multiagent.ps1

# Or run with bypass
powershell -ExecutionPolicy Bypass -File .\deploy_multiagent.ps1
```

## Deployment Logs

All deployment scripts create detailed logs:

- **Linux/macOS:** Console output only (can redirect to file)
- **Windows PowerShell:** `deployment_YYYYMMDD_HHMMSS.log`
- **Windows CMD:** `deployment_YYYYMMDD_HHMMSS.log`

Review logs for:
- Warnings about missing files
- Configuration issues
- Test failures
- Dependency conflicts

## Best Practices

### Before Deployment
1. ✓ Review current system configuration
2. ✓ Backup important data manually
3. ✓ Test in development environment first
4. ✓ Review deployment script parameters
5. ✓ Ensure API keys are configured

### During Deployment
1. ✓ Monitor deployment progress
2. ✓ Review warnings and errors
3. ✓ Don't interrupt the process
4. ✓ Note backup directory location

### After Deployment
1. ✓ Verify system functionality
2. ✓ Test with sample documents
3. ✓ Review configuration changes
4. ✓ Monitor system performance
5. ✓ Keep backup for rollback

## Support

For issues or questions:
1. Check deployment log for errors
2. Review troubleshooting section
3. Consult documentation:
   - `docs/MULTI_AGENT_ARCHITECTURE.md`
   - `docs/MULTIAGENT_CONFIGURATION.md`
   - `docs/MULTIAGENT_TROUBLESHOOTING.md`
4. Check backup location for rollback

## Next Steps

After successful deployment:

1. **Configure System**
   - Review `hybrid_config.json`
   - Adjust agent settings
   - Configure routing thresholds

2. **Test Functionality**
   - Run compliance checks
   - Test HITL workflow
   - Verify parallel execution

3. **Monitor Performance**
   - Check agent execution times
   - Review metrics dashboard
   - Monitor resource usage

4. **Integrate with Workflow**
   - Update CI/CD pipelines
   - Configure automated checks
   - Set up monitoring alerts

## Summary

The deployment scripts provide a complete, automated deployment process for the multi-agent compliance system. They handle all aspects of deployment including backup, dependency installation, testing, and configuration.

Choose the appropriate script for your platform and follow the deployment process. Review logs and troubleshooting guide if issues occur.

For detailed architecture and usage information, see:
- `docs/MULTI_AGENT_ARCHITECTURE.md`
- `docs/AGENT_API.md`
- `README.md`
