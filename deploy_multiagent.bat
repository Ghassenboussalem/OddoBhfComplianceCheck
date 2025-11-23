@echo off
REM Multi-Agent Compliance Checker Deployment Script (Windows CMD)
REM This script deploys the LangGraph-based multi-agent compliance system

setlocal enabledelayedexpansion

REM Configuration
set PYTHON_CMD=python
set BACKUP_DIR=backups\deployment_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%
set CONFIG_FILE=hybrid_config.json
set REQUIREMENTS_FILE=requirements.txt
set DEPLOYMENT_LOG=deployment_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set DEPLOYMENT_LOG=%DEPLOYMENT_LOG: =0%

echo ======================================================================== >> %DEPLOYMENT_LOG%
echo Multi-Agent System Deployment - %date% %time% >> %DEPLOYMENT_LOG%
echo ======================================================================== >> %DEPLOYMENT_LOG%

REM Print header
echo.
echo ========================================================================
echo MULTI-AGENT COMPLIANCE CHECKER DEPLOYMENT
echo ========================================================================
echo.
echo This script will deploy the LangGraph-based multi-agent system
echo Press Ctrl+C to cancel at any time
echo.
echo Deployment log: %DEPLOYMENT_LOG%
echo.
timeout /t 2 /nobreak >nul

REM Step 1: Check Python installation
echo ========================================================================
echo CHECKING PYTHON INSTALLATION
echo ========================================================================
echo.

%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher
    echo [ERROR] Python not found >> %DEPLOYMENT_LOG%
    goto :error
)

for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python found: %PYTHON_VERSION%
echo [SUCCESS] Python found: %PYTHON_VERSION% >> %DEPLOYMENT_LOG%

REM Check Python version (basic check for 3.x)
echo %PYTHON_VERSION% | findstr /r "^3\.[89]" >nul
if errorlevel 1 (
    echo %PYTHON_VERSION% | findstr /r "^3\.1[0-9]" >nul
    if errorlevel 1 (
        echo [WARNING] Python 3.8 or higher is recommended. Found: %PYTHON_VERSION%
        echo [WARNING] Python version: %PYTHON_VERSION% >> %DEPLOYMENT_LOG%
    )
)

echo [SUCCESS] Python version is compatible
echo [SUCCESS] Python version is compatible >> %DEPLOYMENT_LOG%
echo.

REM Step 2: Create backup
echo ========================================================================
echo CREATING BACKUP
echo ========================================================================
echo.

if not exist backups mkdir backups
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
echo [INFO] Backup directory: %BACKUP_DIR%
echo [INFO] Backup directory: %BACKUP_DIR% >> %DEPLOYMENT_LOG%

REM Backup configuration files
if exist %CONFIG_FILE% (
    copy /y %CONFIG_FILE% "%BACKUP_DIR%\" >nul
    echo [SUCCESS] Backed up %CONFIG_FILE%
    echo [SUCCESS] Backed up %CONFIG_FILE% >> %DEPLOYMENT_LOG%
)

if exist .env (
    copy /y .env "%BACKUP_DIR%\" >nul
    echo [SUCCESS] Backed up .env
    echo [SUCCESS] Backed up .env >> %DEPLOYMENT_LOG%
)

if exist review_queue.json (
    copy /y review_queue.json "%BACKUP_DIR%\" >nul
    echo [SUCCESS] Backed up review_queue.json
    echo [SUCCESS] Backed up review_queue.json >> %DEPLOYMENT_LOG%
)

if exist audit_logs (
    xcopy /e /i /y audit_logs "%BACKUP_DIR%\audit_logs" >nul
    echo [SUCCESS] Backed up audit_logs\
    echo [SUCCESS] Backed up audit_logs\ >> %DEPLOYMENT_LOG%
)

if exist checkpoints (
    xcopy /e /i /y checkpoints "%BACKUP_DIR%\checkpoints" >nul
    echo [SUCCESS] Backed up checkpoints\
    echo [SUCCESS] Backed up checkpoints\ >> %DEPLOYMENT_LOG%
)

echo [SUCCESS] Backup completed: %BACKUP_DIR%
echo [SUCCESS] Backup completed >> %DEPLOYMENT_LOG%
echo.

REM Step 3: Install dependencies
echo ========================================================================
echo INSTALLING DEPENDENCIES
echo ========================================================================
echo.

if not exist %REQUIREMENTS_FILE% (
    echo [ERROR] requirements.txt not found
    echo [ERROR] requirements.txt not found >> %DEPLOYMENT_LOG%
    goto :error
)

echo [INFO] Upgrading pip...
echo [INFO] Upgrading pip >> %DEPLOYMENT_LOG%
%PYTHON_CMD% -m pip install --upgrade pip >> %DEPLOYMENT_LOG% 2>&1

echo [INFO] Installing Python packages from %REQUIREMENTS_FILE%...
echo [INFO] Installing dependencies >> %DEPLOYMENT_LOG%
%PYTHON_CMD% -m pip install -r %REQUIREMENTS_FILE% >> %DEPLOYMENT_LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo [ERROR] Failed to install dependencies >> %DEPLOYMENT_LOG%
    goto :error
)

echo [SUCCESS] Dependencies installed successfully
echo [SUCCESS] Dependencies installed >> %DEPLOYMENT_LOG%
echo.

REM Step 4: Validate configuration
echo ========================================================================
echo VALIDATING CONFIGURATION
echo ========================================================================
echo.

if not exist %CONFIG_FILE% (
    echo [ERROR] Configuration file %CONFIG_FILE% not found
    echo [ERROR] Configuration file not found >> %DEPLOYMENT_LOG%
    goto :error
)

echo [SUCCESS] Configuration file found: %CONFIG_FILE%
echo [SUCCESS] Configuration file found >> %DEPLOYMENT_LOG%

if not exist .env (
    echo [WARNING] .env file not found
    echo [WARNING] .env file not found >> %DEPLOYMENT_LOG%
    if exist .env.example (
        echo [INFO] Copying .env.example to .env
        copy /y .env.example .env >nul
        echo [WARNING] Please configure your API keys in .env file
        echo [WARNING] .env needs configuration >> %DEPLOYMENT_LOG%
    )
) else (
    echo [SUCCESS] .env file found
    echo [SUCCESS] .env file found >> %DEPLOYMENT_LOG%
)

echo [INFO] Validating configuration structure...
echo [INFO] Validating configuration >> %DEPLOYMENT_LOG%
%PYTHON_CMD% -c "import json; config = json.load(open('%CONFIG_FILE%')); print('Configuration is valid JSON')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Invalid JSON in configuration file
    echo [ERROR] Invalid JSON in configuration >> %DEPLOYMENT_LOG%
    goto :error
)

echo [SUCCESS] Configuration structure is valid
echo [SUCCESS] Configuration validated >> %DEPLOYMENT_LOG%
echo.

REM Step 5: Create required directories
echo ========================================================================
echo CREATING REQUIRED DIRECTORIES
echo ========================================================================
echo.

set DIRS=checkpoints audit_logs monitoring\logs monitoring\metrics monitoring\visualizations

for %%d in (%DIRS%) do (
    if not exist %%d (
        mkdir %%d
        echo [SUCCESS] Created directory: %%d
        echo [SUCCESS] Created directory: %%d >> %DEPLOYMENT_LOG%
    ) else (
        echo [INFO] Directory already exists: %%d
    )
)
echo.

REM Step 6: Run basic validation test
echo ========================================================================
echo RUNNING VALIDATION TEST
echo ========================================================================
echo.

set /p RUN_TESTS="Run validation test? (y/n): "
if /i "%RUN_TESTS%"=="y" (
    if exist exemple.json (
        echo [INFO] Running basic validation test...
        echo [INFO] Running validation test >> %DEPLOYMENT_LOG%
        %PYTHON_CMD% check_multiagent.py exemple.json --show-metrics >> %DEPLOYMENT_LOG% 2>&1
        if errorlevel 1 (
            echo [WARNING] Validation test completed with violations (expected)
            echo [WARNING] Validation test had violations >> %DEPLOYMENT_LOG%
        ) else (
            echo [SUCCESS] Basic validation test passed
            echo [SUCCESS] Validation test passed >> %DEPLOYMENT_LOG%
        )
    ) else (
        echo [WARNING] exemple.json not found, skipping validation test
        echo [WARNING] exemple.json not found >> %DEPLOYMENT_LOG%
    )
) else (
    echo [INFO] Skipping tests
    echo [INFO] Tests skipped by user >> %DEPLOYMENT_LOG%
)
echo.

REM Step 7: Enable multi-agent mode
echo ========================================================================
echo ENABLING MULTI-AGENT MODE
echo ========================================================================
echo.

echo [INFO] Updating configuration to enable multi-agent mode...
echo [INFO] Enabling multi-agent mode >> %DEPLOYMENT_LOG%

%PYTHON_CMD% -c "import json; config = json.load(open('%CONFIG_FILE%')); config.setdefault('multi_agent', {})['enabled'] = True; json.dump(config, open('%CONFIG_FILE%', 'w'), indent=2)" >> %DEPLOYMENT_LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to update configuration
    echo [ERROR] Failed to update configuration >> %DEPLOYMENT_LOG%
    goto :error
)

echo [SUCCESS] Multi-agent mode enabled
echo [SUCCESS] Multi-agent mode enabled >> %DEPLOYMENT_LOG%
echo.

REM Step 8: Display summary
echo ========================================================================
echo DEPLOYMENT SUMMARY
echo ========================================================================
echo.
echo [SUCCESS] Multi-Agent Compliance Checker deployed successfully!
echo [SUCCESS] Deployment completed >> %DEPLOYMENT_LOG%
echo.
echo [INFO] Backup location: %BACKUP_DIR%
echo [INFO] Configuration: %CONFIG_FILE%
echo [INFO] Entry point: check_multiagent.py
echo [INFO] Deployment log: %DEPLOYMENT_LOG%
echo.
echo Usage:
echo   python check_multiagent.py ^<json_file^> [options]
echo.
echo Examples:
echo   python check_multiagent.py exemple.json
echo   python check_multiagent.py exemple.json --show-metrics
echo   python check_multiagent.py exemple.json --review-mode
echo.
echo Features enabled:
echo   - Multi-agent architecture
echo   - Parallel execution
echo   - State persistence
echo   - Human-in-the-loop integration
echo   - AI-enhanced checking
echo   - Context-aware validation
echo.
echo Next steps:
echo   1. Configure API keys in .env file (if not already done)
echo   2. Review configuration in %CONFIG_FILE%
echo   3. Run a test: python check_multiagent.py exemple.json
echo   4. Check monitoring dashboard: python monitoring\dashboard.py
echo.
echo [SUCCESS] Deployment complete!
echo ========================================================================
echo.

goto :end

:error
echo.
echo ========================================================================
echo [ERROR] DEPLOYMENT FAILED
echo ========================================================================
echo.
echo Please check the deployment log for details: %DEPLOYMENT_LOG%
echo.
echo To restore from backup:
echo   1. Stop any running processes
echo   2. Copy files from %BACKUP_DIR% back to current directory
echo   3. Restart the application
echo.
echo ========================================================================
echo [ERROR] Deployment failed >> %DEPLOYMENT_LOG%
exit /b 1

:end
endlocal
exit /b 0
