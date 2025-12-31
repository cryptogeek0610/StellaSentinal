@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   SOTI Stella Sentinel - Docker Installation Script
echo ============================================================
echo.

:: Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo.
    echo Please install Docker Desktop from:
    echo   https://www.docker.com/products/docker-desktop/
    echo.
    echo After installation, restart this script.
    pause
    exit /b 1
)

echo [OK] Docker is installed.

:: Check if Docker is running
docker info >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not running.
    echo.
    echo Please start Docker Desktop and wait for it to be ready.
    echo Then restart this script.
    pause
    exit /b 1
)

echo [OK] Docker is running.
echo.

:: Check if .env file exists
if not exist ".env" (
    echo [INFO] Creating .env file from template...
    copy env.template .env >nul
    echo [OK] Created .env file.
    echo.
    echo ============================================================
    echo   IMPORTANT: Configure your environment variables
    echo ============================================================
    echo.
    echo Please edit the .env file to configure:
    echo   1. Database credentials (BACKEND_DB_PASS, DW_DB_*, MC_DB_*)
    echo   2. XSight SQL Server connection (DW_DB_HOST, etc.)
    echo   3. Optional: LLM settings for AI explanations
    echo   4. Optional: MobiControl API credentials
    echo.
    echo Opening .env file for editing...
    notepad .env
    echo.
    echo Press any key after you have saved the .env file to continue...
    pause >nul
) else (
    echo [OK] .env file already exists.
)

echo.
echo ============================================================
echo   Building Docker Images
echo ============================================================
echo.
echo This may take several minutes on first run...
echo.

docker-compose build
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Docker build failed.
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo [OK] Docker images built successfully.
echo.

:: Ask if user wants to start the containers
echo ============================================================
echo   Start Application?
echo ============================================================
echo.
set /p START_NOW="Do you want to start the application now? (Y/N): "
if /i "!START_NOW!"=="Y" (
    echo.
    echo Starting containers...
    docker-compose up -d
    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] Failed to start containers.
        pause
        exit /b 1
    )

    echo.
    echo ============================================================
    echo   Application Started Successfully!
    echo ============================================================
    echo.
    echo Waiting for services to be ready...
    timeout /t 10 /nobreak >nul

    echo.
    echo Access the application at:
    echo   - Frontend:    http://localhost:3000
    echo   - API:         http://localhost:8000
    echo   - API Docs:    http://localhost:8000/docs
    echo.
    echo Container Status:
    docker-compose ps
    echo.
    echo ============================================================
    echo   Useful Commands
    echo ============================================================
    echo.
    echo   View logs:           docker-compose logs -f
    echo   Stop application:    docker-compose down
    echo   Restart:             docker-compose restart
    echo   Pull Ollama model:   docker exec stellasentinel-ollama ollama pull llama3.2
    echo.
) else (
    echo.
    echo ============================================================
    echo   Installation Complete
    echo ============================================================
    echo.
    echo To start the application later, run:
    echo   docker-compose up -d
    echo.
)

echo Press any key to exit...
pause >nul
