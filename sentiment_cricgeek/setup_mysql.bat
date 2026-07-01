@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I "%~1"=="/?" goto help
if /I "%~1"=="/help" goto help
if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help

if not exist ".env" (
    echo Missing .env. Copy .env.example to .env and set the required secrets first.
    exit /b 1
)

for /f "usebackq eol=# tokens=1* delims==" %%A in (".env") do (
    if not "%%~A"=="" set "%%A=%%B"
)

if not defined DB_PASSWORD (
    echo DB_PASSWORD is missing from .env.
    exit /b 1
)

if not defined MYSQL_ROOT_PASSWORD (
    echo MYSQL_ROOT_PASSWORD is missing from .env.
    exit /b 1
)

where mysql >nul 2>nul
if not errorlevel 1 (
    echo Found local MySQL client. Initializing database with mysql_init.sql...
    mysql --host=localhost --port=3306 --user=root --password=!MYSQL_ROOT_PASSWORD! < mysql_init.sql
    if errorlevel 1 (
        echo MySQL initialization failed.
        exit /b 1
    )

    echo Creating application tables and sample data...
    python database.py init
    if errorlevel 1 exit /b 1

    python database.py seed
    if errorlevel 1 exit /b 1

    echo MySQL setup completed using the local client.
    exit /b 0
)

where docker >nul 2>nul
if errorlevel 1 (
    echo Neither mysql nor docker is available in PATH.
    exit /b 1
)

echo Local MySQL client not found. Starting the MySQL-backed CricGeek Docker stack instead...
docker compose -f docker-compose.mysql.yml up --build
exit /b %ERRORLEVEL%

:help
echo Usage: setup_mysql.bat
echo Loads values from .env, initializes MySQL with the local client when available, and falls back to Docker Compose.
echo.
echo Prerequisites:
echo   - .env created from .env.example
echo   - mysql.exe on PATH for local setup, or Docker Desktop for the container fallback
exit /b 0