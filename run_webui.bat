@echo off
REM Launches the GIS Co-Scientist WebUI using the dedicated virtual environment.
set VENV_PYTHON=%USERPROFILE%\.venvs\gis-co-scientist\Scripts\python.exe
set APP_PATH=%~dp0WebUI\app.py

if not exist "%VENV_PYTHON%" (
    echo Could not find the virtual environment at "%VENV_PYTHON%".
    echo Please recreate it before running this script.
    pause
    exit /b 1
)

echo Starting GIS Co-Scientist WebUI...
echo Once it says "Running on http://127.0.0.1:4041", open http://127.0.0.1:4041/ in your browser.
echo Press Ctrl+C in this window to stop the server.
echo.

"%VENV_PYTHON%" "%APP_PATH%"

pause
