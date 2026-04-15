@echo off
echo ═══════════════════════════════════════
echo   TerraMind — Starting Backend
echo ═══════════════════════════════════════
cd /d "%~dp0\.."
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
