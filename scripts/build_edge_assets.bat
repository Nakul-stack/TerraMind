@echo off
echo ═══════════════════════════════════════
echo   TerraMind — Building Edge Assets
echo ═══════════════════════════════════════
cd /d "%~dp0\.."
python -m backend.models.compress_edge_model
echo.
echo ═══ Edge assets built! ═══
pause
