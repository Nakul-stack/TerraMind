@echo off
echo ═══════════════════════════════════════
echo   TerraMind — Running Benchmark
echo ═══════════════════════════════════════
cd /d "%~dp0\.."
python -m backend.models.train_local_only_model
python -m backend.services.benchmark_service
echo.
echo ═══ Benchmark complete! ═══
pause
