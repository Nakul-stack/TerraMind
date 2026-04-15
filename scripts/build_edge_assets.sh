#!/bin/bash
echo "═══════════════════════════════════════"
echo "  TerraMind — Building Edge Assets"
echo "═══════════════════════════════════════"
cd "$(dirname "$0")/.."
python -m backend.models.compress_edge_model
echo ""
echo "═══ Edge assets built! ═══"
