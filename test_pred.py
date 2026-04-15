import sys, traceback
from pathlib import Path

# Fix relative imports
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

try:
    from ml.growth_stage_monitor.predict import main
    main()
    print("Success")
except Exception as e:
    with open("error_full.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Failed")
