import sys
from pathlib import Path

# Add root and backend to python path to resolve modules correctly
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / 'backend'))

from backend.app.schemas.advisor import BeforeSowingRequest
from backend.app.services.pre_sowing_pipeline import run_pre_sowing_pipeline

def test():
    req = BeforeSowingRequest(
        N=90, P=40, K=40, 
        temperature=28, 
        humidity=72, 
        rainfall=200, 
        ph=6.5,
        soil_type="loamy", 
        season="kharif", 
        state_name="karnataka", 
        district_name="mysore"
    )
    
    print("Running pipeline...")
    try:
        res = run_pre_sowing_pipeline(req)
        print("\n--- PIPELINE SUCCESS ---")
        print(f"Crop: {res.recommended_crop}")
        print(f"Yield: {res.predicted_yield}")
        print(f"Sunlight: {res.sunlight_hours}")
        print(f"Irrigation Type: {res.irrigation_type}")
        print(f"Irrigation Need: {res.irrigation_need}")
    except Exception as e:
        print(f"\n--- PIPELINE FAILED ---")
        print(e)
        raise e

if __name__ == "__main__":
    test()
