# Saved Models — Post-Symptom Diagnosis

This directory stores the trained model artifacts for the **Post-Symptom Diagnosis** module (TerraMind Model 3).

## Expected structure

```
saved_models/
└── trained_artifacts_fast/
    ├── best_plant_disease_model_fast.pth          # PyTorch checkpoint (fallback)
    ├── class_metadata_fast.json                   # Class names, crop mapping, img_size
    └── plant_disease_model_fast_torchscript.pt    # TorchScript export (primary inference)
```

## How to populate

Copy the contents of the externally-trained `trained_artifacts_fast/` folder into `saved_models/trained_artifacts_fast/`.

## Usage

The inference code in `ml/post_symptom_diagnosis/inference/model_wrapper.py` resolves this path automatically relative to the project root. **Do not use absolute paths.**
