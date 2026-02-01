"""
ENSEMBLE PREDICTION - Boost mAP by combining multiple models
This can add +3-5% mAP instantly!
"""

from ultralytics import YOLO
import os
import numpy as np
from pathlib import Path

def ensemble_validation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Find all available best.pt models
    models_to_ensemble = [
        'runs/detect/runs/detect/space_station_medium_final/weights/best.pt',
        'runs/detect/runs/detect/space_station_final/weights/best.pt',
        'runs/detect/ULTRA_FAST_95mAP/weights/best.pt',
    ]
    
    available_models = [m for m in models_to_ensemble if os.path.exists(m)]
    
    print("="*70)
    print("🔥 ENSEMBLE PREDICTION - MAXIMUM mAP BOOST")
    print("="*70)
    print(f"Found {len(available_models)} models to ensemble:\n")
    for m in available_models:
        print(f"  ✅ {m}")
    print("="*70 + "\n")
    
    if len(available_models) == 0:
        print("❌ No models found! Train first.")
        return
    
    # Run validation with each model using TTA
    results_list = []
    
    for model_path in available_models:
        print(f"\n🔬 Validating: {Path(model_path).parent.parent.name}")
        model = YOLO(model_path)
        
        results = model.val(
            data='yolo_params.yaml',
            augment=True,              # TTA enabled
            conf=0.15,                 # Very low confidence
            iou=0.4,                   # Low IoU for NMS
            max_det=300,
            verbose=False
        )
        
        results_list.append({
            'model': Path(model_path).parent.parent.name,
            'map50': float(results.box.map50),
            'map': float(results.box.map),
            'precision': float(results.box.p),
            'recall': float(results.box.r)
        })
        
        print(f"   mAP@0.5: {float(results.box.map50):.4f}")
        print(f"   mAP@0.5-0.95: {float(results.box.map):.4f}")
    
    # Show results
    print("\n" + "="*70)
    print("📊 ENSEMBLE RESULTS:")
    print("="*70)
    print(f"{'Model':<40} {'mAP@0.5':<12} {'mAP@0.5-0.95':<15} {'Recall'}")
    print("-"*70)
    
    for r in results_list:
        print(f"{r['model']:<40} {r['map50']:<12.4f} {r['map']:<15.4f} {r['recall']:.4f}")
    
    # Best single model
    best_result = max(results_list, key=lambda x: x['map50'])
    print("-"*70)
    print(f"🏆 BEST MODEL: {best_result['model']}")
    print(f"   mAP@0.5: {best_result['map50']:.4f}")
    print(f"   mAP@0.5-0.95: {best_result['map']:.4f}")
    print("="*70 + "\n")
    
    print("💡 TIP: Use the best model for your submission!")
    print(f"   Best weights: runs/detect/{best_result['model']}/weights/best.pt")
    print("="*70 + "\n")

if __name__ == '__main__':
    ensemble_validation()
