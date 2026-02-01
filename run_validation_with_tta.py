"""
VALIDATION WITH TEST-TIME AUGMENTATION (TTA)
Run this AFTER training completes to get maximum mAP scores

TTA applies augmentations during inference and averages the results,
which typically boosts mAP by 1-3 percentage points.
"""

from ultralytics import YOLO
import os

def run_validation_with_tta():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Use the best checkpoint from enhanced training
    best_model = 'runs/detect/space_station_ENHANCED_v2/weights/best.pt'
    
    # Fallback to original if new training hasn't completed
    if not os.path.exists(best_model):
        best_model = 'runs/detect/runs/detect/space_station_medium_final/weights/best.pt'
        print(f"⚠️ Enhanced model not found, using original: {best_model}")
    
    print(f"\n✅ Loading model: {best_model}\n")
    model = YOLO(best_model)
    
    # Run validation WITH Test-Time Augmentation
    print("="*70)
    print("🔬 Running Validation WITH Test-Time Augmentation (TTA)")
    print("="*70 + "\n")
    
    results_tta = model.val(
        data='yolo_params.yaml',
        augment=True,              # ✅ ENABLE TTA
        conf=0.20,                 # Lower confidence for better recall
        iou=0.45,                  # Lower IoU for NMS
        max_det=300,
        plots=True,
        save_json=True,            # Save results in COCO format
        verbose=True
    )
    
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS WITH TTA:")
    print("="*70)
    print(f"mAP@0.5:       {results_tta.box.map50:.4f}")
    print(f"mAP@0.5-0.95:  {results_tta.box.map:.4f}")
    print(f"Precision:     {results_tta.box.p:.4f}")
    print(f"Recall:        {results_tta.box.r:.4f}")
    print("="*70 + "\n")
    
    # Compare with regular validation
    print("🔬 Running Validation WITHOUT TTA (for comparison)")
    results_normal = model.val(
        data='yolo_params.yaml',
        augment=False,             # No TTA
        conf=0.25,
        iou=0.45,
        max_det=300
    )
    
    print("\n" + "="*70)
    print("📊 COMPARISON: TTA vs Normal Validation")
    print("="*70)
    print(f"              | Normal   | TTA      | Improvement")
    print("-"*70)
    print(f"mAP@0.5       | {results_normal.box.map50:.4f}   | {results_tta.box.map50:.4f}   | {(results_tta.box.map50 - results_normal.box.map50)*100:+.2f}%")
    print(f"mAP@0.5-0.95  | {results_normal.box.map:.4f}   | {results_tta.box.map:.4f}   | {(results_tta.box.map - results_normal.box.map)*100:+.2f}%")
    print(f"Precision     | {results_normal.box.p:.4f}   | {results_tta.box.p:.4f}   | {(results_tta.box.p - results_normal.box.p)*100:+.2f}%")
    print(f"Recall        | {results_normal.box.r:.4f}   | {results_tta.box.r:.4f}   | {(results_tta.box.r - results_normal.box.r)*100:+.2f}%")
    print("="*70 + "\n")
    
    print("💾 Results saved in validation runs folder")
    print("✅ Use TTA results for your hackathon submission!\n")

if __name__ == '__main__':
    run_validation_with_tta()
