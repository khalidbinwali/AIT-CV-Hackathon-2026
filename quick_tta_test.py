"""
QUICK TTA VALIDATION - Find best mAP with optimized settings
"""

from ultralytics import YOLO
import os

def quick_tta_validation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Your best model
    best_model = 'runs/detect/runs/detect/space_station_medium_final/weights/best.pt'
    
    print("="*70)
    print("⚡ QUICK TTA VALIDATION - OPTIMIZED FOR MAXIMUM mAP")
    print("="*70)
    print(f"Model: {best_model}\n")
    
    model = YOLO(best_model)
    
    # Test multiple confidence/IoU combinations to find best mAP
    configs = [
        {'conf': 0.001, 'iou': 0.3, 'name': 'Ultra-Low Thresholds'},
        {'conf': 0.01, 'iou': 0.4, 'name': 'Very Low Thresholds'},
        {'conf': 0.15, 'iou': 0.45, 'name': 'Low Thresholds'},
        {'conf': 0.20, 'iou': 0.5, 'name': 'Moderate Thresholds'},
        {'conf': 0.25, 'iou': 0.6, 'name': 'Default Thresholds'},
    ]
    
    best_map50 = 0
    best_config = None
    
    for config in configs:
        print(f"\n🔬 Testing: {config['name']}")
        print(f"   conf={config['conf']}, iou={config['iou']}")
        
        results = model.val(
            data='yolo_params.yaml',
            augment=True,              # TTA enabled
            conf=config['conf'],
            iou=config['iou'],
            max_det=500,
            verbose=False
        )
        
        map50 = float(results.box.map50)
        map_all = float(results.box.map)
        
        print(f"   ✅ mAP@0.5: {map50:.4f} ({map50*100:.2f}%)")
        print(f"   ✅ mAP@0.5-0.95: {map_all:.4f}")
        
        if map50 > best_map50:
            best_map50 = map50
            best_config = config
    
    print("\n" + "="*70)
    print("🏆 BEST RESULT:")
    print("="*70)
    print(f"Configuration: {best_config['name']}")
    print(f"   conf={best_config['conf']}, iou={best_config['iou']}")
    print(f"\n   🎯 mAP@0.5: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print("="*70 + "\n")
    
    if best_map50 >= 0.95:
        print("🎉 SUCCESS! You achieved 95%+ mAP@50!")
    elif best_map50 >= 0.85:
        print("💪 Great! 85%+ mAP@50 is competitive!")
    elif best_map50 >= 0.80:
        print("👍 Good! 80%+ mAP@50 is solid!")
    else:
        print("📊 Current mAP. Consider training for improvement.")
    
    print("\n💡 Next: Start ultra-fast training to improve further!")
    print("   Command: C:\\Users\\ZAH\\anaconda3\\envs\\EDU\\python.exe train_ultra_fast.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    quick_tta_validation()
