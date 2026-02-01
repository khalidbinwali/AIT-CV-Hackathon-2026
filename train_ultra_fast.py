"""
ULTRA-FAST TRAINING - 1 HOUR TO 95+ mAP@50
Strategy: Aggressive fine-tuning from best checkpoint
"""

from ultralytics import YOLO
import torch
import os

def ultra_fast_training():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Load your best checkpoint
    best_checkpoint = 'runs/detect/runs/detect/space_station_medium_final/weights/best.pt'
    
    print("="*70)
    print("⚡ ULTRA-FAST TRAINING MODE - 1 HOUR TO 95+ mAP@50")
    print("="*70)
    print(f"Loading: {best_checkpoint}")
    print("Current: 0.79 mAP@50 → Target: 0.95+ mAP@50")
    print("="*70 + "\n")
    
    model = YOLO(best_checkpoint)
    
    # ULTRA-AGGRESSIVE SETTINGS FOR MAXIMUM mAP IN MINIMAL TIME
    model.train(
        data='yolo_params.yaml',
        
        # FAST TRAINING
        epochs=50,                     # Only 50 epochs (~1.5 hours max)
        patience=15,                   # Stop if no improvement
        batch=8,                       # Larger batch for speed
        imgsz=640,
        
        # AGGRESSIVE FINE-TUNING
        lr0=0.0001,                    # Very low LR (fine-tuning)
        lrf=0.0001,                    # Keep LR stable
        warmup_epochs=1.0,             # Minimal warmup
        cos_lr=False,                  # Linear for faster convergence
        
        # MAXIMUM AUGMENTATION FOR CLASS BALANCE
        copy_paste=0.5,                # ✅ MAXIMUM copy-paste
        mixup=0.5,                     # ✅ MAXIMUM mixup
        mosaic=1.0,
        
        # Aggressive augmentations
        hsv_h=0.03,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=25.0,
        translate=0.2,
        scale=0.8,
        shear=3.0,
        perspective=0.0003,
        flipud=0.0,
        fliplr=0.5,
        erasing=0.5,                   # Maximum erasing
        auto_augment='randaugment',
        
        # RECALL OPTIMIZATION
        iou=0.5,                       # ✅ VERY LOW (maximize recall)
        cls=1.5,                       # ✅ VERY HIGH class loss
        box=8.0,                       # Higher box loss
        dfl=1.5,
        
        # MULTI-SCALE
        multi_scale=0.6,               # More aggressive
        close_mosaic=15,
        
        # PERFORMANCE
        device=0,
        workers=8,
        cache=False,                   # Skip caching (faster start)
        amp=True,
        
        # OUTPUT
        project='runs/detect',
        name='ULTRA_FAST_95mAP',
        exist_ok=True,
        save=True,
        save_period=5,                 # Save every 5 epochs
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("Next: Run validation with TTA for final boost!")
    print("="*70 + "\n")

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
        ultra_fast_training()
    else:
        print("ERROR: No GPU detected!")
