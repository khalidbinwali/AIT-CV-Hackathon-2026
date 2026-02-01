"""
ENHANCED TRAINING SCRIPT - Maximum mAP Optimization
Target: Push mAP from 0.79-0.80 to 0.85+

Key Improvements:
- Continue from best.pt checkpoint (space_station_medium_final)
- Advanced augmentations for class imbalance (copy-paste, mixup)
- Multi-scale training enabled
- Cosine learning rate schedule
- Optimized hyperparameters for recall boost
- Enhanced loss weights (increased cls weight)
"""

from ultralytics import YOLO
import torch
import os

def start_enhanced_training():
    # 1. Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print(f"Working Directory: {current_dir}")

    # 2. Load Best Checkpoint - Continue Training
    # This is your trained model at 0.79-0.80 mAP
    best_checkpoint = os.path.join(
        current_dir, 
        'runs/detect/runs/detect/space_station_medium_final/weights/best.pt'
    )
    
    if os.path.exists(best_checkpoint):
        print(f"\n✅ Loading your best checkpoint:")
        print(f"   {best_checkpoint}")
        print(f"   Current mAP: 0.79-0.80 → Target: 0.85+\n")
        model = YOLO(best_checkpoint)
    else:
        print(f"\n⚠️ Checkpoint not found at: {best_checkpoint}")
        print("   Falling back to yolov8l.pt (larger model for higher mAP)")
        model = YOLO('yolov8l.pt')

    # 3. ENHANCED TRAINING - All Optimizations Applied
    print("🚀 Starting Enhanced Training with ALL Optimizations...\n")
    
    model.train(
        # ===== BASIC CONFIGURATION =====
        data='yolo_params.yaml',
        epochs=400,                    # More epochs for convergence
        imgsz=640,
        batch=6,                       # Reduced for stability with larger model
        patience=30,                   # More patience before early stopping
        
        # ===== OPTIMIZER SETTINGS =====
        optimizer='AdamW',
        lr0=0.0003,                    # LOWER learning rate for fine-tuning
        lrf=0.005,                     # Very low final LR
        momentum=0.95,                 # Increased momentum
        weight_decay=0.0003,           # Reduced weight decay
        warmup_epochs=5.0,             # Longer warmup
        cos_lr=True,                   # ✅ COSINE learning rate schedule
        
        # ===== LOSS WEIGHTS (CRITICAL!) =====
        box=7.5,                       # Box regression loss
        cls=1.2,                       # ✅ INCREASED class loss (was 0.5)
        dfl=1.5,                       # Distribution focal loss
        
        # ===== AUGMENTATIONS - CLASS IMBALANCE & GENERALIZATION =====
        mosaic=1.0,                    # Keep mosaic
        mixup=0.35,                    # ✅ INCREASED mixup
        copy_paste=0.35,               # ✅ NEW - Critical for class imbalance!
        copy_paste_mode='flip',
        
        # Color augmentations
        hsv_h=0.025,                   # ✅ INCREASED hue variation
        hsv_s=0.75,                    # ✅ INCREASED saturation
        hsv_v=0.45,                    # ✅ INCREASED brightness
        
        # Geometric augmentations
        degrees=20.0,                  # ✅ MORE rotation (was 15)
        translate=0.15,                # ✅ MORE translation
        scale=0.7,                     # ✅ MORE scale variation (was 0.5)
        shear=2.0,                     # ✅ NEW - Add shearing
        perspective=0.0002,            # ✅ NEW - Perspective transform
        flipud=0.0,                    # No vertical flip (space station)
        fliplr=0.5,                    # Keep horizontal flip
        
        # Advanced augmentations
        auto_augment='randaugment',    # ✅ Random augmentation policy
        erasing=0.45,                  # ✅ Random erasing (cutout)
        
        # ===== MULTI-SCALE & TRAINING TRICKS =====
        multi_scale=0.5,               # ✅ ENABLED - Multi-scale training
        close_mosaic=25,               # ✅ Close mosaic later (was 10)
        
        # ===== DETECTION OPTIMIZATION =====
        iou=0.55,                      # ✅ LOWERED IoU threshold (better recall)
        max_det=300,                   # Max detections per image
        
        # ===== PERFORMANCE =====
        device=0,                      # GPU
        workers=8,                     # Data loading workers
        cache='ram',                   # ✅ Cache images in RAM (faster!)
        amp=True,                      # Automatic Mixed Precision
        
        # ===== OUTPUT =====
        project='runs/detect',
        name='space_station_ENHANCED_v2',
        exist_ok=True,
        save=True,
        save_period=10,                # Save checkpoint every 10 epochs
        plots=True,                    # Generate plots
        verbose=True
    )
    
    print("\n" + "="*70)
    print("✅ ENHANCED TRAINING COMPLETE!")
    print("="*70)
    print("\n📊 Next Steps:")
    print("1. Check results in: runs/detect/space_station_ENHANCED_v2/")
    print("2. Review results.csv for mAP improvements")
    print("3. Run validation with TTA (Test-Time Augmentation):")
    print("   model.val(augment=True, conf=0.2, iou=0.45)")
    print("\n💡 Expected Improvements:")
    print("   - mAP@0.5: 0.85-0.90 (from 0.79-0.80)")
    print("   - mAP@0.5-0.95: 0.65-0.72 (from ~0.60)")
    print("   - Recall: 78-85% (from 72%)")
    print("="*70 + "\n")

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("="*70)
        print(f"🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("="*70 + "\n")
        start_enhanced_training()
    else:
        print("❌ ERROR: CUDA not found!")
        print("   Training on CPU will be extremely slow.")
        print("   Please ensure GPU drivers are installed.")