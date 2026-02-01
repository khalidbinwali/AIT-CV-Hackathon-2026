from ultralytics import YOLO
import os

def run_test():
    # 1. Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Load your BEST trained model weights
    # Adjusted path to match user's structure
    model_path = os.path.join(current_dir, 'runs/detect/runs/detect/space_station_medium_final/weights/best.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        # Try fallback path just in case
        model_path = os.path.join(current_dir, 'runs/detect/space_station_medium_final/weights/best.pt')
        if os.path.exists(model_path):
            print(f"Found model at fallback path: {model_path}")
        else:
            return

    model = YOLO(model_path)

    # 3. Run Validation specifically on the TEST set
    print("Starting evaluation on 1,400 test images...")
    
    # The 'split' argument is used to specify which subset from your YAML to use.
    # Ensure your yolo_params.yaml has a 'test:' path defined.
    metrics = model.val(
        data='yolo_params.yaml', 
        split='test',            # Specify 'test' split from your yaml
        imgsz=640,
        batch=16,
        augment=True,            # ✅ Enable Test-Time Augmentation (TTA) for max mAP
        conf=0.2,                # Optimal confidence from our earlier test
        iou=0.5,                 # Optimal IoU from our earlier test
        device=0,                # Use your RTX 3050
        save_json=True           # Saves results for hackathon reporting [cite: 78]
    )

    # [cite_start]4. Print Official Results for Submission [cite: 194, 195]
    # [cite_start]4. Print Official Results for Submission [cite: 194, 195]
    print("\n--- TEST RESULTS ---")
    print(f"mAP@50 (Primary Metric): {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"Results saved in: {metrics.save_dir}")
    
    # Save to file
    with open('test_results.txt', 'w') as f:
        f.write("--- TEST RESULTS ---\n")
        f.write(f"mAP@50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@50-95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n")
        f.write(f"Results dir: {metrics.save_dir}\n")

if __name__ == '__main__':
    run_test()