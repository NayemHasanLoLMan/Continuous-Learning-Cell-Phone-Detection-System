import argparse
import os
import shutil
from ultralytics import YOLO
from pathlib import Path
# Note: gatekeeper import is used in the full active loop, 
# here we implement the training side of the MC-ACD framework.

def train_mcacd(phase, seed=42, epochs=20):
    print(f"\n{'='*40}")
    print(f" Starting MC-ACD Training: {phase}")
    print(f"{'='*40}")
    
    # 1. Model Initialization strategy
    if phase == 'phase0':
        print("  Loading pretrained YOLOv11m...")
        model = YOLO('yolo11m.pt') 
    else:
        # Load previous phase weights (Continuous Learning)
        prev_phase_idx = int(phase.replace('phase','')) - 1
        model_path = f"checkpoints/phase{prev_phase_idx}.pt"
        
        if not os.path.exists(model_path):
            print(f"  [Error] Previous checkpoint {model_path} not found.")
            print("  Please train previous phases first.")
            return
            
        print(f"  Loading state from Phase {prev_phase_idx}...")
        model = YOLO(model_path)

    # 2. Configuration
    data_yaml = f"datasets/clcoco_{phase}/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"  [Error] Dataset config {data_yaml} not found. Run dataset_creation.py first.")
        return

    # 3. Training (Rehearsal enabled via dataset mixing in yaml)
    print("  Beginning training loop...")
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,
            seed=seed,
            project="runs/train",
            name=f"mcacd_{phase}",
            augment=True,      # Mosaic augmentation active
            verbose=True,
            exist_ok=True
        )
    except Exception as e:
        print(f"  [Training Error] {e}")
        return

    # 4. Save Academic Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{phase}.pt"
    
    # Export/Copy weights
    train_run_dir = f"runs/train/mcacd_{phase}/weights/best.pt"
    if os.path.exists(train_run_dir):
        shutil.copy(train_run_dir, save_path)
        print(f"\n[Success] Phase complete. Weights saved to: {save_path}")
    else:
        print("\n[Warning] Could not locate best.pt weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, help="e.g., phase0, phase1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_mcacd(args.phase, args.seed)