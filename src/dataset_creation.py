import os
import random
import json
import yaml
from pathlib import Path

# Configuration matching the paper review
PHASES = {
    'phase0': 40,  # Base knowledge (40 classes)
    'phase1': 10,  # Low light/hard classes
    'phase2': 10,  # Occlusion
    'phase3': 10,  # Novel
    'phase4': 10   # Adversarial
}
SEED = 42

def create_clcoco_splits(output_dir="datasets"):
    """
    Simulates the creation of CL-COCO-80 splits.
    In a full run, this would split the COCO instances_train2017.json.
    """
    random.seed(SEED)
    print(f"Generating CL-COCO-80 configuration structure in {output_dir}...")
    
    # Standard COCO 80 classes
    all_classes = list(range(80)) 
    random.shuffle(all_classes)
    
    start = 0
    for phase_name, count in PHASES.items():
        end = start + count
        phase_classes = all_classes[start:end]
        
        # Create Directory Structure
        phase_dir = Path(output_dir) / f"clcoco_{phase_name}"
        (phase_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (phase_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        
        # Create Metadata
        meta = {
            "phase": phase_name,
            "classes": phase_classes,
            "class_count": count
        }
        with open(phase_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        # Create YOLO data.yaml for this phase
        data_config = {
            'path': str(phase_dir.absolute()),
            'train': 'images/train',
            'val': 'images/train', # Using train as val for dummy structure
            'names': {i: f"class_{i}" for i in phase_classes}
        }
        with open(phase_dir / "data.yaml", "w") as f:
            yaml.dump(data_config, f)

        print(f"  [+] Created {phase_dir} ({count} categories)")
        start = end
    
    print("\nDataset structure ready. Place COCO images in respective folders to begin training.")

if __name__ == "__main__":
    create_clcoco_splits()