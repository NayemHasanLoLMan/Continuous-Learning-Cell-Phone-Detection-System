import os
import shutil
from pathlib import Path

def create_directories():
    """Create all necessary directories."""
    directories = [
        "models",
        "models/previous_versions",
        "datasets/initial_dataset/images/train",
        "datasets/initial_dataset/images/val",
        "datasets/initial_dataset/labels/train",
        "datasets/initial_dataset/labels/val",
        "datasets/captured_data/pending_verification/images",
        "datasets/captured_data/verified_positive/images",
        "datasets/captured_data/verified_positive/labels",
        "datasets/captured_data/verified_negative/images",
        "datasets/training_batches",
        "scripts",
        "scripts/utils",
        "config",
        "logs/detection_logs",
        "logs/training_logs",
        "logs/verification_logs",
        "runs/detect",
    ]
    
    print(" Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("  All directories created")

def copy_model():
    """Find and copy existing model."""
    print("\n Looking for trained model...")
    
    # Possible locations
    locations = [
        "runs/detect/cellphone_yolo11m/weights/best.pt",
        "runs/detect/cellphone_improved/weights/best.pt",
        "runs/detect/improved_model/weights/best.pt",
        "best.pt",
        "test.pt"  # Your current model
    ]
    
    dest = "models/current_best.pt"
    
    for loc in locations:
        if os.path.exists(loc):
            shutil.copy2(loc, dest)
            print(f"   Copied model: {loc} → {dest}")
            return True
    
    print("    No model found. Please place your trained model at:")
    print(f"     {dest}")
    return False

def create_config():
    """Create default config file."""
    print("\n  Creating configuration...")
    
    config_path = "config/config.yaml"
    
    if os.path.exists(config_path):
        print(f"    Config already exists: {config_path}")
        return True
    
    config_content = """# Continuous Learning System Configuration

model:
  current_model_path: "models/current_best.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45

data_collection:
  capture_interval: 2.0
  min_confidence_for_capture: 0.15
  max_captures_per_session: 50
  save_format: "jpg"

gemini:
  api_key: "YOUR_GEMINI_API_KEY"  # Get from: https://makersuite.google.com/app/apikey
  model: "gemini-1.5-flash"
  verification_prompt: |
    Is there a cell phone (mobile phone, smartphone) visible in this image?
    Analyze carefully and respond with only 'true' or 'false'.
    - true: if you can see a cell phone/smartphone
    - false: if there is no cell phone or you're not sure
  batch_size: 10

retraining:
  trigger_mode: "batch"
  batch_size: 50
  epochs: 20
  learning_rate: 0.0001
  validation_split: 0.2
  min_map_improvement: 0.01

performance:
  min_map_improvement: 0.01
  track_metrics: true
  send_alerts: false
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"   Created: {config_path}")
    print("    IMPORTANT: Update your Gemini API key in this file!")
    return True

def create_init_files():
    """Create __init__.py files for imports."""
    print("\n Creating Python package files...")
    
    # scripts/__init__.py
    with open("scripts/__init__.py", 'w') as f:
        f.write('"""Scripts package"""\n')
    
    # scripts/utils/__init__.py
    with open("scripts/utils/__init__.py", 'w') as f:
        f.write('''"""Utils package"""
from .dataset_manager import DatasetManager
from .model_manager import ModelManager
from .metrics_tracker import MetricsTracker

__all__ = ['DatasetManager', 'ModelManager', 'MetricsTracker']
''')
    
    print("   Created __init__.py files")

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
env/
ENV/

# Project specific
models/*.pt
!models/.gitkeep
datasets/
runs/
logs/
*.log

# API keys
.env
config/config.yaml

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", 'w') as f:
        f.write(gitignore_content)
    
    print("   Created .gitignore")

def check_dependencies():
    """Check if required packages are installed."""
    print("\n Checking dependencies...")
    
    required = [
        'torch',
        'ultralytics',
        'cv2',
        'google.generativeai',
        'yaml',
        'PIL'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"   {package}")
        except ImportError:
            print(f"   {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print("\n  Missing packages detected!")
        print("  Run: pip install -r requirements.txt")
        return False
    
    return True

def check_gpu():
    """Check GPU availability."""
    print("\n  Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU available: {gpu_name}")
            return True
        else:
            print("    No GPU detected - will use CPU (slower)")
            return False
    except:
        print("    Could not check GPU")
        return False

def main():
    """Run complete setup."""
    print("="*70)
    print(" CONTINUOUS LEARNING SYSTEM - SETUP")
    print("="*70)
    
    # Step 1: Create directories
    print("\n[1/7] Creating directory structure...")
    create_directories()
    
    # Step 2: Copy model
    print("\n[2/7] Setting up model...")
    model_found = copy_model()
    
    # Step 3: Create config
    print("\n[3/7] Creating configuration...")
    create_config()
    
    # Step 4: Create __init__ files
    print("\n[4/7] Creating package files...")
    create_init_files()
    
    # Step 5: Create .gitignore
    print("\n[5/7] Creating .gitignore...")
    create_gitignore()
    
    # Step 6: Check dependencies
    print("\n[6/7] Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Step 7: Check GPU
    print("\n[7/7] Checking hardware...")
    check_gpu()
    
    # Summary
    print("\n" + "="*70)
    print(" SETUP SUMMARY")
    print("="*70)
    
    issues = []
    if not model_found:
        issues.append(" Model not found - place your model at models/current_best.pt")
    else:
        print(" Model ready")
    
    if not deps_ok:
        issues.append(" Missing dependencies - run: pip install -r requirements.txt")
    else:
        print(" Dependencies installed")
    
    print(" Directories created")
    print(" Configuration created")
    
    if issues:
        print("\n  Issues to fix:")
        for issue in issues:
            print(f"  {issue}")
    
    print("\n" + "="*70)
    print(" NEXT STEPS:")
    print("="*70)
    
    if not deps_ok:
        print("1. Install dependencies:")
        print("   pip install -r requirements.txt")
        print()
    
    if not model_found:
        print("2. Place your trained model:")
        print("   Copy best.pt to models/current_best.pt")
        print()
    
    print("3. Get Gemini API key:")
    print("   https://makersuite.google.com/app/apikey")
    print()
    
    print("4. Update config/config.yaml with your API key")
    print()
    
    print("5. Run the system:")
    print("   python run_system.py")
    print("   OR")
    print("   python scripts/continuous_learning.py")
    
    print("="*70)
    
    if model_found and deps_ok:
        print("\n Setup complete! You're ready to go!")
    else:
        print("\n  Please fix the issues above before running the system")

if __name__ == "__main__":
    main()