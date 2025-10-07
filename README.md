# 🔄 Continuous Learning Cell Phone Detection System

A self-improving cell phone detection system using YOLO11m and Google Gemini Vision API for continuous learning.

## 🌟 Features

- **Real-time Detection**: Detect cell phones via webcam
- **Automatic Data Collection**: Capture detections for verification
- **AI Verification**: Use Gemini Vision API to verify detections
- **Continuous Learning**: Automatically retrain model with verified data
- **Model Version Control**: Track and rollback model versions
- **Performance Monitoring**: Track improvements over time

---

## 📋 Quick Start

### 1. Installation

```bash
# Clone or create project directory
mkdir continuous_cellphone_detection
cd continuous_cellphone_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run quick setup
python setup.py
```

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy your API key
5. Update `config/config.yaml`:

```yaml
gemini:
  api_key: "YOUR_ACTUAL_API_KEY_HERE"
```

### 3. Prepare Initial Model

```bash
# If you have COCO-trained model at: runs/detect/cellphone_yolo11m/weights/best.pt
# It will be automatically copied to models/current_best.pt

# Or improve with additional datasets first:
python scripts/1_improve_initial_model.py
```

### 4. Start Continuous Learning

```bash
# Run complete system
python scripts/5_continuous_learning.py

# Or run with specific number of cycles
python scripts/5_continuous_learning.py --cycles 5
```

---

## 📁 Project Structure

```
continuous_cellphone_detection/
│
├── models/
│   ├── current_best.pt              # Active production model
│   ├── previous_versions/           # Archived models
│   └── training_history.json        # Performance history
│
├── datasets/
│   ├── initial_dataset/             # Base training data
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── captured_data/
│   │   ├── pending_verification/    # Awaiting Gemini check
│   │   ├── verified_positive/       # Confirmed phones
│   │   └── verified_negative/       # False positives
│   │
│   └── training_batches/            # Historical training data
│
├── scripts/
│   ├── 1_improve_initial_model.py   # Add Roboflow data
│   ├── 2_webcam_capture.py          # Capture detections
│   ├── 3_gemini_verification.py     # Verify with Gemini
│   ├── 4_retrain_model.py           # Retrain pipeline
│   └── 5_continuous_learning.py     # Full automation
│
├── config/
│   └── config.yaml                  # System configuration
│
├── logs/
│   ├── detection_logs/
│   ├── training_logs/
│   └── verification_logs/
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────┐
│                  CONTINUOUS LEARNING LOOP                │
└─────────────────────────────────────────────────────────┘

1. WEBCAM CAPTURE
   ├─ Run detection on live video
   ├─ Capture frames when phone detected
   └─ Save images + detection metadata
           ↓
2. GEMINI VERIFICATION
   ├─ Send captured images to Gemini API
   ├─ Get true/false verification
   └─ Sort into positive/negative folders
           ↓
3. ACCUMULATION
   ├─ Wait until 50+ verified samples
   └─ Combine with existing dataset
           ↓
4. RETRAINING
   ├─ Train model for 20 epochs
   ├─ Validate performance
   └─ Compare with baseline
           ↓
5. DEPLOYMENT
   ├─ If improved → Deploy new model
   ├─ Archive old model
   └─ Reset capture counter
           ↓
   (Loop back to step 1)
```

---

## 🚀 Individual Scripts

### Script 1: Improve Initial Model

Combines your COCO dataset with additional Roboflow datasets.

```bash
python scripts/1_improve_initial_model.py
```

**What it does:**
- Downloads/uses Roboflow datasets (5000+ images)
- Combines with existing COCO data
- Trains for 50 epochs from your best.pt
- Creates improved baseline model

**Output:**
- `models/current_best.pt` - Improved model
- `models/previous_versions/` - Original model archived

---

### Script 2: Webcam Capture

Captures images from webcam when phones detected.

```bash
python scripts/2_webcam_capture.py
```

**Controls:**
- `q` - Quit
- `c` - Force capture
- `s` - Skip next capture

**Configuration:**
```yaml
data_collection:
  capture_interval: 2.0        # Seconds between captures
  min_confidence_for_capture: 0.15
  max_captures_per_session: 50
```

**Output:**
- `datasets/captured_data/pending_verification/images/` - Captured images
- `datasets/captured_data/pending_verification/detections.json` - Metadata

---

### Script 3: Gemini Verification

Verifies captured images using Gemini Vision API.

```bash
python scripts/3_gemini_verification.py
```

**What it does:**
- Loads pending captures
- Sends to Gemini for verification
- Sorts into positive/negative folders
- Creates YOLO format labels for positives

**Output:**
- `datasets/captured_data/verified_positive/` - Confirmed phones
- `datasets/captured_data/verified_negative/` - False positives
- `logs/verification_logs/` - Verification history

**API Limits:**
- Free tier: 60 requests/minute
- Script includes rate limiting

---

### Script 4: Automated Retraining

Retrains model when enough verified data accumulated.

```bash
python scripts/4_retrain_model.py
```

**Trigger conditions:**
- ≥50 verified positive samples
- Batch mode (configurable)

**What it does:**
- Combines initial + verified data
- Retrains for 20 epochs
- Validates performance
- Deploys if improved (≥1% mAP gain)

**Output:**
- New model (if improved)
- Training metrics
- Performance comparison

---

### Script 5: Continuous Learning

Orchestrates complete continuous learning loop.

```bash
# Run indefinitely
python scripts/5_continuous_learning.py

# Run 5 cycles
python scripts/5_continuous_learning.py --cycles 5

# Custom config
python scripts/5_continuous_learning.py --config my_config.yaml
```

**One cycle includes:**
1. Capture phase (50 images)
2. Verification phase (Gemini check)
3. Retraining phase (if ready)

---

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
model:
  current_model_path: "models/current_best.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45

data_collection:
  capture_interval: 2.0              # Capture every 2 seconds
  min_confidence_for_capture: 0.15   # Low threshold to catch edge cases
  max_captures_per_session: 50       # Stop after 50 captures

gemini:
  api_key: "YOUR_API_KEY"
  model: "gemini-1.5-flash"
  batch_size: 10

retraining:
  trigger_mode: "batch"              # 'batch' or 'continuous'
  batch_size: 50                     # Retrain after 50 samples
  epochs: 20
  learning_rate: 0.0001
  validation_split: 0.2

performance:
  min_map_improvement: 0.01          # Deploy if +1% mAP improvement
  track_metrics: true
```

---

## 📊 Performance Tracking

### View Training History

```python
import json

with open('models/training_history.json', 'r') as f:
    history = json.load(f)

for entry in history:
    print(f"Model: {entry['timestamp']}")
    print(f"  mAP@0.5: {entry['metrics']['map50']:.4f}")
    print(f"  Recall:  {entry['metrics']['recall']:.4f}")
```

### View Verification Logs

```python
import json

with open('logs/verification_logs/verification_log.json', 'r') as f:
    logs = json.load(f)

positive = sum(1 for log in logs if log['is_phone'])
total = len(logs)
print(f"Accuracy: {positive/total*100:.1f}%")
```

---

## 🎯 Expected Results

### Initial Model (COCO only)
- Precision: 75.93%
- Recall: 59.92%
- mAP@0.5: 65.95%

### After Adding Roboflow Data
- Precision: ~82-85%
- Recall: ~75-80%
- mAP@0.5: ~78-82%

### After 3-5 Learning Cycles
- Precision: ~85-88%
- Recall: ~80-85%
- mAP@0.5: ~82-87%
- Fewer false positives
- Better edge case handling

---

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, install matching CUDA toolkit
# PyTorch 2.0+ → CUDA 11.8 or 12.1
```

### Gemini API Errors
- **Rate limit**: Free tier = 60 req/min
- **Invalid key**: Check config.yaml
- **Quota exceeded**: Upgrade plan or wait

### Out of Memory
```yaml
# Reduce batch size in config
retraining:
  batch_size: 25  # Instead of 50
```

### Webcam Not Working
```python
# Try different camera index
cap = cv2.VideoCapture(1)  # Instead of 0
```

### Low Verification Accuracy
- Check Gemini prompt clarity
- Review false positives/negatives
- Adjust capture confidence threshold

---

## 💡 Best Practices

### Data Quality
- ✅ Capture diverse angles and lighting
- ✅ Include different phone models
- ✅ Capture challenging scenarios
- ❌ Avoid duplicate/similar images
- ❌ Avoid low-quality blurry images

### Retraining Strategy
- Start with small batches (50 samples)
- Monitor improvement trends
- Don't deploy if no improvement
- Keep last 5 model versions

### API Usage
- Use Gemini free tier wisely (60/min)
- Batch verification when possible
- Monitor API costs if upgraded
- Cache verification results

### System Monitoring
- Check logs regularly
- Track performance metrics
- Monitor disk space
- Backup models periodically

---

## 🔐 Security & Privacy

### API Keys
```bash
# Use environment variables
export GEMINI_API_KEY="your_key"

# Or use .env file (add to .gitignore)
echo "GEMINI_API_KEY=your_key" > .env
```

### Privacy Considerations
- Blur faces if capturing public spaces
- Delete sensitive data regularly
- Comply with local privacy laws
- Don't upload to public datasets

---

## 📈 Monitoring Dashboard (Optional)

Create a simple monitoring dashboard:

```python
# dashboard.py
import streamlit as st
import json
import pandas as pd

st.title("📊 Continuous Learning Dashboard")

# Load training history
with open('models/training_history.json', 'r') as f:
    history = json.load(f)

df = pd.DataFrame([h['metrics'] for h in history])
st.line_chart(df)

# Show stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Models", len(history))
col2.metric("Best mAP", f"{df['map50'].max():.3f}")
col3.metric("Improvement", f"+{(df['map50'].max() - df['map50'].min()):.3f}")
```

Run with:
```bash
streamlit run dashboard.py
```

---

## 🤝 Contributing

Ideas for improvement:
- [ ] Add web interface for monitoring
- [ ] Support multiple object classes
- [ ] Add data augmentation options
- [ ] Integrate with cloud storage
- [ ] Add email/SMS alerts
- [ ] Support video file processing
- [ ] Add A/B testing framework

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🙏 Acknowledgments

- **Ultralytics YOLO** - Object detection framework
- **Google Gemini** - Vision verification
- **COCO Dataset** - Initial training data
- **Roboflow** - Additional datasets

---

## 📞 Support

Issues? Questions?
1. Check troubleshooting section
2. Review logs in `logs/`
3. Check configuration in `config/config.yaml`

---

## 🎓 Learning Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Active Learning Guide](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))

---

**Happy Learning! 🚀** 

The more you use it, the better it gets!