# 🎯 Food Health Analyzer

An AI-powered application that analyzes food images and provides detailed nutritional information and health ratings.

## 🆕 Two Versions Available

### 1. **Streamlit Version** (Cloud Deployment) ☁️
**File:** `streamlit_app.py`

**Features:**
- 🎯 Hybrid AI model (ViT + ResNet50)
- 📊 Point-based health scoring (1-10 scale)
- ⚠️ Advanced allergen detection
- 🧠 User feedback and learning
- 📋 Detailed ingredient breakdown
- 🌐 Perfect for Streamlit Cloud deployment

**Deploy to Streamlit Cloud:**
```bash
# Already configured with streamlit_app.py
# Just connect your GitHub repo to Streamlit Cloud
```

**Run Locally:**
```bash
streamlit run streamlit_app.py
```

### 2. **Gradio Version with Custom Training** (Local Use) 💻
**File:** `app_with_training.py`

**Features:**
- 🎓 **Train custom models on your own images**
- 🔍 Hybrid recognition (ImageNet + custom)
- 📊 Nutritional analysis from USDA
- 💚 Health rating system
- 🎯 Perfect for personal food recognition

**Run Locally:**
```bash
python app_with_training.py
```

---

## ✨ Key Features Comparison

| Feature | Streamlit (Cloud) | Gradio (Local + Training) |
|---------|------------------|---------------------------|
| Food Recognition | ✅ Advanced (ViT + ResNet) | ✅ Standard (ResNet) |
| Nutrition Data | ✅ USDA API | ✅ USDA API |
| Health Scoring | ✅ 1-10 Point System | ✅ 3-Level System |
| Allergen Detection | ✅ Comprehensive | ❌ Not included |
| **Custom Training** | ❌ Not available | ✅ **YES!** |
| User Feedback | ✅ Learning system | ❌ Not included |
| Deployment | ✅ Cloud-ready | 🏠 Local only |
| Best For | Public/shared use | Personal/custom needs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) USDA API key for better nutrition data

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Schaubia/HealthyFood.git
cd HealthyFood
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Get USDA API Key** (optional but recommended):
- Sign up at [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html)
- Set environment variable:

```bash
# Windows
set USDA_API_KEY=your_api_key_here

# macOS/Linux
export USDA_API_KEY=your_api_key_here
```

### Running the Apps

#### Option 1: Streamlit Version
```bash
streamlit run streamlit_app.py
```
Opens at: `http://localhost:8501`

#### Option 2: Gradio with Training
```bash
python app_with_training.py
```
Opens at: `http://127.0.0.1:7860`

---

## 🎓 Training Your Custom Model (Gradio Version Only)

The Gradio version includes a powerful custom training feature!

### Why Train a Custom Model?

- Recognize regional/local dishes not in standard databases
- Identify your specific recipes and preparations
- Improve accuracy on foods you eat frequently
- Support branded products or specific varieties

### Quick Training Guide

1. **Collect Images**: Take 10-20 photos of each food type
2. **Add to Training Set**: Upload and label in the Train tab
3. **Train Model**: Click "Start Training" (takes 5-15 minutes)
4. **Use Custom Model**: Check the option in Analyze tab

📚 **Full Guide:** See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions

### Training Example

```
Before Training:
Upload: Homemade curry photo
Result: "Stew - 45%", "Soup - 32%"

After Training (10 images):
Upload: Same curry
Result: "My Homemade Curry - 89%" 🎯
```

---

## 💻 Usage

### Analyzing Food (Both Versions)

1. **Upload Image**: Click upload area and select food photo
2. **Analyze**: Click the analyze button
3. **View Results**:
   - Food identification with confidence scores
   - Nutritional information
   - Health rating and recommendations

### Tips for Best Results

- ✅ Use clear, well-lit photos
- ✅ Center the food in frame
- ✅ Avoid overly complex multi-ingredient dishes
- ✅ Single food items work best
- ✅ Keep background simple

---

## 📊 How It Works

### Food Recognition

**Streamlit Version:**
- Vision Transformer (ViT) for complex dishes
- ResNet50 for ingredients and simple foods
- 101 food classes

**Gradio Version:**
- ResNet50 with ImageNet (1000 categories)
- Custom trained model (your foods)
- Hybrid prediction combining both

### Nutritional Analysis

Both versions:
- Query **USDA FoodData Central API**
- Retrieve detailed nutrient information
- Display key nutrients per 100g serving

### Health Rating

**Streamlit (1-10 Points):**
- 8-10 🟢: Highly nutritious
- 5-7 🟡: Moderate/neutral
- 1-4 🔴: Consume sparingly

**Gradio (3 Levels):**
- Healthy 🟢: Nutrient-rich
- Moderate 🟡: Balanced
- Unhealthy 🔴: High in fats/sugars/sodium

---

## 🛠️ Setup Scripts

We provide automated setup for both platforms:

### Windows
```bash
setup.bat
```

### macOS/Linux
```bash
chmod +x setup.sh
./setup.sh
```

These scripts will:
- Check Python version
- Create virtual environment
- Install dependencies
- Verify USDA API key
- Create necessary directories

---

## 📁 Project Structure

```
food-health-analyzer/
├── streamlit_app.py              # Streamlit cloud version
├── app_with_training.py          # Gradio local version with training
├── app.py                        # Original Gradio version (basic)
├── requirements.txt              # Python dependencies
├── TRAINING_GUIDE.md            # Complete training guide
├── README.md                    # This file
├── setup.sh                     # Linux/Mac setup script
├── setup.bat                    # Windows setup script
├── test_installation.py         # Installation verification
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose setup
│
├── training_data/               # Custom training images (Gradio)
│   ├── food_name_1/
│   ├── food_name_2/
│   └── ...
│
├── models/                      # Trained models (Gradio)
│   ├── custom_food_model.h5
│   └── class_mapping.pkl
│
└── examples/                    # Example images (optional)
```

---

## 🔧 Configuration

### Custom Model Training (Gradio)

Default settings in `app_with_training.py`:
```python
TRAINING_DATA_DIR = "training_data"
CUSTOM_MODEL_PATH = "models/custom_food_model.h5"
```

### Health Scoring

Modify health scores in either app:
- Streamlit: Edit `health_scores` dictionary (line 46-104)
- Gradio: Edit `analyze_health()` method

---

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and run
docker-compose up -d

# Stop
docker-compose down
```

### Using Dockerfile

```bash
# Build image
docker build -t food-analyzer .

# Run container
docker run -p 7860:7860 \
  -e USDA_API_KEY=your_key \
  -v $(pwd)/models:/app/models \
  food-analyzer
```

---

## 📝 API Rate Limits

**USDA FoodData Central:**
- Demo key: 1,000 requests/hour
- Registered key: Higher limits (free)
- Get key: https://fdc.nal.usda.gov/api-key-signup.html

**Recommendations:**
- Use registered API key for production
- Implement caching for frequent queries
- Add request delays if hitting limits

---

## 🐛 Troubleshooting

### Installation Issues

**"ModuleNotFoundError"**
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Then install
pip install -r requirements.txt
```

**"Python version error"**
- Need Python 3.8+
- Check with: `python --version`
- Update if necessary

### Runtime Issues

**"API returns no data"**
- Try simpler food names
- Get registered USDA API key
- Check internet connection

**"Model download slow"**
- First run downloads ~100MB
- Normal behavior
- Subsequent runs are fast

**Training Issues (Gradio)**
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) troubleshooting section

### Memory Issues

**High memory usage:**
- Close other applications
- Reduce batch size (training)
- Use smaller model if possible
- Recommended: 4GB+ RAM

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for recipe analysis (multiple ingredients)
- [ ] Meal tracking and history
- [ ] Calorie calculator
- [ ] Export nutrition reports
- [ ] Mobile app version
- [ ] Integration with fitness trackers
- [ ] More allergen detection
- [ ] Multi-language support

---

## 📄 License

This project is open source. Please see individual component licenses:
- TensorFlow: Apache 2.0
- Gradio: Apache 2.0
- Streamlit: Apache 2.0

---

## 🙏 Acknowledgments

- **USDA FoodData Central** for nutritional data
- **ImageNet** for pre-trained models
- **Hugging Face** for Vision Transformer
- **TensorFlow/Keras** for deep learning framework
- **Gradio & Streamlit** for web interfaces

---

## 📮 Contact & Support

**Issues?** Open an issue on GitHub
**Questions?** Check the guides:
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training help
- In-app Help tab for usage questions

---

## 🎯 Quick Decision Guide

**Choose Streamlit if:**
- ✅ Want to deploy to cloud
- ✅ Need allergen detection
- ✅ Want 1-10 health scores
- ✅ Need user feedback system
- ✅ Want ingredient breakdown

**Choose Gradio if:**
- ✅ Want custom model training
- ✅ Need to recognize specific foods
- ✅ Running locally only
- ✅ Want simple 3-level ratings
- ✅ Have unique dietary needs

**Use Both!**
- 🎯 Streamlit for general public/cloud
- 💻 Gradio locally for personal training

---

Made with ❤️ for healthier eating!
