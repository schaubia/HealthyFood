# 🍎 Food Health Analyzer

An AI-powered web application that analyzes food images and provides detailed nutritional information and health ratings.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🔍 **AI Food Recognition**: Uses ResNet50 deep learning model to identify food items
- 📊 **Nutritional Analysis**: Fetches real nutritional data from USDA FoodData Central
- 💚 **Health Rating**: Provides health scores based on nutritional content
- 🌐 **Web Interface**: Easy-to-use Gradio interface accessible via browser
- ⚡ **Fast Processing**: Quick analysis with pre-trained models

## 🎯 What It Does

1. **Upload** a photo of your food
2. **Identify** the food item using AI (top 5 predictions with confidence scores)
3. **Analyze** nutritional content from USDA database
4. **Rate** the healthiness of the food (Healthy 🟢, Moderate 🟡, Unhealthy 🔴)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) USDA API key for better nutrition data access

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/food-health-analyzer.git
cd food-health-analyzer
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up USDA API Key** (optional but recommended)

Get a free API key from [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html)

```bash
# On Windows
set USDA_API_KEY=your_api_key_here

# On macOS/Linux
export USDA_API_KEY=your_api_key_here
```

### Running the Application

```bash
python app.py
```

The application will start and provide a local URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to use the app!

## 💻 Usage

1. **Upload an Image**: Click on the upload box and select a food image from your device
2. **Analyze**: Click the "🔍 Analyze Food" button
3. **View Results**: See:
   - Food identification results with confidence scores
   - Detailed nutritional information
   - Health rating and recommendations

### Tips for Best Results

- Use clear, well-lit photos
- Center the food item in the frame
- Avoid overly complex dishes with many ingredients
- Single food items work best

## 🔧 Configuration

### Using Custom Model Weights

If you have trained custom food classification weights, you can modify the `FoodHealthAnalyzer` class:

```python
# In app.py, add this method to the class:
def load_custom_weights(self, weights_path):
    """Load custom trained weights"""
    self.model.load_weights(weights_path)
```

### Adjusting Health Rating Criteria

Modify the `analyze_health()` method in `app.py` to change how foods are rated based on your preferences.

## 📊 How It Works

### Food Recognition
- Uses **ResNet50** pre-trained on ImageNet
- Identifies food items from 1000+ classes
- Provides confidence scores for top 5 predictions

### Nutritional Analysis
- Queries **USDA FoodData Central API**
- Retrieves detailed nutrient information
- Displays key nutrients per 100g serving

### Health Rating Algorithm
The app analyzes nutrients and assigns ratings:

**Positive Factors** (increase health score):
- High protein content
- High fiber content
- Vitamins and minerals

**Negative Factors** (decrease health score):
- High fat content (especially saturated fats)
- High sugar content
- High sodium content
- High cholesterol

## 🛠️ Project Structure

```
food-health-analyzer/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # License information
├── .gitignore           # Git ignore rules
│
├── examples/            # Example food images (optional)
│   ├── burger.jpg
│   ├── salad.jpg
│   └── pizza.jpg
│
└── models/              # Custom model weights (optional)
    └── food_weights.h5
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more example images
- Improve health rating algorithm
- Add support for multiple food items in one image
- Implement barcode scanning
- Add meal planning features
- Support for different languages

## 📝 API Rate Limits

**USDA FoodData Central API:**
- Demo key: 1,000 requests per hour
- Registered key: Higher limits available (free registration)

If you encounter rate limit errors, consider:
- Getting a registered API key
- Implementing caching for frequent queries
- Adding request delays

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'tensorflow'`
- **Solution**: Ensure you've activated your virtual environment and installed requirements

**Issue**: API returns no nutrition data
- **Solution**: Try searching for simpler food names or get a registered USDA API key

**Issue**: Model downloads slowly on first run
- **Solution**: This is normal - ResNet50 weights (~100MB) download on first use

**Issue**: High memory usage
- **Solution**: Close other applications or use a machine with more RAM (recommended: 4GB+)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow** and **Keras** teams for the deep learning framework
- **USDA FoodData Central** for nutritional data
- **Gradio** team for the amazing web interface library
- **ImageNet** for the pre-trained model weights

## 📧 Contact

Have questions or suggestions? Feel free to open an issue or reach out!

## 🔮 Future Enhancements

- [ ] Support for recipe analysis (multiple ingredients)
- [ ] Meal tracking and history
- [ ] Calorie calculator
- [ ] Dietary restriction filters (vegan, gluten-free, etc.)
- [ ] Export nutrition reports
- [ ] Mobile app version
- [ ] Integration with fitness trackers
- [ ] Custom diet plan recommendations

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ and 🤖 AI**
