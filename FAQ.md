# Frequently Asked Questions (FAQ)

## General Questions

### What is Food Health Analyzer?

Food Health Analyzer is an AI-powered web application that identifies food items from photos and provides detailed nutritional information and health ratings.

### Is it free to use?

Yes! The application is completely free and open-source under the MIT License. However, the USDA API has rate limits (1000 requests/hour with DEMO_KEY, higher with a free registered key).

### What technologies does it use?

- **AI Model**: ResNet50 (pre-trained on ImageNet)
- **Framework**: TensorFlow/Keras
- **Web Interface**: Gradio
- **Nutrition Data**: USDA FoodData Central API
- **Languages**: Python

### How accurate is the food recognition?

The accuracy depends on several factors:
- Image quality and lighting
- How common the food item is
- Complexity of the dish
- Typical accuracy: 70-90% for common foods

### Does it work offline?

Partially. The AI model works offline after the first download, but nutritional data requires an internet connection to access the USDA API.

## Technical Questions

### What are the system requirements?

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection (for initial setup)

**Recommended:**
- Python 3.10+
- 8GB RAM
- SSD storage
- Stable internet connection

### Why is the first run so slow?

The ResNet50 model (~100MB) downloads automatically on first run. Subsequent runs will be much faster.

### Can I run it on GPU?

Yes! If you have TensorFlow-GPU installed and a compatible GPU, the app will automatically use it for faster inference.

### How do I get better API rate limits?

1. Sign up for a free API key at [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html)
2. Set it as an environment variable:
   - Windows: `set USDA_API_KEY=your_key`
   - Unix: `export USDA_API_KEY=your_key`

### Can I train it on custom foods?

The current version uses pre-trained ImageNet weights. To train on custom foods:
1. Collect and label food images
2. Modify the `FoodHealthAnalyzer` class to load custom weights
3. Train using transfer learning from ResNet50

See CONTRIBUTING.md for more details.

### Does it support multiple languages?

Currently, the interface is in English only. Multi-language support is planned for future releases.

## Usage Questions

### What types of food work best?

**Best results:**
- Common, well-known foods
- Single items (apple, burger, pizza)
- Clear, well-lit photos
- Centered food items

**May struggle with:**
- Complex dishes with many ingredients
- Unusual or rare foods
- Home-cooked meals
- Low-quality or blurry images

### Can it identify multiple foods in one image?

Currently, the app identifies the most prominent food item. Multi-food detection is planned for future releases.

### Why isn't my food recognized correctly?

Common reasons:
1. **Poor image quality**: Use better lighting and focus
2. **Uncommon food**: The model may not have been trained on it
3. **Complex dish**: Try photographing individual components
4. **Ambiguous appearance**: Some foods look very similar

### What if nutrition data isn't available?

This can happen if:
- The food isn't in the USDA database
- The food name doesn't match USDA naming conventions
- API rate limits are exceeded

Try:
- Using a simpler food name
- Getting a registered API key
- Searching for similar foods

### How is the health rating calculated?

The health rating considers:

**Positive factors:**
- High protein
- High fiber
- Vitamins and minerals

**Negative factors:**
- High fat (especially saturated)
- High sugar
- High sodium
- High cholesterol

Rating scale:
- 🟢 **Healthy**: More positive than negative factors
- 🟡 **Moderate**: Balanced
- 🔴 **Unhealthy**: More negative factors

### Can I save my analysis history?

Currently, the app doesn't save history. This feature is planned for future releases. As a workaround, you can:
- Take screenshots
- Export the browser page as PDF
- Note down the information manually

## Deployment Questions

### How do I deploy it to production?

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides on:
- Heroku
- Google Cloud Platform
- AWS
- Hugging Face Spaces
- Docker deployment

### Can I use it as an API?

Yes! You can modify the code to expose REST API endpoints. Example:

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    # Process image and return results
    pass
```

### How do I add authentication?

For production deployments, consider adding:
- Basic authentication
- OAuth integration
- API key system
- Rate limiting per user

Example with Gradio:
```python
demo.launch(auth=("username", "password"))
```

## Troubleshooting

### "ModuleNotFoundError" errors

**Solution:**
1. Activate your virtual environment
2. Run `pip install -r requirements.txt`

### "API rate limit exceeded" error

**Solutions:**
1. Get a registered USDA API key
2. Wait an hour for limits to reset
3. Implement caching for common queries

### The app is running but I can't access it

**Check:**
1. Is it running on the correct port? (default: 7860)
2. Is your firewall blocking the port?
3. Are you using the correct URL? (http://127.0.0.1:7860)

### Docker container fails to start

**Common issues:**
1. Port already in use: `docker ps` to check
2. Insufficient resources: Increase Docker memory limit
3. Image not built: Run `docker-compose build`

### Model prediction takes too long

**Solutions:**
1. Use a GPU if available
2. Reduce image size before processing
3. Deploy on a more powerful server
4. Implement batch processing

## Privacy & Security

### What data is collected?

The application:
- ✅ Processes images locally (not stored)
- ✅ Makes API calls to USDA (public data)
- ❌ Does NOT store user images
- ❌ Does NOT collect personal information

### Is my food data private?

Yes. Images are processed in your browser/server and not sent anywhere except for the USDA API call (which only sends the food name, not the image).

### Can I use it for medical purposes?

**No.** This tool is for informational purposes only. It should NOT be used for:
- Medical diagnosis
- Treatment planning
- Dietary prescriptions
- Managing medical conditions

Always consult healthcare professionals for medical advice.

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code contributions
- Bug reports
- Feature requests
- Documentation improvements

### I found a bug. What should I do?

1. Check if it's already reported in GitHub Issues
2. Create a new issue with:
   - Description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Your environment details

### I have a feature idea!

Great! Please:
1. Check existing feature requests
2. Create a new issue with the "enhancement" label
3. Describe the use case and proposed solution

## License & Legal

### What license is this under?

MIT License - you can freely use, modify, and distribute this software. See [LICENSE](LICENSE) for full terms.

### Can I use it commercially?

Yes! The MIT License allows commercial use. Just keep the copyright notice and license in your copies.

### What about the USDA data?

USDA FoodData Central data is in the public domain and free to use. Attribution is appreciated but not required.

---

**Still have questions?** 

- 📧 Open an issue on GitHub
- 📖 Check our [Documentation](README.md)
- 💬 Join the discussions
