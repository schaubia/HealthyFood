# 🚀 Quick Start Guide

Get your Food Health Analyzer up and running in 5 minutes!

## Option 1: Automated Setup (Recommended)

### Windows
1. Open Command Prompt in the project folder
2. Run: `setup.bat`
3. Run: `python app.py`
4. Open: http://127.0.0.1:7860

### macOS/Linux
1. Open Terminal in the project folder
2. Run: `chmod +x setup.sh && ./setup.sh`
3. Run: `source venv/bin/activate`
4. Run: `python app.py`
5. Open: http://127.0.0.1:7860

## Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

## Option 3: Docker

```bash
docker-compose up
```

Then open: http://127.0.0.1:7860

## Testing Your Installation

```bash
python test_installation.py
```

This will verify all dependencies are installed correctly.

## First Time Use

1. **Upload a food image** - Click the upload box
2. **Click "Analyze Food"** - Wait a few seconds (first run downloads the model)
3. **View results** - See food recognition, nutrition info, and health rating!

## Getting Better Results

### Optional: USDA API Key

For unlimited nutrition data access:

1. Sign up: https://fdc.nal.usda.gov/api-key-signup.html
2. Set your key:
   - Windows: `set USDA_API_KEY=your_key_here`
   - macOS/Linux: `export USDA_API_KEY=your_key_here`

### Tips for Best Photos

✅ **DO:**
- Use good lighting
- Center the food
- Take clear, focused photos
- Use common foods

❌ **DON'T:**
- Use blurry/dark photos
- Include multiple dishes
- Photograph complex recipes
- Use very rare foods

## Common Issues

### "Command not found: python"
Try `python3` instead of `python`

### "ModuleNotFoundError"
Make sure you activated the virtual environment!

### Model downloads slowly
This is normal on first run (~100MB download)

### Port already in use
Change the port in app.py or stop the other service

## What's Next?

- 📖 Read the full [README.md](README.md)
- 🚀 Deploy to production: [DEPLOYMENT.md](DEPLOYMENT.md)
- 🤝 Contribute: [CONTRIBUTING.md](CONTRIBUTING.md)
- ❓ Questions: [FAQ.md](FAQ.md)

## Need Help?

- 🐛 Report bugs on GitHub Issues
- 💬 Ask questions in Discussions
- 📧 Check the FAQ

---

**Ready to analyze some food? Let's go! 🍎**

```bash
python app.py
```
