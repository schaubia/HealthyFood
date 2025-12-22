# 📁 Project Structure

## Overview

This document describes the organization of the Food Health Analyzer repository.

## Directory Structure

```
food-health-analyzer/
│
├── .github/                          # GitHub specific files
│   ├── workflows/
│   │   └── python-app.yml           # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md            # Bug report template
│   │   └── feature_request.md       # Feature request template
│   └── PULL_REQUEST_TEMPLATE.md     # PR template
│
├── examples/                         # Example food images
│   └── README.md                    # Guide for adding examples
│
├── models/                          # Model weights directory
│   └── README.md                    # Model documentation
│
├── app.py                           # Main application file ⭐
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── setup.sh                         # Unix/Linux setup script
├── setup.bat                        # Windows setup script
├── test_installation.py             # Installation test script
│
├── Dockerfile                       # Docker container definition
├── docker-compose.yml               # Docker Compose configuration
│
├── README.md                        # Main documentation ⭐
├── QUICKSTART.md                    # Quick start guide
├── CONTRIBUTING.md                  # Contribution guidelines
├── DEPLOYMENT.md                    # Deployment guide
├── FAQ.md                          # Frequently asked questions
├── CHANGELOG.md                     # Version history
├── LICENSE                         # MIT License
└── PROJECT_STRUCTURE.md            # This file
```

## File Descriptions

### Core Files

#### `app.py` ⭐
The main application file containing:
- `FoodHealthAnalyzer` class - AI model wrapper
- Food recognition logic
- Nutrition data fetching
- Health rating algorithm
- Gradio web interface
- Main execution block

**Key Functions:**
- `analyze_food_image()` - Main processing function
- `get_usda_nutrition()` - Fetches nutrition data
- `analyze_health()` - Calculates health rating

#### `requirements.txt`
Python package dependencies:
- tensorflow (AI model)
- gradio (web interface)
- pillow (image processing)
- numpy (numerical operations)
- requests (API calls)

### Setup & Testing

#### `setup.sh` / `setup.bat`
Automated setup scripts that:
- Check Python version
- Create virtual environment
- Install dependencies
- Create directories
- Provide usage instructions

#### `test_installation.py`
Validates installation by testing:
- Package imports
- Model loading
- API connectivity
- Image processing

### Documentation

#### `README.md` ⭐
Main project documentation:
- Project overview
- Features
- Installation instructions
- Usage guide
- Configuration options
- Project structure
- Contributing guidelines

#### `QUICKSTART.md`
5-minute quick start guide for:
- Automated setup
- Manual setup
- Docker deployment
- First time use

#### `CONTRIBUTING.md`
Guidelines for contributors:
- How to contribute
- Code style
- PR process
- Development setup
- Testing requirements

#### `DEPLOYMENT.md`
Comprehensive deployment guide:
- Local development
- Docker deployment
- Cloud platforms (Heroku, GCP, AWS, etc.)
- Production considerations
- Troubleshooting

#### `FAQ.md`
Common questions and answers:
- General questions
- Technical issues
- Usage tips
- Deployment help
- Privacy & security

#### `CHANGELOG.md`
Version history tracking:
- Release notes
- Features added
- Bugs fixed
- Breaking changes

### Docker Files

#### `Dockerfile`
Container definition:
- Base image: Python 3.11-slim
- System dependencies
- Python packages
- App files
- Port exposure (7860)

#### `docker-compose.yml`
Multi-container orchestration:
- Service configuration
- Port mapping
- Environment variables
- Volume mounts
- Health checks

### GitHub Integration

#### `.github/workflows/python-app.yml`
CI/CD pipeline that:
- Tests on multiple OS (Windows, Linux, macOS)
- Tests Python versions (3.8-3.11)
- Checks code syntax
- Runs import tests
- Validates builds

#### Issue Templates
- `bug_report.md` - Standardized bug reporting
- `feature_request.md` - Feature suggestions

#### `PULL_REQUEST_TEMPLATE.md`
PR checklist ensuring:
- Clear description
- Test coverage
- Documentation updates
- Code quality

### Directories

#### `examples/`
Storage for example food images:
- Test images
- Demo images
- README with guidelines

#### `models/`
Storage for custom model weights:
- Custom trained models
- Model documentation
- Version tracking
- Training guides

### Configuration

#### `.gitignore`
Excludes from version control:
- Python cache files
- Virtual environments
- Model weights
- User data
- IDE configurations
- Log files

#### `LICENSE`
MIT License granting:
- Free use
- Modification rights
- Distribution rights
- Commercial use

## Development Workflow

### 1. Initial Setup
```bash
# Clone repository
git clone https://github.com/yourusername/food-health-analyzer.git

# Run setup
./setup.sh  # or setup.bat on Windows

# Test installation
python test_installation.py
```

### 2. Development
```bash
# Activate environment
source venv/bin/activate

# Make changes to app.py

# Test locally
python app.py

# Run tests
python test_installation.py
```

### 3. Contribution
```bash
# Create feature branch
git checkout -b feature/new-feature

# Commit changes
git commit -m "Add: new feature"

# Push and create PR
git push origin feature/new-feature
```

### 4. Deployment
```bash
# Docker
docker-compose up

# Or cloud platform
# See DEPLOYMENT.md for specific guides
```

## Key Technologies

### Backend
- **Python 3.8+** - Core language
- **TensorFlow/Keras** - Deep learning
- **NumPy** - Numerical computing

### Frontend
- **Gradio** - Web interface
- **HTML/CSS** - Custom styling

### AI Model
- **ResNet50** - Image recognition
- **ImageNet weights** - Pre-trained

### APIs
- **USDA FoodData Central** - Nutrition data

### DevOps
- **Docker** - Containerization
- **GitHub Actions** - CI/CD
- **Git** - Version control

## File Size Guidelines

Keep files reasonable:
- Python files: < 500 lines (split if larger)
- Documentation: < 10,000 words per file
- Images: < 2MB each
- Models: Document large files

## Naming Conventions

- **Files**: lowercase_with_underscores.py
- **Classes**: PascalCase
- **Functions**: snake_case
- **Constants**: UPPER_CASE
- **Private**: _leading_underscore

## Code Organization

The `app.py` file is organized as:
1. Imports
2. Constants & Configuration
3. FoodHealthAnalyzer class
4. Helper functions
5. Gradio interface setup
6. Main execution block

## Future Structure (Planned)

```
food-health-analyzer/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── model.py           # Model class
│   ├── nutrition.py       # Nutrition logic
│   ├── health.py          # Health rating
│   └── interface.py       # UI code
├── tests/                 # Test suite
│   ├── test_model.py
│   ├── test_nutrition.py
│   └── test_health.py
├── data/                  # Data files
└── scripts/               # Utility scripts
```

## Questions?

Refer to:
- [README.md](README.md) - Main documentation
- [FAQ.md](FAQ.md) - Common questions
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide

---

**Last Updated**: 2025-12-22
**Version**: 1.0.0
