# Contributing to Food Health Analyzer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit code improvements
- 🖼️ Add example images
- 🧪 Write tests

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/food-health-analyzer.git
   cd food-health-analyzer
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Setup

1. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app** to test your changes:
   ```bash
   python app.py
   ```

## 📋 Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

### Example:
```python
def analyze_food_health(nutrients: dict) -> tuple:
    """
    Analyze health rating based on nutrient content.
    
    Args:
        nutrients: Dictionary of nutrient names and values
        
    Returns:
        tuple: (health_rating, emoji) e.g., ("Healthy", "🟢")
    """
    # Implementation here
    pass
```

## 🐛 Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: How to trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Screenshots**: If applicable
- **Environment**: OS, Python version, etc.

Use the bug report template:

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- TensorFlow Version: [e.g., 2.13.0]
```

## 💡 Suggesting Features

When suggesting features, please include:

- **Use case**: Why this feature would be useful
- **Proposed solution**: How it might work
- **Alternatives**: Other approaches you've considered
- **Additional context**: Any other relevant information

## 🔧 Pull Request Process

1. **Update documentation** if you're adding features
2. **Test your changes** thoroughly
3. **Update README.md** if needed
4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: feature description"
   git commit -m "Fix: bug description"
   git commit -m "Update: what was updated"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create Pull Request** on GitHub

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Code has been tested and works as expected
- [ ] Documentation has been updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included

## 🧪 Testing

Before submitting a PR, please test:

- The app launches successfully
- Your feature works as intended
- No existing features are broken
- Error handling works properly

## 📝 Documentation

When adding new features:

- Update README.md with usage examples
- Add docstrings to new functions
- Include comments for complex logic
- Update CONTRIBUTING.md if process changes

## 🎨 Adding Example Images

When contributing example images:

- Use high-quality, clear photos
- Ensure you have rights to use the images
- Keep file sizes reasonable (< 2MB)
- Name files descriptively (e.g., `healthy_salad.jpg`)
- Place in `examples/` directory

## 🌟 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- The contributors page

## 📧 Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Check existing issues and discussions
- Review the documentation

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and constructive
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Food Health Analyzer! 🎉**
