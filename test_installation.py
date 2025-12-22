"""
Test script for Food Health Analyzer
Run this to verify your installation is working correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import gradio as gr
        print(f"  ✓ Gradio {gr.__version__}")
    except ImportError as e:
        print(f"  ❌ Gradio import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"  ✓ Pillow")
    except ImportError as e:
        print(f"  ❌ Pillow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
    
    try:
        import requests
        print(f"  ✓ Requests {requests.__version__}")
    except ImportError as e:
        print(f"  ❌ Requests import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        from tensorflow.keras.applications import ResNet50
        model = ResNet50(weights='imagenet', include_top=True)
        print("  ✓ ResNet50 model loaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def test_api_connection():
    """Test USDA API connection"""
    print("\n🌐 Testing USDA API connection...")
    
    try:
        import requests
        import os
        
        api_key = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            'api_key': api_key,
            'query': 'apple',
            'pageSize': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            print(f"  ✓ USDA API connection successful (using {api_key})")
            return True
        else:
            print(f"  ⚠️ API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ API connection failed: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    print("\n🖼️ Testing image processing...")
    
    try:
        from PIL import Image
        import numpy as np
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        # Create a test image
        test_img = Image.new('RGB', (224, 224), color='red')
        img_array = image.img_to_array(test_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        print(f"  ✓ Image processing works (shape: {img_array.shape})")
        return True
        
    except Exception as e:
        print(f"  ❌ Image processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🍎 Food Health Analyzer - Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Model Loading Test", test_model_loading()))
    results.append(("API Connection Test", test_api_connection()))
    results.append(("Image Processing Test", test_image_processing()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is working correctly.")
        print("\nYou can now run the application with: python app.py")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure you've activated your virtual environment")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check your internet connection (for model downloads)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
