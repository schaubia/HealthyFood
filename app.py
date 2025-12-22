"""
Food Health Analyzer
A web application that analyzes food images and provides nutritional information
"""

import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from PIL import Image
import requests
import json
import os

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

class FoodHealthAnalyzer:
    def __init__(self):
        """Initialize the food analyzer with ResNet50 model"""
        self.img_size = (224, 224)
        self.model = self.build_model()
        
    def build_model(self):
        """Build model using ResNet50 with ImageNet weights"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        return base_model
    
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        # Resize image
        img = img.resize(self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict_food(self, img, top_k=5):
        """Predict food item from image"""
        img_array = self.preprocess_image(img)
        predictions = self.model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions, top=top_k)[0]
        
        results = []
        for _, label, confidence in decoded:
            results.append({
                'name': label,
                'confidence': float(confidence)
            })
        
        return results
    
    def get_usda_nutrition(self, food_name):
        """Get nutrition information from USDA API"""
        try:
            params = {
                'api_key': USDA_API_KEY,
                'query': food_name,
                'dataType': ['Foundation', 'SR Legacy'],
                'pageSize': 1
            }
            
            response = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data.get('foods') or len(data['foods']) == 0:
                return None
            
            food = data['foods'][0]
            nutrients = {}
            
            # Extract key nutrients
            for nutrient in food.get('foodNutrients', []):
                name = nutrient.get('nutrientName', '')
                value = nutrient.get('value', 0)
                unit = nutrient.get('unitName', '')
                
                if name and value:
                    nutrients[name] = f"{value} {unit}"
            
            return {
                'name': food.get('description', food_name),
                'nutrients': nutrients
            }
            
        except Exception as e:
            print(f"Error fetching nutrition data: {e}")
            return None
    
    def analyze_health(self, nutrients):
        """Analyze health rating based on nutrients"""
        if not nutrients:
            return "Unknown", "⚪"
        
        # Simple health scoring logic
        health_score = 0
        max_score = 0
        
        # Positive nutrients
        positive_nutrients = {
            'Protein': 2,
            'Fiber, total dietary': 2,
            'Vitamin C, total ascorbic acid': 1,
            'Vitamin A, IU': 1,
            'Calcium, Ca': 1,
            'Iron, Fe': 1
        }
        
        # Negative nutrients
        negative_nutrients = {
            'Total lipid (fat)': -2,
            'Fatty acids, total saturated': -2,
            'Sugars, total including NLEA': -2,
            'Sodium, Na': -2,
            'Cholesterol': -1
        }
        
        for nutrient, weight in positive_nutrients.items():
            max_score += abs(weight)
            if nutrient in nutrients:
                try:
                    value = float(nutrients[nutrient].split()[0])
                    if value > 5:  # Significant amount
                        health_score += weight
                except:
                    pass
        
        for nutrient, weight in negative_nutrients.items():
            max_score += abs(weight)
            if nutrient in nutrients:
                try:
                    value = float(nutrients[nutrient].split()[0])
                    # Thresholds for "high" amounts (per 100g)
                    thresholds = {
                        'Total lipid (fat)': 20,
                        'Fatty acids, total saturated': 5,
                        'Sugars, total including NLEA': 10,
                        'Sodium, Na': 500,
                        'Cholesterol': 100
                    }
                    if value > thresholds.get(nutrient, 10):
                        health_score += weight
                except:
                    pass
        
        # Calculate rating
        if max_score == 0:
            return "Unknown", "⚪"
        
        score_ratio = health_score / max_score
        
        if score_ratio > 0.3:
            return "Healthy", "🟢"
        elif score_ratio > -0.3:
            return "Moderate", "🟡"
        else:
            return "Unhealthy", "🔴"

# Initialize analyzer
analyzer = FoodHealthAnalyzer()

def analyze_food_image(img):
    """Main function to analyze food image"""
    if img is None:
        return "Please upload an image", "", ""
    
    # Get predictions
    predictions = analyzer.predict_food(img)
    
    # Format predictions
    pred_text = "### 🍽️ Food Recognition Results:\n\n"
    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred['confidence'] * 100
        pred_text += f"{i}. **{pred['name'].replace('_', ' ').title()}** - {confidence_pct:.2f}%\n"
    
    # Get nutrition info for top prediction
    top_food = predictions[0]['name'].replace('_', ' ')
    nutrition_data = analyzer.get_usda_nutrition(top_food)
    
    if nutrition_data:
        nutrients = nutrition_data['nutrients']
        health_rating, emoji = analyzer.analyze_health(nutrients)
        
        nutrition_text = f"### 📊 Nutritional Information\n\n"
        nutrition_text += f"**Food:** {nutrition_data['name']}\n\n"
        nutrition_text += f"**Health Rating:** {emoji} {health_rating}\n\n"
        nutrition_text += "**Key Nutrients (per 100g):**\n\n"
        
        # Priority nutrients to display
        priority = [
            'Energy',
            'Protein',
            'Total lipid (fat)',
            'Carbohydrate, by difference',
            'Fiber, total dietary',
            'Sugars, total including NLEA',
            'Sodium, Na',
            'Cholesterol'
        ]
        
        displayed = set()
        for nutrient in priority:
            if nutrient in nutrients:
                nutrition_text += f"• **{nutrient}:** {nutrients[nutrient]}\n"
                displayed.add(nutrient)
        
        # Add remaining nutrients
        for nutrient, value in nutrients.items():
            if nutrient not in displayed and len(displayed) < 15:
                nutrition_text += f"• **{nutrient}:** {value}\n"
                displayed.add(nutrient)
        
        health_advice = f"### 💡 Health Insights\n\n"
        if health_rating == "Healthy":
            health_advice += "✅ This food appears to be a healthy choice! It contains beneficial nutrients."
        elif health_rating == "Moderate":
            health_advice += "⚠️ This food is okay in moderation. Be mindful of portion sizes."
        else:
            health_advice += "⚠️ This food may be high in fats, sugars, or sodium. Consume in moderation."
        
        return pred_text, nutrition_text, health_advice
    else:
        return pred_text, "### ℹ️ Nutritional Information\n\nNutrition data not available for this food item.", ""

# Create Gradio interface
with gr.Blocks(title="Food Health Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🍎 Food Health Analyzer
    
    Upload a photo of food to identify it and get nutritional information!
    
    This app uses AI to recognize food items and provides:
    - Food identification with confidence scores
    - Nutritional information from USDA database
    - Health rating based on nutritional content
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Food Image")
            analyze_btn = gr.Button("🔍 Analyze Food", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📝 Tips for best results:
            - Use clear, well-lit photos
            - Center the food in the frame
            - Avoid overly complex dishes
            - One food item works best
            """)
        
        with gr.Column():
            predictions_output = gr.Markdown(label="Recognition Results")
            nutrition_output = gr.Markdown(label="Nutrition Information")
            health_output = gr.Markdown(label="Health Insights")
    
    # Example images section
    gr.Markdown("### 📸 Try these examples:")
    gr.Examples(
        examples=[],  # Add example image paths here
        inputs=input_image,
        label="Example Foods"
    )
    
    # Connect button to function
    analyze_btn.click(
        fn=analyze_food_image,
        inputs=input_image,
        outputs=[predictions_output, nutrition_output, health_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
