"""
Food Health Analyzer
A web application that analyzes food images and provides nutritional information
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Food Health Analyzer",
    page_icon="🍎",
    layout="wide"
)

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

class FoodHealthAnalyzer:
    def __init__(self):
        """Initialize the food analyzer with ResNet50 model"""
        self.img_size = (224, 224)
        self.model = self.build_model()
        
    @st.cache_resource
    def build_model(_self):
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

# Initialize analyzer (cached to avoid reloading model)
@st.cache_resource
def get_analyzer():
    return FoodHealthAnalyzer()

analyzer = get_analyzer()

# Main app
def main():
    st.title("🍎 Food Health Analyzer")
    st.markdown("""
    Upload a photo of food to identify it and get nutritional information!
    
    **This app uses AI to recognize food items and provides:**
    - 🍽️ Food identification with confidence scores
    - 📊 Nutritional information from USDA database
    - 💚 Health rating based on nutritional content
    """)
    
    # Sidebar with tips
    with st.sidebar:
        st.header("📝 Tips for Best Results")
        st.markdown("""
        - Use clear, well-lit photos
        - Center the food in the frame
        - Avoid overly complex dishes
        - One food item works best
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses **ResNet50** trained on ImageNet to recognize food items 
        and provides nutritional analysis using data from the **USDA FoodData Central** database.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a food image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of food"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analyzing...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Predict food
                status_text.text("🤖 Identifying food with AI...")
                progress_bar.progress(33)
                predictions = analyzer.predict_food(img)
                
                # Step 2: Get nutrition
                status_text.text("📊 Fetching nutritional data...")
                progress_bar.progress(66)
                top_food = predictions[0]['name'].replace('_', ' ')
                nutrition_data = analyzer.get_usda_nutrition(top_food)
                
                # Step 3: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Clear progress indicators after a moment
                import time
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                st.success("Analysis complete! 🎉")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
        
        # Display results
        st.markdown("---")
        
        # Food Recognition Results
        st.subheader("🍽️ Food Recognition Results")
        for i, pred in enumerate(predictions, 1):
            confidence_pct = pred['confidence'] * 100
            st.write(f"**{i}.** {pred['name'].replace('_', ' ').title()} - {confidence_pct:.2f}%")
        
        st.markdown("---")
        
        # Nutritional Information
        if nutrition_data:
            nutrients = nutrition_data['nutrients']
            health_rating, emoji = analyzer.analyze_health(nutrients)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Nutritional Information")
                st.write(f"**Food:** {nutrition_data['name']}")
                st.write(f"**Health Rating:** {emoji} {health_rating}")
            
            with col2:
                # Health insight card
                if health_rating == "Healthy":
                    st.success("✅ Healthy Choice!")
                elif health_rating == "Moderate":
                    st.warning("⚠️ Consume in Moderation")
                else:
                    st.error("⚠️ High in Fats/Sugars/Sodium")
            
            st.markdown("**Key Nutrients (per 100g):**")
            
            # Priority nutrients
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
            
            # Display nutrients in a nice table
            nutrient_data = []
            displayed = set()
            
            for nutrient in priority:
                if nutrient in nutrients:
                    nutrient_data.append([nutrient, nutrients[nutrient]])
                    displayed.add(nutrient)
            
            # Add remaining nutrients (up to 15 total)
            for nutrient, value in nutrients.items():
                if nutrient not in displayed and len(displayed) < 15:
                    nutrient_data.append([nutrient, value])
                    displayed.add(nutrient)
            
            # Create two columns for nutrients
            if nutrient_data:
                col1, col2 = st.columns(2)
                mid = len(nutrient_data) // 2
                
                with col1:
                    for nutrient, value in nutrient_data[:mid]:
                        st.write(f"• **{nutrient}:** {value}")
                
                with col2:
                    for nutrient, value in nutrient_data[mid:]:
                        st.write(f"• **{nutrient}:** {value}")
            
            # Health Insights
            st.markdown("---")
            st.subheader("💡 Health Insights")
            
            if health_rating == "Healthy":
                st.markdown("""
                ✅ **This food appears to be a healthy choice!** It contains beneficial nutrients 
                that contribute to a balanced diet.
                """)
            elif health_rating == "Moderate":
                st.markdown("""
                ⚠️ **This food is okay in moderation.** Be mindful of portion sizes and 
                try to balance it with other nutritious foods throughout the day.
                """)
            else:
                st.markdown("""
                ⚠️ **This food may be high in fats, sugars, or sodium.** Consider consuming 
                it occasionally and in small portions as part of a balanced diet.
                """)
        else:
            st.info("ℹ️ Nutritional information not available for this food item. The AI identified it as '" + 
                   top_food + "', but we couldn't find matching nutritional data in the USDA database.")
    
    else:
        # Show example when no image is uploaded
        st.info("👆 Upload an image to get started!")
        
        # Optional: Show example images if you have them
        st.markdown("### 📸 How it works:")
        st.markdown("""
        1. Upload a clear photo of food
        2. AI analyzes the image using ResNet50
        3. Get nutritional information from USDA database
        4. Receive a health rating and insights
        """)

if __name__ == "__main__":
    main()
