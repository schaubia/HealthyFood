"""
Food Health Analyzer - Hybrid Version
Uses both Food-101 (for dishes) and ResNet50 (for ingredients/fruits)
"""

import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
import os
import torch
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Food Health Analyzer",
    page_icon="🍎",
    layout="wide"
)

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

class HybridFoodAnalyzer:
    def __init__(self):
        """Initialize with both models for better coverage"""
        self.vit_extractor, self.vit_model = self.build_vit_model()
        self.resnet_model = self.build_resnet_model()
        self.img_size = (224, 224)
        
    @st.cache_resource
    def build_vit_model(_self):
        """Build ViT model for prepared dishes"""
        model_name = "nateraw/food"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        return feature_extractor, model
    
    @st.cache_resource
    def build_resnet_model(_self):
        """Build ResNet50 for ingredients and fruits"""
        return ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    
    def predict_with_vit(self, img):
        """Predict using ViT Food-101 model"""
        inputs = self.vit_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            logits = outputs.logits
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.vit_model.config.id2label[idx.item()]
            results.append({
                'name': label,
                'confidence': prob.item(),
                'source': 'Food-101'
            })
        return results
    
    def predict_with_resnet(self, img):
        """Predict using ResNet50 ImageNet model"""
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize(self.img_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        predictions = self.resnet_model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions, top=5)[0]
        
        results = []
        for _, label, confidence in decoded:
            results.append({
                'name': label,
                'confidence': float(confidence),
                'source': 'ImageNet'
            })
        return results
    
    def predict_food(self, img):
        """Smart prediction using both models"""
        # Try Food-101 first
        vit_results = self.predict_with_vit(img)
        
        # Try ResNet50 for comparison
        resnet_results = self.predict_with_resnet(img)
        
        # Filter ResNet results for food-related items
        food_keywords = ['fruit', 'vegetable', 'meat', 'fish', 'berry', 'apple', 'orange', 
                        'banana', 'pear', 'grape', 'lemon', 'mushroom', 'corn', 'pepper',
                        'tomato', 'potato', 'carrot', 'broccoli', 'strawberry', 'pizza',
                        'burger', 'sandwich', 'salad', 'bread', 'cheese', 'chocolate']
        
        food_resnet_results = [
            r for r in resnet_results 
            if any(keyword in r['name'].lower() for keyword in food_keywords)
        ]
        
        # Combine results intelligently
        # If top VIT result has low confidence (<0.3), prefer ResNet for simple foods
        if vit_results[0]['confidence'] < 0.3 and food_resnet_results:
            # Use ResNet if it found food items
            if food_resnet_results[0]['confidence'] > 0.5:
                return food_resnet_results[:5]
        
        # Otherwise return VIT results
        return vit_results
    
    def get_recipe_ingredients(self, food_name):
        """Scrape the web to find main ingredients for a dish"""
        try:
            # Use DuckDuckGo search (no API key needed)
            from duckduckgo_search import DDGS
            
            # Search for recipe
            search_query = f"{food_name} recipe main ingredients"
            
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=3))
            
            if not results:
                return None
            
            # Extract ingredients from search results
            ingredients_text = ""
            for result in results:
                snippet = result.get('body', '')
                title = result.get('title', '')
                ingredients_text += f"{title} {snippet} "
            
            # Common food ingredients to look for
            common_ingredients = [
                'egg', 'eggs', 'milk', 'cheese', 'butter', 'oil', 'olive oil',
                'flour', 'wheat', 'rice', 'pasta', 'bread', 'sugar', 'salt',
                'pepper', 'onion', 'garlic', 'tomato', 'chicken', 'beef', 'pork',
                'fish', 'shrimp', 'potato', 'carrot', 'cream', 'yogurt',
                'lemon', 'lime', 'herbs', 'spices', 'basil', 'oregano',
                'parmesan', 'mozzarella', 'cheddar', 'bacon', 'ham',
                'mushroom', 'spinach', 'broccoli', 'lettuce', 'cucumber'
            ]
            
            # Find mentioned ingredients
            found_ingredients = []
            ingredients_lower = ingredients_text.lower()
            
            for ingredient in common_ingredients:
                if ingredient in ingredients_lower and ingredient not in found_ingredients:
                    found_ingredients.append(ingredient)
            
            # Limit to top 10 most commonly found
            found_ingredients = found_ingredients[:10]
            
            if found_ingredients:
                return {
                    'ingredients': found_ingredients,
                    'source': results[0].get('href', 'web search')
                }
            
            return None
            
        except Exception as e:
            print(f"Error scraping ingredients: {e}")
            return None
    
    def analyze_ingredients_health(self, ingredients):
        """Grade ingredients based on health"""
        if not ingredients:
            return [], []
        
        # Categorize ingredients
        healthy = []
        unhealthy = []
        
        healthy_items = [
            'egg', 'eggs', 'chicken', 'fish', 'shrimp', 'mushroom', 
            'spinach', 'broccoli', 'lettuce', 'cucumber', 'tomato',
            'carrot', 'onion', 'garlic', 'herbs', 'basil', 'oregano',
            'yogurt', 'olive oil', 'lemon', 'lime'
        ]
        
        unhealthy_items = [
            'sugar', 'butter', 'cream', 'bacon', 'ham', 'cheddar',
            'mozzarella', 'parmesan', 'oil', 'fried', 'deep fried'
        ]
        
        neutral_items = [
            'flour', 'wheat', 'rice', 'pasta', 'bread', 'potato',
            'milk', 'cheese', 'salt', 'pepper', 'spices'
        ]
        
        for ingredient in ingredients:
            ing_lower = ingredient.lower()
            
            if any(h in ing_lower for h in healthy_items):
                healthy.append(ingredient)
            elif any(u in ing_lower for u in unhealthy_items):
                unhealthy.append(ingredient)
            # Neutral items are not categorized but still shown
        
        return healthy, unhealthy
    
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
        
        health_score = 0
        max_score = 0
        
        positive_nutrients = {
            'Protein': 2,
            'Fiber, total dietary': 2,
            'Vitamin C, total ascorbic acid': 1,
            'Vitamin A, IU': 1,
            'Calcium, Ca': 1,
            'Iron, Fe': 1
        }
        
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
                    if value > 5:
                        health_score += weight
                except:
                    pass
        
        for nutrient, weight in negative_nutrients.items():
            max_score += abs(weight)
            if nutrient in nutrients:
                try:
                    value = float(nutrients[nutrient].split()[0])
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
        
        if max_score == 0:
            return "Unknown", "⚪"
        
        score_ratio = health_score / max_score
        
        if score_ratio > 0.3:
            return "Healthy", "🟢"
        elif score_ratio > -0.3:
            return "Moderate", "🟡"
        else:
            return "Unhealthy", "🔴"

@st.cache_resource
def get_analyzer():
    return HybridFoodAnalyzer()

analyzer = get_analyzer()

def main():
    st.title("🍎 Food Health Analyzer")
    st.markdown("""
    Upload a photo of food to identify it and get nutritional information!
    
    **Features:**
    - 🍽️ Smart AI recognition (dishes + ingredients)
    - 📊 Nutritional information from USDA database
    - 💚 Health rating based on nutritional content
    """)
    
    with st.sidebar:
        st.header("📝 Tips for Best Results")
        st.markdown("""
        - Use clear, well-lit photos
        - Center the food in the frame
        - Works best with:
          - Individual fruits/vegetables
          - Common prepared dishes
          - Single food items
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses **hybrid AI**:
        - **Food-101 ViT** for prepared dishes
        - **ResNet50 ImageNet** for fruits, vegetables, and ingredients
        
        Nutritional data from **USDA FoodData Central**.
        """)
    
    uploaded_file = st.file_uploader(
        "Choose a food image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of food"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analyzing...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🤖 Identifying food with AI...")
                progress_bar.progress(33)
                predictions = analyzer.predict_food(img)
                
                status_text.text("📊 Fetching nutritional data...")
                progress_bar.progress(50)
                top_food = predictions[0]['name'].replace('_', ' ')
                nutrition_data = analyzer.get_usda_nutrition(top_food)
                
                status_text.text("🔍 Searching for ingredients...")
                progress_bar.progress(75)
                ingredients_data = analyzer.get_recipe_ingredients(top_food)
                
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                import time
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                st.success("Analysis complete! 🎉")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
        
        st.markdown("---")
        
        st.subheader("🍽️ Food Recognition Results")
        for i, pred in enumerate(predictions, 1):
            confidence_pct = pred['confidence'] * 100
            source = pred.get('source', 'AI')
            st.write(f"**{i}.** {pred['name'].replace('_', ' ').title()} - {confidence_pct:.2f}% _{source}_")
        
        st.markdown("---")
        
        # Ingredients Section (if found)
        if ingredients_data:
            st.subheader("🥘 Main Ingredients")
            
            ingredients_list = ingredients_data['ingredients']
            healthy_ings, unhealthy_ings = analyzer.analyze_ingredients_health(ingredients_list)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**All Ingredients:**")
                for ing in ingredients_list:
                    st.write(f"• {ing.title()}")
            
            with col2:
                if healthy_ings:
                    st.markdown("**✅ Healthy:**")
                    for ing in healthy_ings:
                        st.write(f"• {ing.title()}")
                else:
                    st.markdown("**✅ Healthy:**")
                    st.write("_None identified_")
            
            with col3:
                if unhealthy_ings:
                    st.markdown("**⚠️ Watch Out:**")
                    for ing in unhealthy_ings:
                        st.write(f"• {ing.title()}")
                else:
                    st.markdown("**⚠️ Watch Out:**")
                    st.write("_None identified_")
            
            # Ingredients health summary
            if len(healthy_ings) > len(unhealthy_ings):
                st.success("💚 This dish contains mostly healthy ingredients!")
            elif len(unhealthy_ings) > len(healthy_ings):
                st.warning("⚠️ This dish contains ingredients to consume in moderation.")
            else:
                st.info("ℹ️ This dish has a balanced mix of ingredients.")
            
            st.markdown("---")
        
        # Nutritional Information
        if nutrition_data:
            nutrients = nutrition_data['nutrients']
            health_rating, emoji = analyzer.analyze_health(nutrients)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Nutritional Information")
                st.write(f"**Food:** {nutrition_data['name']}")
                
                # Extract and display calories prominently
                calories = "Not available"
                for key in nutrients.keys():
                    if 'energy' in key.lower() or 'calor' in key.lower():
                        calories = nutrients[key]
                        break
                
                st.metric(label="🔥 Calories (per 100g)", value=calories)
                st.write(f"**Health Rating:** {emoji} {health_rating}")
            
            with col2:
                if health_rating == "Healthy":
                    st.success("✅ Healthy Choice!")
                elif health_rating == "Moderate":
                    st.warning("⚠️ Consume in Moderation")
                else:
                    st.error("⚠️ High in Fats/Sugars/Sodium")
            
            st.markdown("**Key Nutrients (per 100g):**")
            
            priority = [
                'Energy', 'Protein', 'Total lipid (fat)',
                'Carbohydrate, by difference', 'Fiber, total dietary',
                'Sugars, total including NLEA', 'Sodium, Na', 'Cholesterol'
            ]
            
            nutrient_data = []
            displayed = set()
            
            for nutrient in priority:
                if nutrient in nutrients:
                    nutrient_data.append([nutrient, nutrients[nutrient]])
                    displayed.add(nutrient)
            
            for nutrient, value in nutrients.items():
                if nutrient not in displayed and len(displayed) < 15:
                    nutrient_data.append([nutrient, value])
                    displayed.add(nutrient)
            
            if nutrient_data:
                col1, col2 = st.columns(2)
                mid = len(nutrient_data) // 2
                
                with col1:
                    for nutrient, value in nutrient_data[:mid]:
                        st.write(f"• **{nutrient}:** {value}")
                
                with col2:
                    for nutrient, value in nutrient_data[mid:]:
                        st.write(f"• **{nutrient}:** {value}")
            
            st.markdown("---")
            st.subheader("💡 Health Insights")
            
            if health_rating == "Healthy":
                st.markdown("✅ **This food appears to be a healthy choice!** It contains beneficial nutrients that contribute to a balanced diet.")
            elif health_rating == "Moderate":
                st.markdown("⚠️ **This food is okay in moderation.** Be mindful of portion sizes and try to balance it with other nutritious foods.")
            else:
                st.markdown("⚠️ **This food may be high in fats, sugars, or sodium.** Consider consuming it occasionally and in small portions.")
        else:
            st.info(f"ℹ️ Nutritional information not available for '{top_food}' in the USDA database. Try searching for a similar food name.")
    
    else:
        st.info("👆 Upload an image to get started!")
        
        st.markdown("### 📸 Best Results For:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Individual Ingredients:**
            - 🍎 Fruits (apple, banana, orange, pear)
            - 🥕 Vegetables (carrot, broccoli, tomato)
            - 🍄 Mushrooms
            - 🌽 Corn, peppers
            """)
        with col2:
            st.markdown("""
            **Prepared Dishes:**
            - 🍕 Pizza
            - 🍔 Hamburger
            - 🍣 Sushi
            - 🥗 Salad
            - 🍰 Desserts
            """)

if __name__ == "__main__":
    main()
