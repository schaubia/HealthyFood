"""
Food Health Analyzer - Enhanced with Learning
Uses both Food-101 (for dishes) and ResNet50 (for ingredients/fruits)
WITH USER FEEDBACK AND LEARNING MECHANISM
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
import json
from datetime import datetime
import pickle

# Page configuration
st.set_page_config(
    page_title="Food Health Analyzer",
    page_icon="🍎",
    layout="wide"
)

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Learning configuration
FEEDBACK_FILE = "food_feedback.json"
USER_CORRECTIONS_FILE = "user_corrections.pkl"

class HybridFoodAnalyzer:
    def __init__(self):
        """Initialize with both models for better coverage"""
        self.vit_extractor, self.vit_model = self.build_vit_model()
        self.resnet_model = self.build_resnet_model()
        self.img_size = (224, 224)
        self.load_user_corrections()
        
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
    
    def load_user_corrections(self):
        """Load user corrections from previous sessions"""
        if os.path.exists(USER_CORRECTIONS_FILE):
            with open(USER_CORRECTIONS_FILE, 'rb') as f:
                self.user_corrections = pickle.load(f)
        else:
            self.user_corrections = []
        
        # Also load feedback log
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                self.feedback_log = json.load(f)
        else:
            self.feedback_log = []
    
    def save_user_corrections(self):
        """Save user corrections for future learning"""
        with open(USER_CORRECTIONS_FILE, 'wb') as f:
            pickle.dump(self.user_corrections, f)
        
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(self.feedback_log, f, indent=2)
    
    def extract_image_features(self, img):
        """Extract features from image for similarity matching"""
        # Resize and convert to array
        img_resized = img.resize(self.img_size)
        img_array = image.img_to_array(img_resized)
        
        # Extract basic features
        avg_colors = img_array.mean(axis=(0, 1))
        color_variance = img_array.std(axis=(0, 1))
        brightness = img_array.mean()
        
        # Combine features
        features = np.concatenate([avg_colors, color_variance, [brightness]])
        return features
    
    def check_user_corrections(self, img, features):
        """Check if similar images were corrected by user"""
        if not self.user_corrections:
            return None
        
        # Find similar corrections
        similarities = []
        for correction in self.user_corrections:
            if 'features' in correction:
                saved_features = np.array(correction['features'])
                distance = np.linalg.norm(features - saved_features)
                similarity = 1 / (1 + distance)
                
                if similarity > 0.85:  # High similarity threshold
                    similarities.append({
                        'food': correction['correct_food'],
                        'similarity': similarity,
                        'count': correction.get('count', 1)
                    })
        
        if similarities:
            # Sort by similarity and count
            similarities.sort(key=lambda x: (x['similarity'], x['count']), reverse=True)
            return similarities[0]
        
        return None
    
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
        """Smart prediction using both models and user corrections"""
        # Extract features for similarity matching
        features = self.extract_image_features(img)
        
        # Check user corrections first
        user_match = self.check_user_corrections(img, features)
        
        if user_match:
            # User has corrected similar images before
            st.info(f"🧠 Found similar correction from learning: {user_match['food']} (similarity: {user_match['similarity']:.1%})")
            return [{
                'name': user_match['food'],
                'confidence': 0.95,
                'source': 'User Learning',
                'features': features.tolist()
            }]
        
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
        
        # Add features to results
        for result in vit_results:
            result['features'] = features.tolist()
        
        # Combine results intelligently
        if vit_results[0]['confidence'] < 0.3 and food_resnet_results:
            if food_resnet_results[0]['confidence'] > 0.5:
                for result in food_resnet_results:
                    result['features'] = features.tolist()
                return food_resnet_results[:5]
        
        return vit_results
    
    def add_user_correction(self, predicted_food, correct_food, features, confidence):
        """Add user correction to learning database"""
        # Check if this exact food already exists in corrections
        existing = None
        for i, correction in enumerate(self.user_corrections):
            if correction['correct_food'].lower() == correct_food.lower():
                # Check if features are similar
                saved_features = np.array(correction['features'])
                distance = np.linalg.norm(np.array(features) - saved_features)
                if distance < 50:  # Similar enough
                    existing = i
                    break
        
        if existing is not None:
            # Update existing correction
            self.user_corrections[existing]['count'] = self.user_corrections[existing].get('count', 1) + 1
            self.user_corrections[existing]['last_updated'] = datetime.now().isoformat()
        else:
            # Add new correction
            self.user_corrections.append({
                'predicted_food': predicted_food,
                'correct_food': correct_food,
                'features': features,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'count': 1
            })
        
        # Add to feedback log
        self.feedback_log.append({
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted_food,
            'correct': correct_food,
            'was_correct': predicted_food.lower() == correct_food.lower()
        })
        
        self.save_user_corrections()
    
    def get_learning_stats(self):
        """Get statistics about learning progress"""
        total_corrections = len(self.feedback_log)
        correct_predictions = sum(1 for log in self.feedback_log if log['was_correct'])
        
        if total_corrections > 0:
            accuracy = (correct_predictions / total_corrections) * 100
        else:
            accuracy = 0
        
        unique_foods = len(self.user_corrections)
        
        return {
            'total_feedback': total_corrections,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'unique_foods_learned': unique_foods
        }
    
    def get_recipe_ingredients(self, food_name):
        """Get ingredients using Wikipedia API and fallback database"""
        
        # First, try predefined database for common dishes
        ingredients = self.get_ingredients_from_database(food_name)
        if ingredients:
            st.info(f"✅ Found ingredients from database")
            return {
                'ingredients': ingredients,
                'source': 'Built-in database'
            }
        
        # If not in database, try Wikipedia
        try:
            import wikipediaapi
            
            # Proper user agent as required by Wikipedia
            wiki = wikipediaapi.Wikipedia(
                user_agent='FoodHealthAnalyzer/1.0 (Educational App)',
                language='en'
            )
            
            # Search for the food page
            page = wiki.page(food_name)
            
            if not page.exists():
                # Try with "cuisine" or "food" suffix
                page = wiki.page(f"{food_name} (food)")
            
            if not page.exists():
                st.warning(f"No Wikipedia page found for '{food_name}'")
                return None
            
            # Get page text
            text = page.text.lower()
            st.write(f"📖 Found Wikipedia article")
            
            # Common food ingredients to look for
            common_ingredients = [
                'egg', 'eggs', 'milk', 'cheese', 'butter', 'oil', 'olive oil',
                'flour', 'wheat', 'rice', 'pasta', 'bread', 'sugar', 'salt',
                'pepper', 'onion', 'garlic', 'tomato', 'chicken', 'beef', 'pork',
                'fish', 'shrimp', 'potato', 'carrot', 'cream', 'yogurt',
                'lemon', 'lime', 'herbs', 'spices', 'basil', 'oregano',
                'parmesan', 'mozzarella', 'cheddar', 'bacon', 'ham',
                'mushroom', 'spinach', 'broccoli', 'lettuce', 'cucumber',
                'ricotta', 'meat', 'ground beef', 'sausage', 'marinara',
                'sauce', 'parsley', 'thyme', 'rosemary', 'vanilla', 'chocolate'
            ]
            
            # Find mentioned ingredients
            found_ingredients = []
            
            for ingredient in common_ingredients:
                if ingredient in text:
                    # Avoid duplicates
                    if ingredient == 'eggs' and 'egg' in found_ingredients:
                        continue
                    if ingredient == 'egg' and 'eggs' in found_ingredients:
                        continue
                    
                    if ingredient not in found_ingredients:
                        found_ingredients.append(ingredient)
            
            if found_ingredients:
                st.success(f"✅ Found {len(found_ingredients)} ingredients from Wikipedia")
                return {
                    'ingredients': found_ingredients[:15],
                    'source': 'Wikipedia'
                }
            else:
                st.warning("No ingredients detected in Wikipedia article")
                return None
            
        except Exception as e:
            st.error(f"Error accessing Wikipedia: {str(e)}")
            return None
    
    def get_ingredients_from_database(self, food_name):
        """Built-in ingredient database for common dishes"""
        
        # Normalize food name
        food_lower = food_name.lower().strip()
        
        # Common dishes with known ingredients
        ingredient_db = {
            # Breakfast
            'omelette': ['eggs', 'cheese', 'butter', 'milk', 'salt', 'pepper'],
            'omelet': ['eggs', 'cheese', 'butter', 'milk', 'salt', 'pepper'],
            'scrambled eggs': ['eggs', 'butter', 'milk', 'salt', 'pepper'],
            'fried egg': ['eggs', 'oil', 'salt', 'pepper'],
            'pancake': ['flour', 'eggs', 'milk', 'sugar', 'butter', 'baking powder'],
            'waffle': ['flour', 'eggs', 'milk', 'sugar', 'butter', 'baking powder'],
            'french toast': ['bread', 'eggs', 'milk', 'sugar', 'cinnamon', 'butter'],
            
            # Italian
            'pizza': ['flour', 'yeast', 'tomato', 'mozzarella', 'olive oil', 'basil'],
            'pasta': ['wheat', 'flour', 'eggs', 'salt', 'water'],
            'spaghetti': ['pasta', 'tomato', 'garlic', 'olive oil', 'basil'],
            'lasagna': ['pasta', 'beef', 'tomato', 'ricotta', 'mozzarella', 'parmesan'],
            'ravioli': ['pasta', 'ricotta', 'cheese', 'eggs', 'spinach', 'flour'],
            'risotto': ['rice', 'butter', 'parmesan', 'onion', 'chicken broth', 'white wine'],
            'carbonara': ['pasta', 'eggs', 'bacon', 'parmesan', 'pepper', 'salt'],
            
            # American
            'hamburger': ['beef', 'bread', 'lettuce', 'tomato', 'onion', 'cheese', 'pickles'],
            'cheeseburger': ['beef', 'bread', 'cheese', 'lettuce', 'tomato', 'onion'],
            'hot dog': ['sausage', 'bread', 'mustard', 'ketchup', 'onion'],
            'sandwich': ['bread', 'meat', 'cheese', 'lettuce', 'tomato', 'mayo'],
            'french fries': ['potato', 'oil', 'salt'],
            'mashed potato': ['potato', 'butter', 'milk', 'salt', 'pepper'],
            
            # Asian
            'sushi': ['rice', 'fish', 'seaweed', 'vinegar', 'soy sauce', 'wasabi'],
            'ramen': ['noodles', 'broth', 'egg', 'pork', 'onion', 'soy sauce'],
            'fried rice': ['rice', 'eggs', 'soy sauce', 'vegetables', 'oil', 'garlic'],
            'pad thai': ['rice noodles', 'shrimp', 'eggs', 'peanuts', 'lime', 'fish sauce'],
            'spring roll': ['rice paper', 'shrimp', 'vegetables', 'noodles', 'herbs'],
            
            # Mexican
            'tacos': ['tortilla', 'beef', 'lettuce', 'cheese', 'tomato', 'salsa'],
            'burrito': ['tortilla', 'rice', 'beans', 'meat', 'cheese', 'salsa'],
            'quesadilla': ['tortilla', 'cheese', 'chicken', 'peppers', 'onion'],
            'nachos': ['tortilla chips', 'cheese', 'beans', 'salsa', 'sour cream', 'jalapeño'],
            
            # Desserts
            'cake': ['flour', 'sugar', 'eggs', 'butter', 'milk', 'baking powder'],
            'chocolate cake': ['flour', 'sugar', 'eggs', 'butter', 'cocoa', 'milk'],
            'cheesecake': ['cream cheese', 'sugar', 'eggs', 'graham crackers', 'butter'],
            'ice cream': ['milk', 'cream', 'sugar', 'vanilla', 'eggs'],
            'cookie': ['flour', 'sugar', 'butter', 'eggs', 'chocolate chips'],
            'brownie': ['chocolate', 'butter', 'sugar', 'eggs', 'flour'],
            'donut': ['flour', 'sugar', 'eggs', 'milk', 'butter', 'yeast'],
            'creme brulee': ['cream', 'eggs', 'sugar', 'vanilla'],
            'tiramisu': ['mascarpone', 'eggs', 'coffee', 'sugar', 'ladyfingers', 'cocoa'],
            'apple pie': ['apples', 'flour', 'sugar', 'butter', 'cinnamon'],
            
            # Salads
            'caesar salad': ['lettuce', 'parmesan', 'croutons', 'caesar dressing', 'chicken'],
            'greek salad': ['lettuce', 'tomato', 'cucumber', 'feta', 'olives', 'olive oil'],
            'salad': ['lettuce', 'tomato', 'cucumber', 'onion', 'olive oil', 'vinegar'],
            
            # Soups
            'chicken soup': ['chicken', 'broth', 'carrot', 'celery', 'onion', 'noodles'],
            'tomato soup': ['tomato', 'cream', 'onion', 'garlic', 'basil', 'butter'],
            'minestrone': ['pasta', 'beans', 'tomato', 'carrot', 'celery', 'onion'],
            
            # Meat dishes
            'steak': ['beef', 'salt', 'pepper', 'butter', 'garlic'],
            'chicken breast': ['chicken', 'salt', 'pepper', 'oil', 'herbs'],
            'pork chop': ['pork', 'salt', 'pepper', 'oil', 'garlic'],
            'fish fillet': ['fish', 'salt', 'pepper', 'lemon', 'butter'],
            
            # Breakfast
            'croissant': ['flour', 'butter', 'yeast', 'sugar', 'milk', 'eggs'],
            'bagel': ['flour', 'yeast', 'salt', 'sugar', 'water'],
            'muffin': ['flour', 'sugar', 'eggs', 'milk', 'butter', 'baking powder']
        }
        
        # Try exact match
        if food_lower in ingredient_db:
            return ingredient_db[food_lower]
        
        # Try partial match
        for dish_name, ingredients in ingredient_db.items():
            if dish_name in food_lower or food_lower in dish_name:
                return ingredients
        
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
    st.title("🍎 Food Health Analyzer with Learning")
    st.markdown("""
    Upload a photo of food to identify it and get nutritional information!
    
    **Features:**
    - 🍽️ Smart AI recognition (dishes + ingredients)
    - 📊 Nutritional information from USDA database
    - 💚 Health rating based on nutritional content
    - 🧠 **NEW: Learns from your corrections!**
    """)
    
    with st.sidebar:
        st.header("🧠 Learning Statistics")
        
        stats = analyzer.get_learning_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Feedback", stats['total_feedback'])
            st.metric("Unique Foods", stats['unique_foods_learned'])
        
        with col2:
            st.metric("Correct", stats['correct_predictions'])
            if stats['total_feedback'] > 0:
                st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
        
        if stats['total_feedback'] > 0:
            st.progress(stats['accuracy'] / 100, text=f"Learning Progress")
        
        st.markdown("---")
        
        if analyzer.user_corrections:
            st.subheader("📚 Learned Foods")
            for correction in analyzer.user_corrections[-5:]:
                count = correction.get('count', 1)
                st.write(f"✅ {correction['correct_food']} (×{count})")
        
        st.markdown("---")
        
        if st.button("🔄 Reset Learning Data", type="secondary"):
            analyzer.user_corrections = []
            analyzer.feedback_log = []
            analyzer.save_user_corrections()
            st.success("Learning data reset!")
            st.rerun()
        
        st.markdown("---")
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
        - **User Learning** from your corrections
        
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
        
        # Prediction Results with Feedback
        st.subheader("🍽️ Food Recognition Results")
        
        # Store prediction in session state
        if 'current_prediction' not in st.session_state:
            st.session_state.current_prediction = None
        
        st.session_state.current_prediction = {
            'food': top_food,
            'predictions': predictions,
            'features': predictions[0].get('features', [])
        }
        
        # Show predictions
        for i, pred in enumerate(predictions, 1):
            confidence_pct = pred['confidence'] * 100
            source = pred.get('source', 'AI')
            
            if i == 1:
                st.markdown(f"### **Top Prediction:** {pred['name'].replace('_', ' ').title()}")
                st.progress(pred['confidence'], text=f"Confidence: {confidence_pct:.1f}% _{source}_")
            else:
                st.write(f"**{i}.** {pred['name'].replace('_', ' ').title()} - {confidence_pct:.2f}% _{source}_")
        
        # Feedback Section
        st.markdown("---")
        st.subheader("📝 Was this prediction correct?")
        
        feedback_col1, feedback_col2 = st.columns(2)
        
        with feedback_col1:
            if st.button("✅ Yes, Correct!", type="primary", use_container_width=True):
                analyzer.add_user_correction(
                    top_food,
                    top_food,
                    st.session_state.current_prediction['features'],
                    predictions[0]['confidence']
                )
                st.success("✅ Thanks! The model will remember this.")
                st.balloons()
                time.sleep(1)
                st.rerun()
        
        with feedback_col2:
            if st.button("❌ No, Wrong", type="secondary", use_container_width=True):
                st.session_state.show_correction_form = True
        
        # Correction Form
        if st.session_state.get('show_correction_form', False):
            st.markdown("---")
            st.subheader("🔧 Help the Model Learn")
            
            correct_food_name = st.text_input(
                "What is the correct food name?",
                placeholder="e.g., Caesar Salad, Grilled Chicken, Apple Pie",
                help="Enter the actual name of the food in the image"
            )
            
            if st.button("💾 Submit Correction", type="primary", disabled=not correct_food_name):
                if correct_food_name:
                    analyzer.add_user_correction(
                        top_food,
                        correct_food_name,
                        st.session_state.current_prediction['features'],
                        predictions[0]['confidence']
                    )
                    st.success(f"✅ Thank you! The model learned that this is **{correct_food_name}**")
                    st.info("🧠 Next time you upload a similar image, the model will recognize it!")
                    st.session_state.show_correction_form = False
                    time.sleep(2)
                    st.rerun()
        
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
        
        st.markdown("---")
        st.markdown("### 🧠 How Learning Works")
        st.markdown("""
        1. **Upload** a food image and get AI predictions
        2. **Confirm** if correct, or provide the **correct name**
        3. **The model learns** and will recognize similar foods better next time
        4. **Track progress** in the sidebar statistics
        
        The more you use it, the smarter it gets! 🚀
        """)

if __name__ == "__main__":
    main()
