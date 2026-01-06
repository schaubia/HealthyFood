"""
Food Health Analyzer - Enhanced with Learning and Point-Based Rating System
Uses both Food-101 (for dishes) and ResNet50 (for ingredients/fruits)
WITH USER FEEDBACK, LEARNING MECHANISM, AND 1-10 HEALTH SCORING
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
import time

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
        
        # Enhanced health categorization with point system (1-10)
        self.health_scores = {
            # Very Healthy Foods (8-10 points)
            'vegetables': 9, 'broccoli': 10, 'spinach': 10, 'kale': 10, 'carrot': 9,
            'tomato': 9, 'lettuce': 9, 'cucumber': 9, 'bell pepper': 9, 'zucchini': 9,
            'cauliflower': 9, 'brussels sprouts': 9, 'asparagus': 9, 'celery': 9,
            'fruits': 8, 'apple': 9, 'banana': 8, 'orange': 9, 'berries': 10,
            'strawberry': 9, 'blueberry': 10, 'raspberry': 9, 'watermelon': 9,
            'pear': 8, 'grape': 8, 'pineapple': 8, 'mango': 8, 'avocado': 9,
            'salmon': 9, 'tuna': 8, 'sardines': 9, 'mackerel': 9,
            'chicken breast': 8, 'turkey': 8, 'lean meat': 8,
            'lentils': 9, 'chickpeas': 9, 'beans': 9, 'quinoa': 9, 'oatmeal': 9,
            'brown rice': 8, 'whole grain': 8, 'nuts': 8, 'almonds': 8, 'walnuts': 9,
            'greek yogurt': 8, 'cottage cheese': 8, 'tofu': 8, 'tempeh': 8,
            'edamame': 9, 'hummus': 8, 'salad': 9,
            
            # Moderately Healthy/Neutral Foods (5-7 points)
            'pasta': 6, 'white rice': 6, 'bread': 6, 'whole wheat bread': 7,
            'rice': 6, 'noodles': 6, 'couscous': 6, 'polenta': 6,
            'potato': 6, 'sweet potato': 7, 'corn': 6, 'peas': 7,
            'egg': 7, 'eggs': 7, 'cheese': 6, 'milk': 7, 'yogurt': 7,
            'peanut butter': 6, 'honey': 6, 'dark chocolate': 7,
            'olive oil': 7, 'coconut oil': 6, 'butter': 5,
            'pork': 6, 'beef': 6, 'lamb': 6, 'sausage': 5,
            'shrimp': 7, 'crab': 7, 'lobster': 7, 'mussels': 7,
            'soup': 6, 'stew': 6, 'curry': 6, 'chili': 6,
            'sandwich': 6, 'wrap': 6, 'taco': 6, 'burrito': 5,
            'sushi': 7, 'maki': 7, 'nigiri': 7,
            'smoothie': 7, 'protein shake': 7, 'juice': 6,
            'granola': 6, 'cereal': 6, 'muesli': 7, 'bagel': 5,
            'tortilla': 6, 'pita': 6, 'crackers': 5,
            
            # Unhealthy Foods (1-4 points)
            'pizza': 4, 'burger': 3, 'hamburger': 3, 'cheeseburger': 3,
            'french fries': 2, 'fries': 2, 'chips': 2, 'nachos': 3,
            'hot dog': 3, 'corn dog': 2, 'fried chicken': 3,
            'doughnut': 2, 'donut': 2, 'pastry': 3, 'croissant': 4,
            'cake': 3, 'cupcake': 2, 'brownie': 3, 'cookie': 3, 'cookies': 3,
            'candy': 1, 'chocolate bar': 3, 'ice cream': 3, 'gelato': 3,
            'milkshake': 2, 'soda': 1, 'energy drink': 2, 'sports drink': 3,
            'bacon': 3, 'pepperoni': 3, 'salami': 3, 'hot wings': 3,
            'fried': 2, 'deep fried': 2, 'battered': 2, 'breaded': 3,
            'onion rings': 2, 'mozzarella sticks': 3, 'cheese fries': 2,
            'mac and cheese': 4, 'alfredo': 3, 'carbonara': 4,
            'ramen': 4, 'instant noodles': 3, 'cup noodles': 3,
            'white bread': 5, 'white toast': 5, 'pancakes': 4, 'waffles': 4,
            'syrup': 2, 'jam': 4, 'frosting': 2, 'whipped cream': 3
        }
        
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
        img_resized = img.resize(self.img_size)
        img_array = image.img_to_array(img_resized)
        
        avg_colors = img_array.mean(axis=(0, 1))
        color_variance = img_array.std(axis=(0, 1))
        brightness = img_array.mean()
        
        features = np.concatenate([avg_colors, color_variance, [brightness]])
        return features
    
    def check_user_corrections(self, img, features):
        """Check if similar images were corrected by user"""
        if not self.user_corrections:
            return None
        
        similarities = []
        for correction in self.user_corrections:
            if 'features' in correction:
                saved_features = np.array(correction['features'])
                distance = np.linalg.norm(features - saved_features)
                similarity = 1 / (1 + distance)
                
                if similarity > 0.85:
                    similarities.append({
                        'food': correction['correct_food'],
                        'similarity': similarity,
                        'count': correction.get('count', 1)
                    })
        
        if similarities:
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
        features = self.extract_image_features(img)
        
        user_match = self.check_user_corrections(img, features)
        
        if user_match:
            st.info(f"🧠 Found similar correction from learning: {user_match['food']} (similarity: {user_match['similarity']:.1%})")
            return [{
                'name': user_match['food'],
                'confidence': 0.95,
                'source': 'User Learning',
                'features': features.tolist()
            }]
        
        vit_results = self.predict_with_vit(img)
        resnet_results = self.predict_with_resnet(img)
        
        food_keywords = ['fruit', 'vegetable', 'meat', 'fish', 'berry', 'apple', 'orange', 
                        'banana', 'pear', 'grape', 'lemon', 'mushroom', 'corn', 'pepper',
                        'tomato', 'potato', 'carrot', 'broccoli', 'strawberry', 'pizza',
                        'burger', 'sandwich', 'salad', 'bread', 'cheese', 'chocolate']
        
        food_resnet_results = [
            r for r in resnet_results 
            if any(keyword in r['name'].lower() for keyword in food_keywords)
        ]
        
        combined_results = vit_results + food_resnet_results
        combined_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        for result in combined_results:
            result['features'] = features.tolist()
        
        return combined_results[:5] if combined_results else vit_results
    
    def get_health_score(self, food_name):
        """
        Get health score (1-10) for a food item
        1-4: Unhealthy (red)
        5-7: Neutral/Moderate (yellow/orange)
        8-10: Healthy (green)
        """
        food_lower = food_name.lower().strip()
        
        # Direct match
        if food_lower in self.health_scores:
            return self.health_scores[food_lower]
        
        # Partial match
        for key, score in self.health_scores.items():
            if key in food_lower or food_lower in key:
                return score
        
        # Default to neutral if unknown
        return 6
    
    def get_health_category(self, score):
        """Convert score to category"""
        if score >= 8:
            return "Healthy", "🟢", "#28a745"
        elif score >= 5:
            return "Neutral", "🟡", "#ffc107"
        else:
            return "Unhealthy", "🔴", "#dc3545"
    
    def get_health_advice(self, score, food_name):
        """Generate personalized health advice based on score"""
        category, emoji, color = self.get_health_category(score)
        
        if score >= 9:
            return f"{emoji} **Excellent choice!** {food_name.title()} is highly nutritious and great for your health. This food is rich in beneficial nutrients that support overall wellness."
        elif score >= 8:
            return f"{emoji} **Great option!** {food_name.title()} is a healthy food that contributes positively to a balanced diet. Enjoy it regularly!"
        elif score >= 7:
            return f"{emoji} **Good choice!** {food_name.title()} is moderately healthy with decent nutritional value. It's a solid part of a balanced diet."
        elif score >= 6:
            return f"{emoji} **Fair option.** {food_name.title()} is neutral - not particularly healthy or unhealthy. Balance it with more nutritious foods throughout the day."
        elif score >= 5:
            return f"🟠 **Consume mindfully.** {food_name.title()} should be eaten in moderation. Try to pair it with healthier options and be aware of portion sizes."
        elif score >= 3:
            return f"🔴 **Occasional treat only.** {food_name.title()} is high in calories, fats, or sugars. Enjoy it rarely and in small portions as part of an otherwise healthy diet."
        else:
            return f"🔴 **Limit intake.** {food_name.title()} is very unhealthy and should be avoided or consumed very rarely. Consider healthier alternatives whenever possible."
    
    def analyze_ingredients_health_with_scores(self, ingredients):
        """Categorize ingredients by health score"""
        healthy_ings = []      # 8-10
        neutral_ings = []      # 5-7
        unhealthy_ings = []    # 1-4
        
        for ing in ingredients:
            score = self.get_health_score(ing)
            if score >= 8:
                healthy_ings.append((ing, score))
            elif score >= 5:
                neutral_ings.append((ing, score))
            else:
                unhealthy_ings.append((ing, score))
        
        return healthy_ings, neutral_ings, unhealthy_ings
    
    def add_user_correction(self, predicted_food, correct_food, features, confidence):
        """Add user correction to learning dataset"""
        correction_entry = {
            'predicted': predicted_food,
            'correct_food': correct_food,
            'features': features,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'count': 1
        }
        
        # Check if similar correction exists
        found = False
        for correction in self.user_corrections:
            if correction['correct_food'].lower() == correct_food.lower():
                if 'features' in correction:
                    saved_features = np.array(correction['features'])
                    current_features = np.array(features)
                    distance = np.linalg.norm(saved_features - current_features)
                    similarity = 1 / (1 + distance)
                    
                    if similarity > 0.85:
                        correction['count'] = correction.get('count', 1) + 1
                        correction['last_seen'] = datetime.now().isoformat()
                        found = True
                        break
        
        if not found:
            self.user_corrections.append(correction_entry)
        
        # Add to feedback log
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted_food,
            'correct': correct_food,
            'confidence': confidence,
            'action': 'correction'
        }
        self.feedback_log.append(feedback_entry)
        
        self.save_user_corrections()
    
    def get_food_from_usda(self, food_name, num_results=5):
        """Fetch food data from USDA API"""
        try:
            params = {
                'api_key': USDA_API_KEY,
                'query': food_name,
                'pageSize': num_results,
                'dataType': ['Foundation', 'SR Legacy']
            }
            
            response = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('foods'):
                    return data['foods']
            return None
        except Exception as e:
            st.warning(f"⚠️ Could not fetch USDA data: {str(e)}")
            return None
    
    def extract_nutrition_info(self, food_data):
        """Extract key nutritional information"""
        if not food_data:
            return None
        
        first_food = food_data[0]
        nutrients = {}
        
        if 'foodNutrients' in first_food:
            for nutrient in first_food['foodNutrients']:
                nutrient_name = nutrient.get('nutrientName', '')
                value = nutrient.get('value', 0)
                unit = nutrient.get('unitName', '')
                
                if value and nutrient_name:
                    nutrients[nutrient_name] = f"{value:.1f} {unit}"
        
        return {
            'name': first_food.get('description', 'Unknown'),
            'nutrients': nutrients
        }
    
    def extract_ingredients(self, food_data):
        """Extract ingredients list"""
        if not food_data:
            return None
        
        first_food = food_data[0]
        ingredients_text = first_food.get('ingredients', '')
        
        if ingredients_text:
            ingredients_list = [
                ing.strip() 
                for ing in ingredients_text.lower().split(',')
                if ing.strip()
            ]
            return {
                'ingredients': ingredients_list[:10],
                'raw_text': ingredients_text
            }
        
        return None
    
    def analyze_health_from_nutrients(self, nutrients):
        """Analyze health rating based on nutritional data"""
        if not nutrients:
            return "Unknown", "❓"
        
        issues = []
        
        for key, value in nutrients.items():
            value_str = str(value).split()[0]
            try:
                num_value = float(value_str)
                
                if 'Total lipid (fat)' in key and num_value > 20:
                    issues.append('high_fat')
                if 'Sugars' in key and num_value > 15:
                    issues.append('high_sugar')
                if 'Sodium' in key and num_value > 400:
                    issues.append('high_sodium')
                if 'Cholesterol' in key and num_value > 50:
                    issues.append('high_cholesterol')
            except:
                continue
        
        if len(issues) >= 2:
            return "Unhealthy", "🔴"
        elif len(issues) == 1:
            return "Moderate", "🟡"
        else:
            return "Healthy", "🟢"

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return HybridFoodAnalyzer()

def main():
    st.title("🍎 Food Health Analyzer with Smart Scoring")
    st.markdown("**AI-powered food recognition with 1-10 health scoring system**")
    
    analyzer = get_analyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Learning Statistics")
        
        total_corrections = len(analyzer.user_corrections)
        total_feedback = len(analyzer.feedback_log)
        
        st.metric("Total Corrections", total_corrections)
        st.metric("Total Interactions", total_feedback)
        
        if analyzer.user_corrections:
            st.markdown("### 🧠 Most Corrected Foods")
            food_counts = {}
            for correction in analyzer.user_corrections:
                food = correction['correct_food']
                food_counts[food] = food_counts.get(food, 0) + correction.get('count', 1)
            
            sorted_foods = sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for food, count in sorted_foods:
                st.write(f"• **{food.title()}**: {count}x")
        
        st.markdown("---")
        st.markdown("### 🎯 Health Score Guide")
        st.markdown("""
        **🟢 8-10**: Highly nutritious
        **🟡 5-7**: Neutral/Moderate
        **🔴 1-4**: Limit consumption
        """)
        
        st.markdown("---")
        if st.button("🗑️ Clear Learning Data"):
            if os.path.exists(USER_CORRECTIONS_FILE):
                os.remove(USER_CORRECTIONS_FILE)
            if os.path.exists(FEEDBACK_FILE):
                os.remove(FEEDBACK_FILE)
            st.success("✅ Learning data cleared!")
            time.sleep(1)
            st.rerun()
    
    # Main content
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📤 Upload a food image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of food for AI analysis"
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Analyzing image with AI..."):
                predictions = analyzer.predict_food(img)
            
            if predictions:
                st.subheader("🤖 AI Predictions")
                
                for i, pred in enumerate(predictions[:3], 1):
                    food_name = pred['name'].replace('_', ' ').title()
                    confidence = pred['confidence']
                    source = pred['source']
                    
                    # Get health score
                    score = analyzer.get_health_score(pred['name'])
                    category, emoji, color = analyzer.get_health_category(score)
                    
                    # Display prediction with score
                    st.markdown(f"""
                    **{i}. {food_name}** ({source})  
                    Confidence: {confidence:.1%} | Health Score: {emoji} **{score}/10** ({category})
                    """)
                
                st.session_state.current_prediction = predictions[0]
        
        st.markdown("---")
        
        # Detailed analysis for top prediction
        top_prediction = predictions[0]
        top_food = top_prediction['name'].replace('_', ' ')
        
        st.header(f"📋 Detailed Analysis: {top_food.title()}")
        
        # Health Score Section
        score = analyzer.get_health_score(top_food)
        category, emoji, color = analyzer.get_health_category(score)
        advice = analyzer.get_health_advice(score, top_food)
        
        # Score visualization
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"### {emoji}")
        
        with col2:
            st.markdown(f"### Health Score: **{score}/10**")
            st.progress(score / 10)
            st.markdown(f"**Category:** {category}")
        
        with col3:
            if score >= 8:
                st.success("Excellent!")
            elif score >= 5:
                st.warning("Moderate")
            else:
                st.error("Limit Intake")
        
        st.info(advice)
        
        st.markdown("---")
        
        # Feedback Section
        st.subheader("💬 Is this prediction correct?")
        
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
        
        # Try to get USDA nutritional data
        with st.spinner("🔍 Fetching nutritional data..."):
            usda_data = analyzer.get_food_from_usda(top_food)
            nutrition_data = analyzer.extract_nutrition_info(usda_data)
            ingredients_data = analyzer.extract_ingredients(usda_data)
        
        # Ingredients Section with Scores
        if ingredients_data:
            st.subheader("🥘 Ingredient Analysis")
            
            ingredients_list = ingredients_data['ingredients']
            healthy_ings, neutral_ings, unhealthy_ings = analyzer.analyze_ingredients_health_with_scores(ingredients_list)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🟢 Healthy (8-10)**")
                if healthy_ings:
                    for ing, score in healthy_ings:
                        st.write(f"• {ing.title()} ({score}/10)")
                else:
                    st.write("_None identified_")
            
            with col2:
                st.markdown("**🟡 Neutral (5-7)**")
                if neutral_ings:
                    for ing, score in neutral_ings:
                        st.write(f"• {ing.title()} ({score}/10)")
                else:
                    st.write("_None identified_")
            
            with col3:
                st.markdown("**🔴 Unhealthy (1-4)**")
                if unhealthy_ings:
                    for ing, score in unhealthy_ings:
                        st.write(f"• {ing.title()} ({score}/10)")
                else:
                    st.write("_None identified_")
            
            # Overall ingredient score
            if ingredients_list:
                total_score = (
                    sum(score for _, score in healthy_ings) +
                    sum(score for _, score in neutral_ings) +
                    sum(score for _, score in unhealthy_ings)
                )
                avg_score = total_score / len(ingredients_list)
                
                st.markdown(f"**Overall Ingredient Score: {avg_score:.1f}/10**")
                
                if avg_score >= 7:
                    st.success("💚 This dish contains mostly healthy ingredients!")
                elif avg_score >= 5:
                    st.info("ℹ️ This dish has a balanced mix of ingredients.")
                else:
                    st.warning("⚠️ This dish contains ingredients to consume in moderation.")
            
            st.markdown("---")
        
        # Nutritional Information
        if nutrition_data:
            nutrients = nutrition_data['nutrients']
            health_rating, health_emoji = analyzer.analyze_health_from_nutrients(nutrients)
            
            st.subheader("📊 Nutritional Information")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Food:** {nutrition_data['name']}")
                
                # Extract calories
                calories = "Not available"
                for key in nutrients.keys():
                    if 'energy' in key.lower() or 'calor' in key.lower():
                        calories = nutrients[key]
                        break
                
                st.metric(label="🔥 Calories (per 100g)", value=calories)
                st.write(f"**Nutrient-Based Rating:** {health_emoji} {health_rating}")
            
            with col2:
                if health_rating == "Healthy":
                    st.success("✅ Nutritious")
                elif health_rating == "Moderate":
                    st.warning("⚠️ Moderate")
                else:
                    st.error("⚠️ Watch Intake")
            
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
        else:
            st.info(f"ℹ️ Nutritional information not available for '{top_food}' in the USDA database.")
    
    else:
        st.info("👆 Upload an image to get started!")
        
        st.markdown("### 📸 Best Results For:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Individual Ingredients:**
            - 🍎 Fruits (apple, banana, orange)
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
        st.markdown("### 🎯 How the Scoring Works")
        st.markdown("""
        **Health scores range from 1-10:**
        - **8-10 (🟢)**: Highly nutritious foods rich in vitamins, minerals, and beneficial nutrients
        - **5-7 (🟡)**: Neutral foods that are okay in moderation
        - **1-4 (🔴)**: Foods high in calories, fats, sugars, or sodium - consume sparingly
        
        The app combines AI image recognition with nutritional science to give you instant health insights!
        """)

if __name__ == "__main__":
    main()
