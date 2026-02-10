"""
Food Health Analyzer - OPTIMIZED VERSION
Enhanced with:
- Better error handling
- API response caching
- Optimized model loading
- Progress indicators
- Improved performance

CHANGES FROM ORIGINAL:
✅ Added comprehensive error handling
✅ Implemented API caching (24hr TTL)
✅ Added retry logic with exponential backoff
✅ Better progress indicators
✅ Lazy model loading option
✅ Improved logging
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
import logging
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Cache configuration
CACHE_TTL = 86400  # 24 hours


def safe_file_read(filepath, default_value, file_type='json'):
    """Safely read a file with error handling"""
    try:
        if not os.path.exists(filepath):
            return default_value
        
        with open(filepath, 'r' if file_type == 'json' else 'rb') as f:
            if file_type == 'json':
                return json.load(f)
            else:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return default_value


def safe_file_write(filepath, data, file_type='json'):
    """Safely write to a file with error handling"""
    try:
        with open(filepath, 'w' if file_type == 'json' else 'wb') as f:
            if file_type == 'json':
                json.dump(data, f, indent=2)
            else:
                pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {e}")
        st.error(f"Failed to save data: {str(e)}")
        return False


def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    return None


class HybridFoodAnalyzer:
    def __init__(self):
        """Initialize with both models for better coverage"""
        self.img_size = (224, 224)
        self.load_user_corrections()
        
        # Initialize models lazily (only when needed)
        self._vit_model = None
        self._vit_extractor = None
        self._resnet_model = None
        
        # Enhanced health categorization with point system (1-10)
        self.health_scores = {
            # Very Healthy Foods (8-10 points)
            'vegetables': 9, 'broccoli': 10, 'spinach': 10, 'kale': 10, 'beet': 9, 'beetroot': 9,
            'carrot': 9, 'tomato': 9, 'lettuce': 9, 'cucumber': 9, 'bell pepper': 9, 
            'zucchini': 9,'cauliflower': 9, 'brussels sprouts': 9, 'asparagus': 9, 'celery': 9,
            'onion': 8, 'garlic': 9, 'ginger': 8, 'fruits': 8, 'fruit': 8,'apple': 10, 'banana': 8, 
            'orange': 9, 'berries': 10, 'strawberry': 10, 'blueberry': 10, 'raspberry': 10, 
            'cherry':10,  'strawberries': 10, 'blueberries': 10, 'raspberries': 10,
            'peach':9,  'quince':10, 'nectarine':9, 'apricot':9, 'medlar':9, 'melon':9, 
            'watermelon': 9, 'pear': 8, 'grape': 8, 'pineapple': 8, 'mango': 8, 'avocado': 9,
            'salmon': 8, 'tuna': 8, 'sardines': 9, 'mackerel': 9, 'fish': 8, 'kohlrabi':10, 
            'turnip':9, 'artichoke':10, 'salad': 9, 'arugula':9, 'rocket salad':9, 'rumex':10, 
            'shrimp': 7, 'crab': 7, 'lobster': 7, 'mussels': 7, 'seafood': 7,
            'chicken breast': 8, 'turkey': 8, 'lean meat': 8, 'chicken': 7,
            'lentils': 9, 'chickpeas': 9, 'beans': 9, 'quinoa': 9, 'oatmeal': 9,
            'brown rice': 8, 'whole grain': 8, 'nuts': 8, 'almonds': 8, 'walnuts': 9,
            'greek yogurt': 8, 'cottage cheese': 8, 'white cheese': 8, 'tempeh': 8, 'edamame': 9, 'hummus': 8, 
            'seaweed': 9, 'herbs': 8, 'basil': 8, 'parsley': 8, 'cilantro': 8, 'dill':9, 
            'lime': 8, 'lemon': 8, 'mushroom': 9, 'fungi':9, 'peppers': 9, 'ginger': 8,
            
            # Moderately Healthy/Neutral Foods (5-7 points)
            'pasta': 6, 'white rice': 6, 'bread': 6, 'whole wheat bread': 6,
            'rice': 6, 'noodles': 6, 'couscous': 6, 'polenta': 6,
            'potato': 6, 'sweet potato': 7, 'corn': 5, 'peas': 7,
            'egg': 7, 'eggs': 7, 'cheese': 6, 'milk': 7, 'yogurt': 7,
            'peanut butter': 6, 'honey': 6, 'dark chocolate': 7, 'peanuts': 6,
            'olive oil': 7, 'coconut oil': 6, 'butter': 5, 'oil': 5, 'cream': 5,
            'pork': 6, 'beef': 6, 'lamb': 6, 'sausage': 5, 'meat': 6,
            'soup': 6, 'stew': 6, 'curry': 6, 'chili': 5, 'pickles': 5,
            'sandwich': 5, 'wrap': 5, 'taco': 5, 'burrito': 5,
            'sushi': 7, 'maki': 7, 'nigiri': 7, 'mustard': 5, 'cinnamon': 7, 
            'smoothie': 7, 'protein shake': 6, 'juice': 6,
            'granola': 6, 'cereal': 6, 'muesli': 7, 'bagel': 5,
            'tortilla': 6, 'pita': 6, 'crackers': 5, 'vinegar': 7, 
            'salt': 5, 'pepper': 7, 'spices': 7, 'yeast': 6,
            'flour': 5, 'wheat': 6, 'water': 7, 'broth': 6,
            'soy sauce': 5, 'salsa': 6, 'sauce': 5, 'marinara': 6, 'pesto': 6,
            'mozzarella': 6, 'parmesan': 6, 'ricotta': 6, 'feta': 5, 'cheddar': 5,
            'cream cheese': 5, 'sour cream': 5, 'mascarpone': 5, 'burrata': 5,
            'cocoa': 6, 'chocolate': 5, 'vanilla': 6, 'coffee': 6,
            'rice paper': 6, 'seitan': 7, 'chickpea': 9,'milkshake': 5,
            'chocolate bar': 5,  'gelato': 5, 'pancakes': 5,'pancake': 5,
            
            # Unhealthy Foods (1-4 points)
            'pizza': 4, 'burger': 3, 'hamburger': 3, 'cheeseburger': 3,
            'french fries': 2, 'fries': 2, 'chips': 2, 'nachos': 3,
            'hot dog': 3, 'corn dog': 2, 'fried chicken': 3, 'ketchup': 3, 
            'doughnut': 2, 'donut': 2, 'pastry': 3, 'croissant': 4, 'popcorn': 4,
            'cake': 3, 'cupcake': 2, 'brownie': 3, 'cookie': 4, 'cookies': 4,
            'candy': 1,  'soda': 1, 'energy drink': 1, 'sports drink': 2,
            'baking powder': 3, 'baking soda': 3,'tofu': 3, 'mayo': 4, 'mayonnaise': 4,
            'bacon': 3, 'pepperoni': 2, 'salami': 2, 'hot wings': 3,
            'fried': 2, 'deep fried': 2, 'battered': 2, 'breaded': 3, 'breadcrumbs': 3,
            'onion rings': 2, 'mozzarella sticks': 3, 'cheese fries': 2,
            'mac and cheese': 4, 'alfredo': 3, 'carbonara': 4,
            'ramen': 4, 'instant noodles': 3, 'cup noodles': 3,
            'white bread': 4, 'white toast': 3,  'waffles': 4, 'waffle': 4,'ice cream': 4,
            'syrup': 2, 'jam': 4, 'frosting': 2, 'whipped cream': 3
        }
        
        # Common food allergens (FDA's Big 9 + other common allergens)
        self.allergens = {
            # FDA Big 9 Allergens
            'milk': ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'casein', 
                    'lactose', 'cream cheese', 'sour cream', 'ice cream', 'gelato',
                    'mozzarella', 'parmesan', 'cheddar', 'ricotta', 'feta', 'mascarpone',
                    'cottage cheese', 'greek yogurt', 'milkshake', 'dairy', 'whipped cream'],
            
            'eggs': ['egg', 'eggs', 'mayonnaise', 'mayo'],
            
            'fish': ['fish', 'salmon', 'tuna', 'cod', 'bass', 'sardines', 'sardine','mackerel',
                    'anchovy', 'herring', 'trout', 'halibut', 'catfish'],
            
            'shellfish': ['shrimp', 'crab', 'lobster', 'prawns', 'crayfish', 'mussels',
                         'oyster', 'clams', 'scallops', 'prawn', 'mussel', 'seafood'],
            
            'tree nuts': ['nuts', 'almonds', 'walnuts', 'cashews', 'pecans', 'pistachios',
                         'hazelnuts', 'macadamia', 'pine nuts', 'chestnuts', 'nut', 
                         'almond', 'walnut', 'cashew', 'pecan', 'pistachio',
                         'hazelnut', 'pine nut', 'chestnut'],
            
            'peanuts': ['peanuts', 'peanut butter', 'peanut oil'],
            
            'wheat': ['wheat', 'flour', 'bread', 'pasta', 'noodles', 'couscous',
                     'crackers', 'bagel', 'croissant', 'pita', 'tortilla',
                     'whole wheat bread', 'white bread', 'pancakes', 'waffles',
                     'cake', 'cookie', 'brownie', 'muffin', 'donut', 'pastry',
                     'white toast', 'french toast'],
            
            'soybeans': ['soy', 'soy sauce', 'tofu', 'tempeh', 'edamame', 'miso', 'soybeans'],
            
            'sesame': ['sesame', 'tahini', 'sesame oil', 'sesame seeds'],
            
            # Other Common Allergens
            'gluten': ['wheat', 'flour', 'bread', 'pasta', 'rye', 'barley', 'oats',
                      'noodles', 'couscous', 'seitan', 'cracker', 'crackers', 'bagel', 'pancakes', 'pancake'],
            
            'corn': ['corn', 'corn syrup', 'popcorn', 'corn oil', 'cornmeal', 'polenta'],
            
            'sulfites': ['wine', 'dried fruit', 'vinegar'],
            
            'mustard': ['mustard', 'mustard seeds', 'mustard oil'],
        }
        
        # Allergen severity/commonality (for display purposes)
        self.allergen_info = {
            'milk': {'severity': 'high', 'emoji': '🥛', 'description': 'Dairy/Lactose'},
            'eggs': {'severity': 'high', 'emoji': '🥚', 'description': 'Eggs'},
            'fish': {'severity': 'medium', 'emoji': '🐟', 'description': 'Fish'},
            'shellfish': {'severity': 'high', 'emoji': '🦐', 'description': 'Shellfish/Crustaceans'},
            'tree nuts': {'severity': 'high', 'emoji': '🌰', 'description': 'Tree Nuts'},
            'peanuts': {'severity': 'high', 'emoji': '🥜', 'description': 'Peanuts'},
            'wheat': {'severity': 'medium', 'emoji': '🌾', 'description': 'Wheat'},
            'soybeans': {'severity': 'medium', 'emoji': '🫘', 'description': 'Soy'},
            'sesame': {'severity': 'medium', 'emoji': '🫘', 'description': 'Sesame'},
            'gluten': {'severity': 'medium', 'emoji': '🌾', 'description': 'Gluten'},
            'corn': {'severity': 'low', 'emoji': '🌽', 'description': 'Corn'},
            'sulfites': {'severity': 'low', 'emoji': '🍷', 'description': 'Sulfites'},
            'mustard': {'severity': 'low', 'emoji': '🌭', 'description': 'Mustard'},
        }
    
    @property
    def vit_model(self):
        """Lazy load ViT model"""
        if self._vit_model is None:
            with st.spinner('🔄 Loading ViT model (first time only)...'):
                self._vit_extractor, self._vit_model = self.build_vit_model()
        return self._vit_model
    
    @property
    def vit_extractor(self):
        """Lazy load ViT extractor"""
        if self._vit_extractor is None:
            with st.spinner('🔄 Loading ViT model (first time only)...'):
                self._vit_extractor, self._vit_model = self.build_vit_model()
        return self._vit_extractor
    
    @property
    def resnet_model(self):
        """Lazy load ResNet model"""
        if self._resnet_model is None:
            with st.spinner('🔄 Loading ResNet model (first time only)...'):
                self._resnet_model = self.build_resnet_model()
        return self._resnet_model
    
    @st.cache_resource
    def build_vit_model(_self):
        """Build ViT model for prepared dishes with error handling"""
        try:
            model_name = "nateraw/food"
            logger.info(f"Loading ViT model: {model_name}")
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            
            logger.info("ViT model loaded successfully")
            return feature_extractor, model
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            st.error(f"⚠️ Could not load ViT model. Using ResNet only. Error: {str(e)}")
            return None, None
    
    @st.cache_resource
    def build_resnet_model(_self):
        """Build ResNet50 for ingredients and fruits with error handling"""
        try:
            logger.info("Loading ResNet50 model")
            model = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            logger.info("ResNet50 model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load ResNet model: {e}")
            st.error(f"⚠️ Could not load ResNet model. Error: {str(e)}")
            raise
    
    def load_user_corrections(self):
        """Load user corrections from previous sessions"""
        self.user_corrections = safe_file_read(USER_CORRECTIONS_FILE, [], 'pickle')
        self.feedback_log = safe_file_read(FEEDBACK_FILE, [], 'json')
        logger.info(f"Loaded {len(self.user_corrections)} corrections and {len(self.feedback_log)} feedback entries")
    
    def save_user_corrections(self):
        """Save user corrections for future learning"""
        safe_file_write(USER_CORRECTIONS_FILE, self.user_corrections, 'pickle')
        safe_file_write(FEEDBACK_FILE, self.feedback_log, 'json')
    
    def extract_image_features(self, img):
        """Extract features from image for similarity matching"""
        try:
            img_resized = img.resize(self.img_size)
            img_array = image.img_to_array(img_resized)
            
            avg_colors = img_array.mean(axis=(0, 1))
            color_variance = img_array.std(axis=(0, 1))
            brightness = img_array.mean()
            
            features = np.concatenate([avg_colors, color_variance, [brightness]])
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(7)  # Return zero vector on failure
    
    def check_user_corrections(self, img, features):
        """Check if similar images were corrected by user"""
        if not self.user_corrections:
            return None
        
        try:
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
        except Exception as e:
            logger.error(f"Error checking user corrections: {e}")
        
        return None
    
    def predict_with_vit(self, img):
        """Predict using ViT Food-101 model with error handling"""
        try:
            if self._vit_model is None or self._vit_extractor is None:
                logger.warning("ViT model not available")
                return []
            
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
        except Exception as e:
            logger.error(f"ViT prediction failed: {e}")
            st.warning("⚠️ ViT model prediction failed. Using ResNet only.")
            return []
    
    def predict_with_resnet(self, img):
        """Predict using ResNet50 ImageNet model with error handling"""
        try:
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
        except Exception as e:
            logger.error(f"ResNet prediction failed: {e}")
            st.error(f"⚠️ Image recognition failed: {str(e)}")
            return []
    
    def predict_food(self, img):
        """Smart prediction using both models and user corrections"""
        try:
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
            
            # Use progress bar for model predictions
            progress_text = st.empty()
            progress_text.text("🔍 Analyzing with AI models...")
            
            vit_results = self.predict_with_vit(img)
            resnet_results = self.predict_with_resnet(img)
            
            progress_text.empty()
            
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
        except Exception as e:
            logger.error(f"Food prediction failed: {e}")
            st.error(f"⚠️ Prediction failed: {str(e)}")
            return []
    
    def get_health_score(self, food_name):
        """
        Get health score (1-10) for a food item
        1-4: Unhealthy (red)
        5-7: Neutral/Moderate (yellow/orange)
        8-10: Healthy (green)
        """
        food_lower = food_name.lower().strip()
        
        # Step 1: Try exact match first
        if food_lower in self.health_scores:
            return self.health_scores[food_lower]
        
        # Step 2: Find matches and prioritize longer/more specific matches
        matches = []
        for key, score in self.health_scores.items():
            if key in food_lower:
                matches.append((key, score, len(key)))
            elif food_lower in key:
                matches.append((key, score, len(key)))
        
        if matches:
            matches.sort(key=lambda x: x[2], reverse=True)
            return matches[0][1]
        
        # Step 3: Default to neutral if unknown
        return 6
    
    def detect_allergens(self, ingredients_list):
        """Detect allergens in a list of ingredients"""
        detected_allergens = {}
        
        try:
            for ingredient in ingredients_list:
                ingredient_lower = ingredient.lower().strip()
                
                for allergen_name, allergen_items in self.allergens.items():
                    for allergen_item in allergen_items:
                        if allergen_item in ingredient_lower or ingredient_lower in allergen_item:
                            if allergen_name not in detected_allergens:
                                detected_allergens[allergen_name] = []
                            if ingredient not in detected_allergens[allergen_name]:
                                detected_allergens[allergen_name].append(ingredient)
                            break
        except Exception as e:
            logger.error(f"Allergen detection failed: {e}")
        
        return detected_allergens
    
    def get_allergen_summary(self, detected_allergens):
        """Create a formatted summary of detected allergens"""
        if not detected_allergens:
            return None
        
        summary = {
            'total_count': len(detected_allergens),
            'allergens': []
        }
        
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        
        for allergen_name in detected_allergens.keys():
            allergen_data = self.allergen_info.get(allergen_name, {
                'severity': 'medium',
                'emoji': '⚠️',
                'description': allergen_name.title()
            })
            
            summary['allergens'].append({
                'name': allergen_name,
                'emoji': allergen_data['emoji'],
                'description': allergen_data['description'],
                'severity': allergen_data['severity'],
                'ingredients': detected_allergens[allergen_name]
            })
        
        summary['allergens'].sort(key=lambda x: (severity_order.get(x['severity'], 1), x['name']))
        
        return summary
    
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
        healthy_ings = []
        neutral_ings = []
        unhealthy_ings = []
        
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
        try:
            correction_entry = {
                'predicted': predicted_food,
                'correct_food': correct_food,
                'features': features,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'count': 1
            }
            
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
            
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'predicted': predicted_food,
                'correct': correct_food,
                'confidence': confidence,
                'was_correct': predicted_food.lower() == correct_food.lower(),
                'action': 'feedback'
            }
            self.feedback_log.append(feedback_entry)
            
            self.save_user_corrections()
            logger.info(f"User correction saved: {predicted_food} -> {correct_food}")
        except Exception as e:
            logger.error(f"Failed to save user correction: {e}")
            st.error("Failed to save your feedback. Please try again.")
    
    def get_learning_stats(self):
        """Get statistics about learning progress"""
        total_feedback = len(self.feedback_log)
        correct_predictions = sum(1 for log in self.feedback_log if log.get('was_correct', False))
        
        if total_feedback > 0:
            accuracy = (correct_predictions / total_feedback) * 100
        else:
            accuracy = 0
        
        unique_foods = len(self.user_corrections)
        
        return {
            'total_feedback': total_feedback,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'unique_foods_learned': unique_foods
        }
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def fetch_nutrition_data_cached(_self, food_name):
        """Fetch nutrition data with caching (24 hour TTL)"""
        return _self._fetch_nutrition_data_impl(food_name)
    
    def _fetch_nutrition_data_impl(self, food_name):
        """Internal implementation of nutrition data fetching with retry logic"""
        def make_request():
            params = {
                'api_key': USDA_API_KEY,
                'query': food_name,
                'pageSize': 1,
                'dataType': ['Foundation', 'SR Legacy']
            }
            
            response = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        try:
            logger.info(f"Fetching nutrition data for: {food_name}")
            data = retry_with_backoff(make_request, max_retries=3)
            
            if not data or not data.get('foods'):
                logger.warning(f"No nutrition data found for: {food_name}")
                return None
            
            food_item = data['foods'][0]
            
            nutrients_dict = {}
            for nutrient in food_item.get('foodNutrients', []):
                name = nutrient.get('nutrientName', '')
                value = nutrient.get('value', 0)
                unit = nutrient.get('unitName', '')
                
                if name and value:
                    nutrients_dict[name] = f"{value} {unit}"
            
            result = {
                'name': food_item.get('description', food_name),
                'nutrients': nutrients_dict
            }
            
            logger.info(f"Successfully fetched nutrition data for: {food_name}")
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"USDA API timeout for: {food_name}")
            st.warning("⏱️ USDA API is taking too long. Please try again later.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"USDA API request failed for {food_name}: {e}")
            st.warning(f"⚠️ Could not fetch nutrition data: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching nutrition for {food_name}: {e}")
            st.error(f"❌ Unexpected error: {str(e)}")
            return None
    
    def analyze_health_from_nutrients(self, nutrients):
        """Analyze health rating based on nutritional content"""
        try:
            # Extract numeric values
            def get_nutrient_value(key_fragment):
                for key, value in nutrients.items():
                    if key_fragment.lower() in key.lower():
                        try:
                            return float(value.split()[0])
                        except:
                            return 0
                return 0
            
            protein = get_nutrient_value('protein')
            fat = get_nutrient_value('fat')
            saturated_fat = get_nutrient_value('saturated')
            carbs = get_nutrient_value('carbohydrate')
            fiber = get_nutrient_value('fiber')
            sugar = get_nutrient_value('sugar')
            sodium = get_nutrient_value('sodium')
            
            # Simple scoring algorithm
            score = 0
            
            # Positive factors
            if protein > 10: score += 1
            if fiber > 5: score += 1
            if protein > 15: score += 1
            
            # Negative factors
            if saturated_fat > 5: score -= 1
            if sugar > 20: score -= 2
            if sodium > 500: score -= 1
            if fat > 30: score -= 1
            
            # Determine category
            if score >= 1:
                return "Healthy", "🟢"
            elif score >= -1:
                return "Moderate", "🟡"
            else:
                return "Unhealthy", "🔴"
        except Exception as e:
            logger.error(f"Health analysis failed: {e}")
            return "Unknown", "⚪"
    
    def extract_ingredients(self, food_name):
        """Extract potential ingredients from food name"""
        food_lower = food_name.lower().strip()
        
        ingredients = []
        
        separators = [' and ', ' with ', ',', ' in ', ' on ']
        parts = [food_lower]
        
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        for part in parts:
            part = part.strip()
            if len(part) > 2:
                ingredients.append(part)
        
        return ingredients if len(ingredients) > 1 else [food_lower]
    
    def get_ingredients_from_database(self, food_name):
        """Built-in ingredient database for common dishes"""
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
            'banitsa': ['phyllo dough', 'eggs', 'cheese', 'soda', 'butter'],
            
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
            'nachos': ['tortilla chips', 'cheese', 'beans', 'salsa', 'sour cream', 'jalapeno'],
            
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
            'shopska salad': ['tomato', 'cucumber', 'onion', 'pepper', 'white cheese', 'olive oil', 'vinegar'],
            'vitamine salad': ['beet', 'carrot', 'apple', 'кohlrabi', 'olive oil'],
            'potato salad': ['mashed potatoes', 'celery', 'onion', 'eggs', 'olive oil'],
            
            # Soups
            'chicken soup': ['chicken', 'broth', 'carrot', 'celery', 'onion', 'noodles'],
            'tomato soup': ['tomato', 'cream', 'onion', 'garlic', 'basil', 'butter'],
            'minestrone': ['pasta', 'beans', 'tomato', 'carrot', 'celery', 'onion'],
            'mushroom soup': ['mushrooms', 'cream', 'garlic', 'onion', 'thyme', 'parsley'],
            'goulash': ['beef', 'pepper', 'garlic', 'onion', 'tomato', 'cheddar', 'noodles'],
            
            # Meat dishes
            'steak': ['beef', 'salt', 'pepper', 'butter', 'garlic'],
            'chicken breast': ['chicken', 'salt', 'pepper', 'oil', 'herbs'],
            'pork chop': ['pork', 'salt', 'pepper', 'oil', 'garlic'],
            'fish fillet': ['fish', 'salt', 'pepper', 'lemon', 'butter'],
            
            # More breakfast
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
    
    def get_recipe_ingredients(self, food_name):
        """Get ingredients using Wikipedia API and fallback database"""
        
        # First, try predefined database for common dishes
        ingredients = self.get_ingredients_from_database(food_name)
        if ingredients:
            return {
                'ingredients': ingredients,
                'source': 'Built-in Database'
            }
        
        # If not in database, try Wikipedia
        try:
            import wikipediaapi
            
            wiki = wikipediaapi.Wikipedia(
                user_agent='FoodHealthAnalyzer/1.0 (Educational App)',
                language='en'
            )
            
            page = wiki.page(food_name)
            
            if not page.exists():
                page = wiki.page(f"{food_name} (food)")
            
            if not page.exists():
                return None
            
            text = page.text.lower()
            
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
            
            found_ingredients = []
            
            for ingredient in common_ingredients:
                if ingredient in text:
                    if ingredient == 'eggs' and 'egg' in found_ingredients:
                        continue
                    if ingredient == 'egg' and 'eggs' in found_ingredients:
                        continue
                    
                    if ingredient not in found_ingredients:
                        found_ingredients.append(ingredient)
            
            if found_ingredients:
                return {
                    'ingredients': found_ingredients[:15],
                    'source': 'Wikipedia'
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Wikipedia fetch failed: {e}")
            return None
    
    def get_food_from_usda(self, food_name, num_results=5):
        """Fetch food data from USDA API (uses cached version)"""
        return self.fetch_nutrition_data_cached(food_name)
    
    def extract_nutrition_info(self, food_data):
        """Extract key nutritional information"""
        if not food_data:
            return None
        
        try:
            # food_data is already processed nutrition data from cached method
            return food_data
        except Exception as e:
            logger.error(f"Nutrition extraction failed: {e}")
            return None
    
    def extract_ingredients_from_usda(self, food_data):
        """Extract ingredients list from USDA data - NOT IMPLEMENTED in current API"""
        # Note: The current caching structure doesn't return raw USDA data
        # This method is kept for compatibility but won't work with current implementation
        return None


# Initialize analyzer
@st.cache_resource
def get_analyzer():
    """Get or create the analyzer instance"""
    logger.info("Initializing HybridFoodAnalyzer")
    return HybridFoodAnalyzer()


def main():
    st.title("🍎 Food Health Analyzer with Smart Scoring")
    st.markdown("**AI-powered food recognition with 1-10 health scoring system**")
    
    try:
        analyzer = get_analyzer()
    except Exception as e:
        st.error(f"❌ Failed to initialize analyzer: {str(e)}")
        logger.error(f"Analyzer initialization failed: {e}", exc_info=True)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🧠 Learning Statistics")
        
        stats = analyzer.get_learning_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Feedback", stats['total_feedback'])
            st.metric("Corrections", stats['unique_foods_learned'])
        
        with col2:
            st.metric("Correct", stats['correct_predictions'])
            if stats['total_feedback'] > 0:
                st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
        
        if stats['total_feedback'] > 0:
            st.progress(stats['accuracy'] / 100, text=f"Model Accuracy: {stats['accuracy']:.1f}%")
        
        st.markdown("---")
        
        if analyzer.user_corrections:
            st.markdown("### 📚 Recently Learned")
            recent_corrections = analyzer.user_corrections[-5:]
            for correction in reversed(recent_corrections):
                food_name = correction['correct_food'].title()
                count = correction.get('count', 1)
                st.write(f"✅ {food_name} (×{count})")
        
        st.markdown("---")
        st.markdown("### 🎯 Health Score Guide")
        st.markdown("""
        **🟢 8-10**: Highly nutritious
        **🟡 5-7**: Neutral/Moderate
        **🔴 1-4**: Limit consumption
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Allergen Detection")
        st.markdown("""
        Automatically detects:
        - 🥛 Dairy/Milk
        - 🥚 Eggs
        - 🐟 Fish
        - 🦐 Shellfish
        - 🌰 Tree Nuts
        - 🥜 Peanuts
        - 🌾 Wheat/Gluten
        - 🫘 Soy
        - And more!
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
        try:
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to open image: {str(e)}")
            logger.error(f"Image opening failed: {e}")
            st.stop()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            predictions = analyzer.predict_food(img)
            
            if not predictions:
                st.error("❌ Failed to analyze image. Please try again with a different image.")
                st.stop()
            
            st.subheader("🤖 AI Predictions")
            
            # Top prediction
            top_pred = predictions[0]
            top_food_name = top_pred['name'].replace('_', ' ').title()
            top_confidence = top_pred['confidence']
            top_source = top_pred['source']
            
            top_score = analyzer.get_health_score(top_pred['name'])
            top_category, top_emoji, top_color = analyzer.get_health_category(top_score)
            
            st.markdown(f"### **Top Match:** {top_food_name}")
            st.progress(top_confidence, text=f"🎯 Recognition Confidence: {top_confidence:.1%} ({top_source})")
            st.markdown(f"Health Score: {top_emoji} **{top_score}/10** ({top_category})")
            
            # Additional predictions
            if len(predictions) > 1:
                st.markdown("---")
                st.markdown("**Other Possibilities:**")
                
                for i, pred in enumerate(predictions[1:4], 2):
                    food_name = pred['name'].replace('_', ' ').title()
                    confidence = pred['confidence']
                    source = pred['source']
                    
                    score = analyzer.get_health_score(pred['name'])
                    category, emoji, color = analyzer.get_health_category(score)
                    
                    confidence_pct = confidence * 100
                    st.write(f"**{i}.** {food_name} - {confidence_pct:.2f}% ({source}) | {emoji} {score}/10")
            
            st.session_state.current_prediction = predictions[0]
        
        st.markdown("---")
        
        # Detailed analysis
        top_prediction = predictions[0]
        top_food = top_prediction['name'].replace('_', ' ')
        
        st.header(f"📋 Detailed Analysis: {top_food.title()}")
        
        # Recognition Accuracy
        st.subheader("🎯 Recognition Accuracy")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_pct = top_prediction['confidence'] * 100
            st.metric(
                label="Model Confidence",
                value=f"{confidence_pct:.1f}%",
                help="How confident the AI is in this prediction"
            )
        
        with col2:
            st.metric(
                label="Detection Source",
                value=top_prediction['source'],
                help="Which AI model made this prediction"
            )
        
        with col3:
            if confidence_pct >= 90:
                confidence_rating = "Very High ⭐⭐⭐"
            elif confidence_pct >= 75:
                confidence_rating = "High ⭐⭐"
            elif confidence_pct >= 60:
                confidence_rating = "Moderate ⭐"
            else:
                confidence_rating = "Low"
            
            st.metric(
                label="Accuracy Rating",
                value=confidence_rating,
                help="Overall reliability of this prediction"
            )
        
        # Confidence interpretation
        if confidence_pct >= 90:
            st.success("✅ **Very confident prediction** - The model is highly certain about this identification.")
        elif confidence_pct >= 75:
            st.info("ℹ️ **Confident prediction** - The model has good certainty about this identification.")
        elif confidence_pct >= 60:
            st.warning("⚠️ **Moderate confidence** - The model is somewhat uncertain. Please verify the prediction.")
        else:
            st.error("⚠️ **Low confidence** - The model is not very certain. Consider providing a correction to help it learn.")
        
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
        
        # Health Score Section
        score = analyzer.get_health_score(top_food)
        category, emoji, color = analyzer.get_health_category(score)
        advice = analyzer.get_health_advice(score, top_food)
        
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
        
        # Try to get recipe ingredients
        with st.spinner("🔍 Looking for recipe ingredients..."):
            recipe_data = analyzer.get_recipe_ingredients(top_food)
        
        if recipe_data:
            st.subheader(f"🥘 Recipe Ingredients ({recipe_data['source']})")
            
            if recipe_data['source'] == 'Wikipedia':
                st.info("✅ Found ingredients from Wikipedia article")
            elif recipe_data['source'] == 'Built-in Database':
                st.info("✅ Found ingredients from built-in recipe database")
            
            ingredients_list = recipe_data['ingredients']
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
                
                st.markdown(f"**Overall Recipe Score: {avg_score:.1f}/10**")
                st.progress(avg_score / 10)
                
                if avg_score >= 7:
                    st.success("💚 This dish contains mostly healthy ingredients!")
                elif avg_score >= 5:
                    st.info("ℹ️ This dish has a balanced mix of ingredients.")
                else:
                    st.warning("⚠️ This dish contains ingredients to consume in moderation.")
            
            # Allergen Detection
            detected_allergens = analyzer.detect_allergens(ingredients_list)
            allergen_summary = analyzer.get_allergen_summary(detected_allergens)
            
            if allergen_summary:
                st.markdown("---")
                st.subheader(f"⚠️ Allergen Warning ({allergen_summary['total_count']} detected)")
                
                st.warning("**This dish may contain the following allergens:**")
                
                high_severity = [a for a in allergen_summary['allergens'] if a['severity'] == 'high']
                medium_severity = [a for a in allergen_summary['allergens'] if a['severity'] == 'medium']
                low_severity = [a for a in allergen_summary['allergens'] if a['severity'] == 'low']
                
                if high_severity:
                    st.markdown("**🔴 High Priority Allergens (Major):**")
                    for allergen in high_severity:
                        ingredients_str = ", ".join([ing.title() for ing in allergen['ingredients']])
                        st.write(f"• {allergen['emoji']} **{allergen['description']}**: {ingredients_str}")
                
                if medium_severity:
                    st.markdown("**🟡 Medium Priority Allergens:**")
                    for allergen in medium_severity:
                        ingredients_str = ", ".join([ing.title() for ing in allergen['ingredients']])
                        st.write(f"• {allergen['emoji']} **{allergen['description']}**: {ingredients_str}")
                
                if low_severity:
                    with st.expander("🟢 Low Priority Allergens (Click to expand)"):
                        for allergen in low_severity:
                            ingredients_str = ", ".join([ing.title() for ing in allergen['ingredients']])
                            st.write(f"• {allergen['emoji']} **{allergen['description']}**: {ingredients_str}")
                
                st.info("💡 **Note:** This is an automated detection based on ingredient names. Always verify with the manufacturer or restaurant for accurate allergen information.")
            
            st.markdown("---")
        
        # Try to get USDA nutritional data
        with st.spinner("🔍 Fetching USDA nutritional data..."):
            nutrition_data = analyzer.fetch_nutrition_data_cached(top_food)
        
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
