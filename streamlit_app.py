"""
Food Health Analyzer - CLOUD OPTIMIZED VERSION
Specifically tuned for Streamlit Cloud (1 CPU, 800MB RAM)

OPTIMIZATIONS:
✅ Conditional model loading (load only what's needed)
✅ Aggressive memory management
✅ Minimal dependencies
✅ Smart caching strategies
✅ Lazy everything
✅ Progress indicators for perceived performance
"""

import streamlit as st
from PIL import Image
import requests
import os
import numpy as np
import json
from datetime import datetime
import time
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration - BEFORE any other Streamlit commands
st.set_page_config(
    page_title="Food Health Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"  # Save screen space
)

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Learning configuration
FEEDBACK_FILE = "food_feedback.json"
USER_CORRECTIONS_FILE = "user_corrections.pkl"

# Cache configuration - longer for cloud to reduce reloads
CACHE_TTL = 604800  # 7 days instead of 24 hours

# CLOUD OPTIMIZATION: Defer heavy imports until actually needed
_TENSORFLOW_LOADED = False
_TORCH_LOADED = False


def load_tensorflow():
    """Lazy load TensorFlow only when needed"""
    global _TENSORFLOW_LOADED
    if not _TENSORFLOW_LOADED:
        import tensorflow as tf
        # Optimize TensorFlow for cloud
        tf.config.set_visible_devices([], 'GPU')  # Force CPU mode
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings
        _TENSORFLOW_LOADED = True
        logger.info("TensorFlow loaded and optimized")


def load_torch():
    """Lazy load PyTorch only when needed"""
    global _TORCH_LOADED
    if not _TORCH_LOADED:
        import torch
        torch.set_num_threads(1)  # Single thread for cloud
        _TORCH_LOADED = True
        logger.info("PyTorch loaded")


def safe_file_read(filepath, default_value, file_type='json'):
    """Safely read a file with error handling"""
    try:
        if not os.path.exists(filepath):
            return default_value
        
        with open(filepath, 'r' if file_type == 'json' else 'rb') as f:
            if file_type == 'json':
                return json.load(f)
            else:
                import pickle
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
                import pickle
                pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {e}")
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


class CloudOptimizedFoodAnalyzer:
    """Streamlit Cloud optimized analyzer - minimal memory footprint"""
    
    def __init__(self):
        """Initialize with minimal memory - lazy load everything"""
        self.img_size = (224, 224)
        
        # Load user data
        self.user_corrections = safe_file_read(USER_CORRECTIONS_FILE, [], 'pickle')
        self.feedback_log = safe_file_read(FEEDBACK_FILE, [], 'json')
        
        # Models start as None - loaded only when needed
        self._resnet_model = None
        self._vit_model = None
        self._vit_processor = None
        
        # Track which models are available
        self.resnet_available = self._check_tensorflow_available()
        self.vit_available = self._check_torch_available()
        
        # Model usage stats
        self._model_usage = {'resnet': 0, 'vit': 0, 'both': 0}
        
        # Compact health scores (all your manual entries)
        self.health_scores = {
            # Very Healthy (8-10)
            'vegetables': 9, 'broccoli': 10, 'spinach': 10, 'kale': 10, 'beet': 9, 'beetroot': 9,
            'carrot': 9, 'tomato': 9, 'lettuce': 9, 'cucumber': 9, 'bell pepper': 9, 
            'zucchini': 9, 'cauliflower': 9, 'brussels sprouts': 9, 'asparagus': 9, 'celery': 9,
            'onion': 8, 'garlic': 9, 'ginger': 8, 'fruits': 8, 'fruit': 8, 'apple': 10, 'banana': 8, 
            'orange': 9, 'berries': 10, 'strawberry': 10, 'blueberry': 10, 'raspberry': 10, 
            'cherry': 10, 'strawberries': 10, 'blueberries': 10, 'raspberries': 10,
            'peach': 9, 'quince': 10, 'nectarine': 9, 'apricot': 9, 'medlar': 9, 'melon': 9, 'persimmon': 9,
            'watermelon': 9, 'pear': 8, 'grape': 8, 'pineapple': 8, 'mango': 8, 'avocado': 9,
            'salmon': 8, 'tuna': 8, 'sardines': 9, 'mackerel': 9, 'fish': 8, 'kohlrabi': 10, 
            'turnip': 9, 'artichoke': 10, 'salad': 9, 'arugula': 9, 'rocket salad': 9, 'rumex': 10, 
            'shrimp': 7, 'crab': 7, 'lobster': 7, 'mussels': 7, 'seafood': 7,
            'chicken breast': 8, 'turkey': 8, 'lean meat': 8, 'chicken': 7,
            'lentils': 9, 'chickpeas': 9, 'beans': 9, 'quinoa': 9, 'oatmeal': 9,
            'brown rice': 8, 'whole grain': 8, 'nuts': 8, 'almonds': 8, 'walnuts': 9,
            'greek yogurt': 8, 'cottage cheese': 8, 'white cheese': 8, 'tempeh': 8, 'edamame': 9, 'hummus': 8, 
            'seaweed': 9, 'herbs': 8, 'basil': 8, 'parsley': 8, 'cilantro': 8, 'dill': 9, 
            'lime': 8, 'lemon': 8, 'mushroom': 9, 'fungi': 9, 'peppers': 9,
            
            # Neutral (5-7)
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
            'flour': 5, 'wheat': 6, 'water': 10, 'broth': 6,
            'soy sauce': 5, 'salsa': 6, 'sauce': 5, 'marinara': 6, 'pesto': 6,
            'mozzarella': 6, 'parmesan': 6, 'ricotta': 6, 'cheddar': 5,
            'cream cheese': 5, 'sour cream': 5, 'mascarpone': 5, 'burrata': 5,
            'cocoa': 6, 'chocolate': 5, 'vanilla': 6, 'coffee': 6,
            'rice paper': 6, 'seitan': 7, 'chickpea': 9, 'milkshake': 5,
            'chocolate bar': 5, 'gelato': 5, 'pancakes': 5, 'pancake': 5,
            
            # Unhealthy (1-4)
            'pizza': 4, 'burger': 3, 'hamburger': 3, 'cheeseburger': 3,
            'french fries': 2, 'fries': 2, 'chips': 2, 'nachos': 3, 'feta': 4, 
            'hot dog': 3, 'corn dog': 2, 'fried chicken': 3, 'ketchup': 3, 
            'doughnut': 2, 'donut': 2, 'pastry': 3, 'croissant': 4, 'popcorn': 4,
            'cake': 3, 'cupcake': 2, 'brownie': 3, 'cookie': 4, 'cookies': 4,
            'candy': 1, 'soda': 1, 'energy drink': 1, 'sports drink': 2,
            'baking powder': 3, 'baking soda': 3, 'tofu': 3, 'mayo': 4, 'mayonnaise': 4,
            'bacon': 3, 'pepperoni': 2, 'salami': 2, 'hot wings': 3,
            'fried': 2, 'deep fried': 2, 'battered': 2, 'breaded': 3, 'breadcrumbs': 3,
            'onion rings': 2, 'mozzarella sticks': 3, 'cheese fries': 2,
            'mac and cheese': 4, 'alfredo': 3, 'carbonara': 4,
            'ramen': 4, 'instant noodles': 3, 'cup noodles': 3,
            'white bread': 4, 'white toast': 3, 'waffles': 4, 'waffle': 4, 'ice cream': 4,
            'syrup': 2, 'jam': 4, 'frosting': 2, 'whipped cream': 3
        }
        
        # Auto-score cache
        self._auto_score_cache = {}
        
        logger.info(f"Analyzer initialized. ResNet: {self.resnet_available}, ViT: {self.vit_available}")
    
    def _check_tensorflow_available(self):
        """Check if TensorFlow is available without loading it"""
        try:
            import tensorflow
            return True
        except ImportError:
            return False
    
    def _check_torch_available(self):
        """Check if PyTorch and transformers are available"""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    
    @property
    def resnet_model(self):
        """Lazy load ResNet model only when needed"""
        if self._resnet_model is None and self.resnet_available:
            load_tensorflow()
            from tensorflow.keras.applications import ResNet50
            
            with st.spinner('🔄 Loading ResNet model (one-time, ~30s)...'):
                self._resnet_model = ResNet50(
                    weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3)
                )
            logger.info("ResNet50 loaded")
        return self._resnet_model
    
    @property
    def vit_model(self):
        """Lazy load ViT model only when needed"""
        if self._vit_model is None and self.vit_available:
            load_torch()
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            with st.spinner('🔄 Loading ViT model (one-time, ~45s)...'):
                self._vit_processor = AutoImageProcessor.from_pretrained("nateraw/food")
                self._vit_model = AutoModelForImageClassification.from_pretrained("nateraw/food")
            logger.info("ViT loaded")
        return self._vit_model
    
    def predict_with_resnet(self, img):
        """Predict using ResNet50"""
        if not self.resnet_available:
            return []
        
        try:
            load_tensorflow()
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
            
            model = self.resnet_model
            if model is None:
                return []
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize(self.img_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array, verbose=0)
            decoded = decode_predictions(predictions, top=5)[0]
            
            results = []
            for _, label, confidence in decoded:
                results.append({
                    'name': label,
                    'confidence': float(confidence),
                    'source': 'ResNet50'
                })
            return results
        except Exception as e:
            logger.error(f"ResNet prediction failed: {e}")
            return []
    
    def predict_with_vit(self, img):
        """Predict using ViT"""
        if not self.vit_available:
            return []
        
        try:
            load_torch()
            import torch
            
            model = self.vit_model
            if model is None or self._vit_processor is None:
                return []
            
            inputs = self._vit_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = model.config.id2label[idx.item()]
                results.append({
                    'name': label,
                    'confidence': prob.item(),
                    'source': 'ViT'
                })
            return results
        except Exception as e:
            logger.error(f"ViT prediction failed: {e}")
            return []
    
    def predict_food(self, img):
        """Smart prediction - use best available model"""
        # Check for cached user corrections first
        features = self.extract_image_features(img)
        user_match = self.check_user_corrections(features)
        
        if user_match:
            st.info(f"🧠 Learned: {user_match['food']} ({user_match['similarity']:.0%})")
            return [{
                'name': user_match['food'],
                'confidence': 0.95,
                'source': 'Learning',
                'features': features.tolist()
            }]
        
        # Try models in order of availability and speed
        results = []
        
        if self.resnet_available:
            self._model_usage['resnet'] += 1
            results.extend(self.predict_with_resnet(img))
        
        if self.vit_available and len(results) == 0:
            # Only use ViT if ResNet failed
            self._model_usage['vit'] += 1
            results.extend(self.predict_with_vit(img))
        
        # Add features for learning
        for r in results:
            r['features'] = features.tolist()
        
        return results[:5] if results else []
    
    def extract_image_features(self, img):
        """Quick feature extraction for similarity matching"""
        try:
            img_resized = img.resize((64, 64))  # Small for speed
            img_array = np.array(img_resized)
            
            avg_colors = img_array.mean(axis=(0, 1))
            color_variance = img_array.std(axis=(0, 1))
            brightness = img_array.mean()
            
            return np.concatenate([avg_colors, color_variance, [brightness]])
        except:
            return np.zeros(7)
    
    def check_user_corrections(self, features):
        """Check if similar images were corrected"""
        if not self.user_corrections:
            return None
        
        try:
            current_features = np.array(features)
            best_match = None
            best_similarity = 0.0
            
            for correction in self.user_corrections:
                if 'features' in correction:
                    saved_features = np.array(correction['features'])
                    distance = np.linalg.norm(current_features - saved_features)
                    similarity = 1 / (1 + distance)
                    
                    if similarity > 0.85 and similarity > best_similarity:
                        best_match = {
                            'food': correction['correct_food'],
                            'similarity': similarity,
                            'count': correction.get('count', 1)
                        }
                        best_similarity = similarity
            
            return best_match
        except:
            return None
    
    def get_health_score(self, food_name: str) -> int:
        """Get health score - manual first, then auto-calculate"""
        food_lower = food_name.lower().strip()
        
        # 1. Exact match in manual
        if food_lower in self.health_scores:
            return self.health_scores[food_lower]
        
        # 2. Token matching
        for key, score in self.health_scores.items():
            if key in food_lower or food_lower in key:
                return score
        
        # 3. Session cache
        if food_lower in self._auto_score_cache:
            return self._auto_score_cache[food_lower]
        
        # 4. Auto-calculate from USDA
        try:
            nutrition = self.fetch_nutrition_data_cached(food_lower)
            if nutrition and nutrition.get('nutrients'):
                score = self.calculate_score_from_nutrients(nutrition['nutrients'])
                self._auto_score_cache[food_lower] = score
                return score
        except:
            pass
        
        return 6  # Neutral default
    
    def calculate_score_from_nutrients(self, nutrients: dict) -> int:
        """Calculate 1-10 score from USDA nutrients"""
        def get(fragment):
            for k, v in nutrients.items():
                if fragment.lower() in k.lower():
                    try:
                        return float(str(v).split()[0])
                    except:
                        pass
            return 0.0
        
        protein = get('protein')
        fiber = get('fiber')
        sugar = get('sugar')
        sat_fat = get('saturated')
        sodium = get('sodium')
        calories = get('energy') or get('calor')
        
        raw = 0
        if protein > 15: raw += 2
        elif protein > 5: raw += 1
        if fiber > 5: raw += 2
        elif fiber > 2: raw += 1
        if calories > 0 and calories < 100: raw += 1
        if calories > 600: raw -= 2
        elif calories > 400: raw -= 1
        if sat_fat > 10: raw -= 2
        elif sat_fat > 5: raw -= 1
        if sugar > 20: raw -= 2
        elif sugar > 10: raw -= 1
        if sodium > 600: raw -= 1
        
        raw = max(-4, min(6, raw))
        score = round(1 + (raw + 4) * 9 / 10)
        return max(1, min(10, score))
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def fetch_nutrition_data_cached(_self, food_name):
        """Fetch USDA nutrition data with 7-day cache"""
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
            data = retry_with_backoff(make_request, max_retries=3)
            if not data or not data.get('foods'):
                return None
            
            food_item = data['foods'][0]
            nutrients_dict = {}
            
            for nutrient in food_item.get('foodNutrients', []):
                name = nutrient.get('nutrientName', '')
                value = nutrient.get('value', 0)
                unit = nutrient.get('unitName', '')
                if name and value:
                    nutrients_dict[name] = f"{value} {unit}"
            
            return {
                'name': food_item.get('description', food_name),
                'nutrients': nutrients_dict
            }
        except:
            return None
    
    def add_user_correction(self, predicted, correct, features, confidence):
        """Save user correction"""
        try:
            import pickle
            
            correction = {
                'predicted': predicted,
                'correct_food': correct,
                'features': features,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'count': 1
            }
            
            # Check for existing similar correction
            found = False
            for existing in self.user_corrections:
                if existing['correct_food'].lower() == correct.lower():
                    if 'features' in existing:
                        saved = np.array(existing['features'])
                        current = np.array(features)
                        similarity = 1 / (1 + np.linalg.norm(saved - current))
                        if similarity > 0.85:
                            existing['count'] = existing.get('count', 1) + 1
                            existing['last_seen'] = datetime.now().isoformat()
                            found = True
                            break
            
            if not found:
                self.user_corrections.append(correction)
            
            self.feedback_log.append({
                'timestamp': datetime.now().isoformat(),
                'predicted': predicted,
                'correct': correct,
                'confidence': confidence,
                'was_correct': predicted.lower() == correct.lower()
            })
            
            safe_file_write(USER_CORRECTIONS_FILE, self.user_corrections, 'pickle')
            safe_file_write(FEEDBACK_FILE, self.feedback_log, 'json')
            
            return True
        except:
            return False
    
    def get_learning_stats(self):
        """Get learning statistics"""
        total = len(self.feedback_log)
        correct = sum(1 for log in self.feedback_log if log.get('was_correct', False))
        
        return {
            'total_feedback': total,
            'correct_predictions': correct,
            'accuracy': (correct / total * 100) if total > 0 else 0,
            'unique_foods_learned': len(self.user_corrections)
        }
    
    def get_health_category(self, score):
        """Convert score to category"""
        if score >= 8:
            return "Healthy", "🟢", "#28a745"
        elif score >= 5:
            return "Neutral", "🟡", "#ffc107"
        else:
            return "Unhealthy", "🔴", "#dc3545"
    
    def get_health_advice(self, score, food_name):
        """Generate health advice"""
        category, emoji, _ = self.get_health_category(score)
        
        if score >= 9:
            return f"{emoji} Excellent choice! {food_name.title()} is highly nutritious."
        elif score >= 8:
            return f"{emoji} Great option! {food_name.title()} is healthy."
        elif score >= 7:
            return f"{emoji} Good choice! {food_name.title()} is moderately healthy."
        elif score >= 6:
            return f"{emoji} Fair option. Balance {food_name.title()} with healthier foods."
        elif score >= 5:
            return f"🟡 Consume {food_name.title()} mindfully in moderation."
        elif score >= 3:
            return f"🟠 {food_name.title()} is best as an occasional treat."
        else:
            return f"🔴 Limit {food_name.title()} intake. Consider healthier alternatives."


# CLOUD OPTIMIZATION: Use session state for singleton
@st.cache_resource
def get_analyzer():
    """Get or create analyzer instance"""
    return CloudOptimizedFoodAnalyzer()


def main():
    # Compact title for cloud
    st.title("🍎 Food Health Analyzer")
    
    try:
        analyzer = get_analyzer()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        st.stop()
    
    # Compact sidebar
    with st.sidebar:
        st.header("📊 Stats")
        
        stats = analyzer.get_learning_stats()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Feedback", stats['total_feedback'])
        with col2:
            if stats['total_feedback'] > 0:
                st.metric("Accuracy", f"{stats['accuracy']:.0f}%")
        
        st.markdown("---")
        st.markdown("### 🎯 Score Guide")
        st.markdown("🟢 8-10: Healthy")
        st.markdown("🟡 5-7: Neutral")
        st.markdown("🔴 1-4: Limit")
        
        if st.button("🗑️ Clear Data"):
            import os
            if os.path.exists(USER_CORRECTIONS_FILE):
                os.remove(USER_CORRECTIONS_FILE)
            if os.path.exists(FEEDBACK_FILE):
                os.remove(FEEDBACK_FILE)
            st.success("Cleared!")
            time.sleep(1)
            st.rerun()
    
    # Main content - compact
    uploaded_file = st.file_uploader(
        "Upload food image",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to open image: {str(e)}")
            st.stop()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, use_container_width=True)
        
        with col2:
            # Analyze with progress
            with st.spinner("🔍 Analyzing..."):
                predictions = analyzer.predict_food(img)
            
            if not predictions:
                st.error("❌ Could not analyze image")
                st.stop()
            
            # Top prediction
            top = predictions[0]
            top_name = top['name'].replace('_', ' ').title()
            
            score = analyzer.get_health_score(top['name'])
            category, emoji, _ = analyzer.get_health_category(score)
            
            st.markdown(f"### {top_name}")
            st.progress(top['confidence'], text=f"{top['confidence']:.0%} confident")
            st.markdown(f"**Health:** {emoji} {score}/10 ({category})")
            
            # Store in session
            st.session_state.current_pred = predictions[0]
        
        st.markdown("---")
        
        # Feedback - compact
        st.subheader("Correct?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes", use_container_width=True):
                analyzer.add_user_correction(
                    top['name'],
                    top['name'],
                    st.session_state.current_pred['features'],
                    top['confidence']
                )
                st.success("Thanks!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("❌ No", use_container_width=True):
                st.session_state.show_form = True
        
        if st.session_state.get('show_form'):
            correct_name = st.text_input("Correct food name:")
            if st.button("Submit", disabled=not correct_name):
                if correct_name:
                    analyzer.add_user_correction(
                        top['name'],
                        correct_name,
                        st.session_state.current_pred['features'],
                        top['confidence']
                    )
                    st.success(f"Learned: {correct_name}")
                    st.session_state.show_form = False
                    time.sleep(1)
                    st.rerun()
        
        # Health advice - compact
        advice = analyzer.get_health_advice(score, top['name'])
        st.info(advice)
        
        # Nutrition data - optional
        with st.expander("📊 Nutrition Details"):
            with st.spinner("Fetching..."):
                nutrition = analyzer.fetch_nutrition_data_cached(top['name'])
            
            if nutrition:
                st.write(f"**{nutrition['name']}**")
                nutrients = nutrition['nutrients']
                
                # Key nutrients only
                key_nutrients = [
                    'Energy', 'Protein', 'Total lipid (fat)',
                    'Carbohydrate, by difference', 'Fiber, total dietary',
                    'Sugars, total including NLEA'
                ]
                
                for nutrient in key_nutrients:
                    if nutrient in nutrients:
                        st.write(f"• {nutrient}: {nutrients[nutrient]}")
            else:
                st.write("Nutrition data not available")
    
    else:
        st.info("👆 Upload an image to get started")
        
        with st.expander("💡 Tips"):
            st.markdown("""
            - Clear, well-lit photos work best
            - Center the food in frame
            - Simple backgrounds preferred
            - Single food items are easier to identify
            """)


if __name__ == "__main__":
    main()
