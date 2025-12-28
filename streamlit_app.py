"""
Food Health Analyzer
AI-powered food recognition with nutritional analysis
"""

import streamlit as st
import requests
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Food Health Analyzer",
    page_icon="🍎",
    layout="wide"
)

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Hugging Face Inference API
HF_API_TOKEN = os.environ.get('HF_TOKEN', None)
HF_API_URL = "https://api-inference.huggingface.co/models/nateraw/food"

@st.cache_data
def image_to_bytes(image):
    """Convert PIL Image to bytes"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def predict_food_hf_api(image):
    """Use HF Inference API for food classification"""
    try:
        headers = {}
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        
        img_bytes = image_to_bytes(image)
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=img_bytes,
            timeout=30
        )
        
        # Debug: Show what we got back
        st.write(f"Primary API Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            
            # Check if results is an error dict
            if isinstance(results, dict) and 'error' in results:
                st.warning(f"Model error: {results['error']}")
                if 'estimated_time' in results:
                    st.info(f"Model is loading, estimated time: {results['estimated_time']}s. Using fallback...")
                return predict_food_fallback(image)
            
            # Process valid results
            predictions = []
            for item in results[:5]:
                predictions.append({
                    'name': item['label'],
                    'confidence': item['score']
                })
            return predictions
        else:
            st.warning(f"Primary model returned status {response.status_code}, trying fallback...")
            return predict_food_fallback(image)
            
    except Exception as e:
        st.warning(f"Primary model error: {str(e)}, using fallback...")
        return predict_food_fallback(image)

def predict_food_fallback(image):
    """Fallback using alternative food model"""
    try:
        alt_url = "https://api-inference.huggingface.co/models/Kaludi/food-category-classification-v2.0"
        
        headers = {}
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        
        img_bytes = image_to_bytes(image)
        
        response = requests.post(
            alt_url,
            headers=headers,
            data=img_bytes,
            timeout=30
        )
        
        st.write(f"Fallback API Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            
            # Check if results is an error dict
            if isinstance(results, dict) and 'error' in results:
                st.warning(f"Fallback model error: {results['error']}")
                if 'estimated_time' in results:
                    st.info(f"Fallback model is loading, estimated time: {results['estimated_time']}s. Using general model...")
                return predict_food_general(image)
            
            predictions = []
            for item in results[:5]:
                predictions.append({
                    'name': item['label'],
                    'confidence': item['score']
                })
            return predictions
        else:
            st.warning(f"Fallback model returned status {response.status_code}, trying general model...")
            return predict_food_general(image)
            
    except Exception as e:
        st.warning(f"Fallback model error: {str(e)}, using general model...")
        return predict_food_general(image)

def predict_food_general(image):
    """Use general image classification as last resort"""
    try:
        general_url = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
        
        headers = {}
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        
        img_bytes = image_to_bytes(image)
        
        response = requests.post(
            general_url,
            headers=headers,
            data=img_bytes,
            timeout=30
        )
        
        st.write(f"General API Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            
            # Check if results is an error dict
            if isinstance(results, dict) and 'error' in results:
                st.error(f"All models unavailable: {results['error']}")
                if 'estimated_time' in results:
                    st.info(f"Model is loading. Please wait {results['estimated_time']}s and try again.")
                return [{'name': 'Model loading - please try again in a moment', 'confidence': 0.0}]
            
            predictions = []
            for item in results[:5]:
                predictions.append({
                    'name': item['label'],
                    'confidence': item['score']
                })
            return predictions
        else:
            st.error(f"All models returned errors. Status: {response.status_code}")
            st.info("The models may be loading. Please wait 30 seconds and try again.")
            return [{'name': 'Models loading - please retry', 'confidence': 0.0}]
            
    except Exception as e:
        st.error(f"Error accessing classification models: {str(e)}")
        return [{'name': 'Classification error', 'confidence': 0.0}]

def get_usda_nutrition(food_name):
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
        return None

def analyze_health(nutrients):
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

# Main app
def main():
    st.title("🍎 Food Health Analyzer")
    st.markdown("""
    Upload a photo of food to identify it and get nutritional information!
    
    **Features:**
    - 🤖 AI-powered food recognition
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
        This app uses AI to recognize food items and provides nutritional analysis 
        using data from the USDA FoodData Central database.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a food image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of food"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analyzing...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Predict food
                status_text.text("Identifying food...")
                progress_bar.progress(33)
                predictions = predict_food_hf_api(image)
                
                # Step 2: Get nutrition
                status_text.text("Fetching nutritional data...")
                progress_bar.progress(66)
                top_food = predictions[0]['name'].replace('_', ' ')
                nutrition_data = get_usda_nutrition(top_food)
                
                # Step 3: Analyze health
                status_text.text("Analyzing health rating...")
                progress_bar.progress(100)
                
                # Clear progress indicators
                status_text.empty()
                progress_bar.empty()
                
                st.success("Analysis complete!")
                
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
            health_rating, emoji = analyze_health(nutrients)
            
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
                ✅ This food appears to be a healthy choice! It contains beneficial nutrients 
                that contribute to a balanced diet.
                """)
            elif health_rating == "Moderate":
                st.markdown("""
                ⚠️ This food is okay in moderation. Be mindful of portion sizes and 
                try to balance it with other nutritious foods throughout the day.
                """)
            else:
                st.markdown("""
                ⚠️ This food may be high in fats, sugars, or sodium. Consider consuming 
                it occasionally and in small portions as part of a balanced diet.
                """)
        else:
            st.info("ℹ️ Nutritional information not available for this food item.")
    
    else:
        # Show example when no image is uploaded
        st.info("👆 Upload an image to get started!")

if __name__ == "__main__":
    main()
