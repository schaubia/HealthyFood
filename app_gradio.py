"""
Food Health Analyzer with Training Capability
A web application that analyzes food images, provides nutritional information,
and allows you to train the model on your own images
"""

import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import requests
import json
import os
import shutil
from pathlib import Path
import pickle

# USDA API configuration
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Training configuration
TRAINING_DATA_DIR = "training_data"
CUSTOM_MODEL_PATH = "models/custom_food_model.h5"
CLASS_MAPPING_PATH = "models/class_mapping.pkl"

class FoodHealthAnalyzer:
    def __init__(self):
        """Initialize the food analyzer with ResNet50 model"""
        self.img_size = (224, 224)
        self.model = self.build_model()
        self.custom_model = None
        self.class_mapping = {}
        
        # Create necessary directories
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Load custom model if available
        self.load_custom_model()
        
    def build_model(self):
        """Build base model using ResNet50 with ImageNet weights"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        return base_model
    
    def build_custom_model(self, num_classes):
        """Build custom model for fine-tuning"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_custom_model(self):
        """Load previously trained custom model if available"""
        try:
            if os.path.exists(CUSTOM_MODEL_PATH) and os.path.exists(CLASS_MAPPING_PATH):
                self.custom_model = load_model(CUSTOM_MODEL_PATH)
                with open(CLASS_MAPPING_PATH, 'rb') as f:
                    self.class_mapping = pickle.load(f)
                print(f"✓ Loaded custom model with {len(self.class_mapping)} classes")
                return True
        except Exception as e:
            print(f"Could not load custom model: {e}")
        return False
    
    def save_custom_model(self):
        """Save custom model and class mapping"""
        if self.custom_model is not None:
            save_model(self.custom_model, CUSTOM_MODEL_PATH)
            with open(CLASS_MAPPING_PATH, 'wb') as f:
                pickle.dump(self.class_mapping, f)
            print("✓ Custom model saved successfully")
    
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img = img.resize(self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict_food(self, img, top_k=5, use_custom=True):
        """Predict food item from image"""
        img_array = self.preprocess_image(img)
        
        results = []
        
        # Try custom model first if available and requested
        if use_custom and self.custom_model is not None:
            custom_predictions = self.custom_model.predict(img_array, verbose=0)[0]
            
            # Get top predictions from custom model
            top_indices = np.argsort(custom_predictions)[-top_k:][::-1]
            
            for idx in top_indices:
                class_name = self.class_mapping.get(idx, f"Class_{idx}")
                confidence = float(custom_predictions[idx])
                results.append({
                    'name': class_name,
                    'confidence': confidence,
                    'source': 'custom'
                })
        
        # Also get predictions from base model
        base_predictions = self.model.predict(img_array, verbose=0)
        decoded = decode_predictions(base_predictions, top=top_k)[0]
        
        for _, label, confidence in decoded:
            results.append({
                'name': label.replace('_', ' '),
                'confidence': float(confidence),
                'source': 'imagenet'
            })
        
        # Sort by confidence and return top_k
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:top_k]
    
    def add_training_image(self, img, label):
        """Add an image to the training dataset"""
        # Create class directory
        class_dir = os.path.join(TRAINING_DATA_DIR, label.lower().replace(' ', '_'))
        os.makedirs(class_dir, exist_ok=True)
        
        # Count existing images in this class
        existing_images = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Save image
        img_path = os.path.join(class_dir, f"{label}_{existing_images + 1}.jpg")
        img.save(img_path)
        
        return img_path, existing_images + 1
    
    def get_training_stats(self):
        """Get statistics about training data"""
        stats = {}
        total_images = 0
        
        if not os.path.exists(TRAINING_DATA_DIR):
            return stats, total_images
        
        for class_name in os.listdir(TRAINING_DATA_DIR):
            class_path = os.path.join(TRAINING_DATA_DIR, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) 
                                if f.endswith(('.jpg', '.png', '.jpeg'))])
                if num_images > 0:
                    stats[class_name] = num_images
                    total_images += num_images
        
        return stats, total_images
    
    def train_model(self, epochs=10, batch_size=16, validation_split=0.2):
        """Train the custom model on collected images"""
        # Check if we have training data
        stats, total_images = self.get_training_stats()
        
        if total_images < 10:
            return False, "Need at least 10 images total to start training"
        
        if len(stats) < 2:
            return False, "Need at least 2 different food classes to train"
        
        try:
            # Create data generators
            datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                validation_split=validation_split
            )
            
            train_generator = datagen.flow_from_directory(
                TRAINING_DATA_DIR,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            
            validation_generator = datagen.flow_from_directory(
                TRAINING_DATA_DIR,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
            
            # Build or update model
            num_classes = len(train_generator.class_indices)
            self.custom_model = self.build_custom_model(num_classes)
            
            # Store class mapping
            self.class_mapping = {v: k.replace('_', ' ').title() 
                                for k, v in train_generator.class_indices.items()}
            
            # Train model
            history = self.custom_model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                verbose=1
            )
            
            # Save model
            self.save_custom_model()
            
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
            
            return True, f"Training completed! Final accuracy: {final_acc:.2%}, Validation accuracy: {final_val_acc:.2%}"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
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
            'Protein': 1,
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

# Initialize analyzer
analyzer = FoodHealthAnalyzer()

def analyze_food_image(img, use_custom_model):
    """Main function to analyze food image"""
    if img is None:
        return "Please upload an image", "", ""
    
    predictions = analyzer.predict_food(img, use_custom=use_custom_model)
    
    pred_text = "### 🍽️ Food Recognition Results:\n\n"
    for i, pred in enumerate(predictions, 1):
        confidence_pct = pred['confidence'] * 100
        source_badge = "🎯 Custom" if pred['source'] == 'custom' else "🌐 ImageNet"
        pred_text += f"{i}. **{pred['name'].title()}** - {confidence_pct:.2f}% ({source_badge})\n"
    
    top_food = predictions[0]['name'].replace('_', ' ')
    nutrition_data = analyzer.get_usda_nutrition(top_food)
    
    if nutrition_data:
        nutrients = nutrition_data['nutrients']
        health_rating, emoji = analyzer.analyze_health(nutrients)
        
        nutrition_text = f"### 📊 Nutritional Information\n\n"
        nutrition_text += f"**Food:** {nutrition_data['name']}\n\n"
        nutrition_text += f"**Health Rating:** {emoji} {health_rating}\n\n"
        nutrition_text += "**Key Nutrients (per 100g):**\n\n"
        
        priority = [
            'Energy', 'Protein', 'Total lipid (fat)',
            'Carbohydrate, by difference', 'Fiber, total dietary',
            'Sugars, total including NLEA', 'Sodium, Na', 'Cholesterol'
        ]
        
        displayed = set()
        for nutrient in priority:
            if nutrient in nutrients:
                nutrition_text += f"• **{nutrient}:** {nutrients[nutrient]}\n"
                displayed.add(nutrient)
        
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

def add_to_training(img, food_label):
    """Add image to training dataset"""
    if img is None:
        return "⚠️ Please upload an image first"
    
    if not food_label or food_label.strip() == "":
        return "⚠️ Please enter a food label"
    
    try:
        img_path, count = analyzer.add_training_image(img, food_label.strip())
        return f"✅ Image added successfully! This is image #{count} for '{food_label}'\n\nSaved to: {img_path}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def show_training_stats():
    """Display training dataset statistics"""
    stats, total = analyzer.get_training_stats()
    
    if total == 0:
        return "📊 No training images yet. Add some images to get started!"
    
    stats_text = f"### 📊 Training Dataset Statistics\n\n"
    stats_text += f"**Total Images:** {total}\n"
    stats_text += f"**Total Classes:** {len(stats)}\n\n"
    stats_text += "**Images per class:**\n\n"
    
    for class_name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        stats_text += f"• **{class_name.replace('_', ' ').title()}:** {count} images\n"
    
    if total < 10:
        stats_text += f"\n⚠️ Need at least 10 images total to train (currently have {total})"
    elif len(stats) < 2:
        stats_text += f"\n⚠️ Need at least 2 different classes to train (currently have {len(stats)})"
    else:
        stats_text += f"\n✅ Ready to train!"
    
    return stats_text

def train_custom_model(epochs, batch_size):
    """Train the custom model"""
    success, message = analyzer.train_model(epochs=int(epochs), batch_size=int(batch_size))
    
    if success:
        return f"✅ {message}\n\nYou can now use the custom model for predictions!"
    else:
        return f"❌ {message}"

# Create Gradio interface
with gr.Blocks(title="Food Health Analyzer with Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 Food Health Analyzer with Custom Training
    
    **Two modes:**
    1. **Analyze**: Upload food photos to identify them and get nutritional information
    2. **Train**: Add your own images to improve recognition accuracy
    
    This app uses AI to recognize food items and provides:
    - Food identification with confidence scores
    - Nutritional information from USDA database
    - Health rating based on nutritional content
    - **Custom model training on your own images!**
    """)
    
    with gr.Tabs():
        # Analysis Tab
        with gr.Tab("🔍 Analyze Food"):
            with gr.Row():
                with gr.Column():
                    analyze_image = gr.Image(type="pil", label="Upload Food Image")
                    use_custom = gr.Checkbox(
                        label="Use Custom Trained Model (if available)",
                        value=True,
                        info="Check to use your trained model alongside ImageNet"
                    )
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
            
            analyze_btn.click(
                fn=analyze_food_image,
                inputs=[analyze_image, use_custom],
                outputs=[predictions_output, nutrition_output, health_output]
            )
        
        # Training Tab
        with gr.Tab("🎓 Train Custom Model"):
            gr.Markdown("""
            ### 📚 Build Your Custom Food Recognizer
            
            Train the model to recognize your specific foods! This is especially useful for:
            - Local or regional dishes not in ImageNet
            - Specific brands or preparations
            - Foods you eat frequently
            - Improving accuracy on foods you care about
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Add Training Images")
                    training_image = gr.Image(type="pil", label="Upload Training Image")
                    food_label_input = gr.Textbox(
                        label="Food Label",
                        placeholder="e.g., 'Homemade Pizza' or 'Red Apple'",
                        info="Enter the name of the food in this image"
                    )
                    add_btn = gr.Button("➕ Add to Training Set", variant="secondary")
                    add_status = gr.Markdown()
                    
                    gr.Markdown("""
                    **Guidelines:**
                    - Add at least 5-10 images per food type
                    - Use varied angles and lighting
                    - Keep labels consistent (same spelling/capitalization)
                    - More images = better accuracy
                    """)
                
                with gr.Column():
                    gr.Markdown("#### Step 2: View Training Stats")
                    stats_btn = gr.Button("📊 Show Training Stats")
                    stats_output = gr.Markdown()
                    
                    gr.Markdown("#### Step 3: Train Model")
                    epochs_input = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        label="Training Epochs",
                        info="More epochs = longer training, potentially better results"
                    )
                    batch_size_input = gr.Slider(
                        minimum=8,
                        maximum=32,
                        value=16,
                        step=8,
                        label="Batch Size"
                    )
                    train_btn = gr.Button("🚀 Start Training", variant="primary")
                    train_status = gr.Markdown()
                    
                    gr.Markdown("""
                    **Note:** Training may take several minutes depending on:
                    - Number of images
                    - Number of classes
                    - Number of epochs
                    - Your computer's processing power
                    """)
            
            add_btn.click(
                fn=add_to_training,
                inputs=[training_image, food_label_input],
                outputs=add_status
            )
            
            stats_btn.click(
                fn=show_training_stats,
                outputs=stats_output
            )
            
            train_btn.click(
                fn=train_custom_model,
                inputs=[epochs_input, batch_size_input],
                outputs=train_status
            )
        
        # Help Tab
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## 📖 How to Use This App
            
            ### Analyzing Food
            1. Go to the **Analyze Food** tab
            2. Upload an image of food
            3. Click **Analyze Food**
            4. View the results: food identification, nutrition info, and health rating
            
            ### Training Your Custom Model
            
            #### Why Train?
            The base model knows 1000 ImageNet categories, but might not recognize:
            - Your specific recipes or preparations
            - Regional/local foods
            - Branded products
            - Specific varieties (e.g., "Fuji Apple" vs just "Apple")
            
            #### Training Steps:
            
            **1. Collect Images (5-10+ per food type)**
            - Take photos from different angles
            - Use various lighting conditions
            - Show different portions/servings
            - Keep the food centered and clear
            
            **2. Add to Training Set**
            - Upload each image
            - Enter a consistent label (e.g., "Margherita Pizza")
            - Click "Add to Training Set"
            - Repeat for all images
            
            **3. Check Your Stats**
            - Click "Show Training Stats"
            - Verify you have enough images
            - Make sure labels are correct
            
            **4. Train the Model**
            - Set training epochs (10-20 recommended for start)
            - Click "Start Training"
            - Wait for training to complete
            - Check accuracy results
            
            **5. Use Your Model**
            - Go back to Analyze tab
            - Make sure "Use Custom Trained Model" is checked
            - Upload your food images
            - See improved recognition!
            
            ### Tips for Best Results
            
            **For Training:**
            - Minimum 5 images per class (10+ recommended)
            - At least 2 different food types
            - Consistent labeling (same name for same food)
            - Varied but clear photos
            - Similar to how you'll use it (same camera, lighting, etc.)
            
            **For Analysis:**
            - Clear, well-lit photos
            - Food centered in frame
            - Minimal background clutter
            - One food item at a time
            
            ### Understanding Results
            
            **Confidence Scores:**
            - 80%+ = Very confident
            - 50-80% = Fairly confident
            - Below 50% = Uncertain (might be wrong)
            
            **Sources:**
            - 🎯 Custom = From your trained model
            - 🌐 ImageNet = From base model
            
            **Health Ratings:**
            - 🟢 Healthy = Nutrient-rich, low in harmful components
            - 🟡 Moderate = Balanced, okay in moderation
            - 🔴 Unhealthy = High in fats/sugars/sodium
            
            ### Troubleshooting
            
            **"Need at least 10 images to train"**
            - Add more images to your training set
            - Aim for 5-10 per food type
            
            **"Training failed"**
            - Check if you have enough variety in images
            - Make sure images are valid (JPG, PNG)
            - Try with fewer epochs first
            
            **Low accuracy**
            - Add more training images
            - Use more varied photos
            - Train for more epochs
            - Make sure photos are clear and focused
            
            **Custom model not working**
            - Check "Use Custom Trained Model" is checked
            - Make sure training completed successfully
            - Verify model file exists in models/folder
            
            ### Model Files Location
            - Training images: `training_data/[food_name]/`
            - Custom model: `models/custom_food_model.h5`
            - Class mapping: `models/class_mapping.pkl`
            
            ### API Limits
            - USDA API (DEMO_KEY): 1000 requests/hour
            - Get free API key at: https://fdc.nal.usda.gov/api-key-signup.html
            """)

if __name__ == "__main__":
    print("🍎 Food Health Analyzer with Training")
    print("=" * 50)
    
    # Show training stats on startup
    stats, total = analyzer.get_training_stats()
    if total > 0:
        print(f"📊 Training data: {total} images across {len(stats)} classes")
        if analyzer.custom_model is not None:
            print(f"✓ Custom model loaded with {len(analyzer.class_mapping)} classes")
    else:
        print("ℹ️  No training data yet. Use the Train tab to add images!")
    
    print("=" * 50)
    
    demo.launch(share=False)
