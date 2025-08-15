from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import base64

app = Flask(__name__)

# Candidate captions (same as your original list)
CANDIDATE_CAPTIONS = [
    "Trees, Travel and Tea!",
    "A refreshing beverage.",
    "A moment of indulgence.",
    "The perfect thirst quencher.",
    "Your daily dose of delight.",
    "Taste the tradition.",
    "Savor the flavor.",
    "Refresh and rejuvenate.",
    "Unwind and enjoy.",
    "The taste of home.",
    "A treat for your senses.",
    "A taste of adventure.",
    "A moment of bliss.",
    "Your travel companion.",
    "Fuel for your journey.",
    "The essence of nature.",
    "The warmth of comfort.",
    "A sip of happiness.",
    "Pure indulgence.",
    "Quench your thirst, ignite your spirit.",
    "Awaken your senses, embrace the moment.",
    "The taste of faraway lands.",
    "A taste of home, wherever you are.",
    "Your daily dose of delight.",
    "Your moment of serenity.",
    "The perfect pick-me-up.",
    "The perfect way to unwind.",
    "Taste the difference.",
    "Experience the difference.",
    "A refreshing escape.",
    "A delightful escape.",
    "The taste of tradition, the spirit of adventure.",
    "The warmth of home, the joy of discovery.",
    "Your passport to flavor.",
    "Your ticket to tranquility.",
    "Sip, savor, and explore.",
    "Indulge, relax, and rejuvenate.",
    "The taste of wanderlust.",
    "The comfort of home.",
    "A journey for your taste buds.",
    "A haven for your senses.",
    "Your refreshing companion.",
    "Your delightful escape.",
    "Taste the world, one sip at a time.",
    "Embrace the moment, one cup at a time.",
    "The essence of exploration.",
    "The comfort of connection.",
    "Quench your thirst for adventure.",
    "Savor the moment of peace.",
    "The taste of discovery.",
    "The warmth of belonging.",
    "Your travel companion, your daily delight.",
    "Your moment of peace, your daily indulgence.",
    "The spirit of exploration, the comfort of home.",
    "The joy of discovery, the warmth of connection.",
    "Sip, savor, and set off on an adventure.",
    "Indulge, relax, and find your peace.",
    "A delightful beverage.",
    "A moment of relaxation.",
    "The perfect way to start your day.",
    "The perfect way to end your day.",
    "A treat for yourself.",
    "Something to savor.",
    "A moment of calm.",
    "A taste of something special.",
    "A refreshing pick-me-up.",
    "A comforting drink.",
    "A taste of adventure.",
    "A moment of peace.",
    "A small indulgence.",
    "A daily ritual.",
    "A way to connect with others.",
    "A way to connect with yourself.",
    "A taste of home.",
    "A taste of something new.",
    "A moment to enjoy.",
    "A moment to remember."
]

# Global variables to store loaded models (so they don't reload every time)
processor = None
model = None

def load_models():
    """Load CLIP model and processor (only once)"""
    global processor, model
    if processor is None or model is None:
        print("Loading CLIP model... This might take a moment.")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print("Model loaded successfully!")

def process_image(image):
    """Process the uploaded image"""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image if it's too large (speeds up processing)
    max_size = 512
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Process image for CLIP
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def get_image_features(inputs):
    """Get image features from CLIP model"""
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def match_captions(image_features, captions):
    """Find the best matching captions for the image"""
    # Process text
    text_inputs = processor(text=captions, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    
    # Convert to numpy for similarity calculation
    image_features_np = image_features.detach().cpu().numpy()
    text_features_np = text_features.detach().cpu().numpy()
    
    # Calculate similarities
    similarities = cosine_similarity(image_features_np, text_features_np)
    
    # Get best matches (sorted from highest to lowest)
    best_indices = similarities.argsort(axis=1)[0][::-1]
    best_captions = [captions[i] for i in best_indices]
    best_scores = similarities[0][best_indices].tolist()
    
    return best_captions, best_scores

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Handle image analysis API endpoint"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load models if not already loaded
        load_models()
        
        # Open and process the image
        image = Image.open(file.stream)
        inputs = process_image(image)
        
        # Get image features
        image_features = get_image_features(inputs)
        
        # Match captions
        best_captions, similarities = match_captions(image_features, CANDIDATE_CAPTIONS)
        
        # Return top 5 results
        results = []
        for i in range(min(5, len(best_captions))):
            results.append({
                'rank': i + 1,
                'caption': best_captions[i],
                'confidence': float(similarities[i])
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting AI Image Caption Matcher...")
    print("📝 Loading AI models on first request...")
    print("🌐 Open your browser to: http://localhost:5000")
    print("⏰ First analysis will take 2-3 minutes (downloading AI model)")
    print("⚡ Subsequent analyses will be much faster!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)