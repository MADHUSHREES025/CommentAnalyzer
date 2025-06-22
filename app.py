import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EmotionClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.emotion_colors = {
            'happy': '#4CAF50',      # Green
            'sad': '#2196F3',        # Blue
            'angry': '#F44336',      # Red
            'fear': '#FF9800',       # Orange
            'surprise': '#9C27B0',   # Purple
            'disgust': '#795548',    # Brown
            'neutral': '#607D8B',    # Blue Grey
            'love': '#E91E63'        # Pink
        }
        
    def load_models(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Load model
            self.model = load_model('best_emotion_model.h5')
            
            # Load tokenizer
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load label encoder
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load model configuration
            with open('model_config.pkl', 'rb') as f:
                self.config = pickle.load(f)
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def predict_emotion(self, text):
        """Predict emotion for a single text"""
        if not text.strip():
            return 'neutral', 0.0
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text.strip():
            return 'neutral', 0.0
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.config['max_len'])
        
        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        # Get emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return emotion, confidence
    
    def classify_sentences(self, text):
        """Classify emotions for each sentence in the text"""
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        results = []
        for sentence in sentences:
            if sentence.strip():
                emotion, confidence = self.predict_emotion(sentence)
                results.append({
                    'sentence': sentence,
                    'emotion': emotion,
                    'confidence': confidence,
                    'color': self.emotion_colors.get(emotion, '#000000')
                })
        
        return results

def create_colored_html(results):
    """Create HTML with colored sentences"""
    html_content = ""
    for result in results:
        html_content += f"""
        <span style="color: {result['color']}; font-weight: bold; margin: 5px 0; display: inline-block;">
            {result['sentence']}
        </span>
        <span style="color: #666; font-size: 0.8em; margin-left: 10px;">
            ({result['emotion']} - {result['confidence']:.2f})
        </span>
        <br><br>
        """
    return html_content

def main():
    st.set_page_config(
        page_title="Emotion Text Classifier",
        page_icon="😊",
        layout="wide"
    )
    
    # Title and description
    st.title("🎭 Emotion Text Classifier")
    st.markdown("---")
    st.markdown("""
    This app analyzes the emotional content of your text by classifying each sentence individually.
    Each sentence will be colored according to its detected emotion.
    """)
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = EmotionClassifier()
        
    # Load models
    with st.spinner("Loading emotion classification models..."):
        if not st.session_state.classifier.model:
            model_loaded = st.session_state.classifier.load_models()
            if not model_loaded:
                st.error("⚠️ Could not load the trained models. Please ensure the following files exist:")
                st.code("""
                - emotion_model.h5
                - tokenizer.pkl
                - label_encoder.pkl
                - model_config.pkl
                """)
                st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        
        # Text input
        user_text = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Type or paste your text here. The app will analyze each sentence and color it based on the detected emotion..."
        )
        
        # Classification button
        classify_button = st.button("🔍 Classify Emotions", type="primary", use_container_width=True)
        
        # Sample texts
        st.subheader("📋 Try Sample Texts")
        sample_texts = {
            "Mixed Emotions": "I am so happy about my promotion today! However, I'm also scared about the new responsibilities. The workload makes me a bit angry sometimes, but I love the challenge.",
            "Happy Story": "What a wonderful day! The sun is shining brightly. I feel so grateful for all the good things in my life. This ice cream tastes amazing!",
            "Sad Story": "I lost my best friend yesterday. The house feels so empty without him. I can't stop crying when I think about all our memories together.",
            "Fear and Surprise": "I heard a strange noise downstairs. My heart is pounding with fear. Wait, it's just my cat! What a surprise, I thought someone broke in."
        }
        
        for name, text in sample_texts.items():
            if st.button(f"📖 {name}", use_container_width=True):
                user_text = text
                st.rerun()
    
    with col2:
        st.subheader("🎨 Emotion Analysis Results")
        
        if classify_button and user_text:
            with st.spinner("Analyzing emotions..."):
                # Classify sentences
                results = st.session_state.classifier.classify_sentences(user_text)
                
                if results:
                    # Display colored results
                    st.markdown("### Colored Text by Emotion:")
                    colored_html = create_colored_html(results)
                    st.markdown(colored_html, unsafe_allow_html=True)
                    
                    # Display emotion legend
                    st.markdown("### 🎨 Emotion Color Legend:")
                    legend_cols = st.columns(4)
                    emotions = list(st.session_state.classifier.emotion_colors.items())
                    
                    for i, (emotion, color) in enumerate(emotions):
                        with legend_cols[i % 4]:
                            st.markdown(f'<span style="color: {color}; font-weight: bold;">● {emotion.title()}</span>', unsafe_allow_html=True)
                    
                    # Display detailed results
                    st.markdown("### 📊 Detailed Analysis:")
                    df = pd.DataFrame([
                        {
                            'Sentence': result['sentence'][:50] + "..." if len(result['sentence']) > 50 else result['sentence'],
                            'Emotion': result['emotion'].title(),
                            'Confidence': f"{result['confidence']:.3f}"
                        }
                        for result in results
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Emotion distribution
                    st.markdown("### 📈 Emotion Distribution:")
                    emotion_counts = pd.Series([r['emotion'] for r in results]).value_counts()
                    st.bar_chart(emotion_counts)
                    
                else:
                    st.warning("⚠️ No sentences could be analyzed. Please check your input text.")
        
        elif not user_text and classify_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. **Enter Text**: Type or paste your text in the input area
    2. **Classify**: Click the "Classify Emotions" button
    3. **View Results**: See each sentence colored by its detected emotion
    4. **Analyze**: Check the detailed analysis and emotion distribution
    
    ### 🎯 Supported Emotions:
    - **Happy** (Green): Joy, happiness, excitement
    - **Sad** (Blue): Sadness, depression, sorrow
    - **Angry** (Red): Anger, frustration, rage
    - **Fear** (Orange): Fear, worry, anxiety
    - **Surprise** (Purple): Surprise, amazement
    - **Disgust** (Brown): Disgust, revulsion
    - **Neutral** (Grey): Neutral, calm
    - **Love** (Pink): Love, affection, care
    """)

if __name__ == "__main__":
    main()