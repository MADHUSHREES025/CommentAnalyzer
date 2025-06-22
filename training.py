"""
🎭 Emotion Classification Model Training Script

This script implements a deep learning-based emotion classification system using a hybrid
BiLSTM-CNN architecture. It can classify text into 8 different emotion categories:
Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral, and Love.

Author: Your Name
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import pickle
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, 
    Dense, Dropout, Concatenate, BatchNormalization,
    SpatialDropout1D, GlobalAveragePooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ML utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Download NLTK data if needed
def download_nltk_data():
    """Download required NLTK datasets"""
    datasets = ['punkt', 'stopwords']
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
        except LookupError:
            print(f"Downloading {dataset}...")
            nltk.download(dataset)

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class EmotionClassifier:
    """
    A neural network-based emotion classifier that combines BiLSTM and CNN architectures
    for analyzing emotional content in text.
    """
    
    def __init__(self, vocab_size=10000, sequence_length=100, embedding_dim=128):
        """
        Initialize the emotion classifier
        
        Args:
            vocab_size: Maximum number of words to keep in vocabulary
            sequence_length: Maximum length of input sequences
            embedding_dim: Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        
        # Will be initialized during training
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.training_history = None
        
        # Mapping for emotion standardization
        self.emotion_mapping = {
            'joy': 'happy', 'happiness': 'happy', 'enthusiasm': 'happy', 'relief': 'happy', 'fun': 'happy',
            'sadness': 'sad', 'empty': 'sad',
            'anger': 'angry', 'hate': 'angry',
            'fear': 'fear', 'worry': 'fear',
            'surprise': 'surprise',
            'disgust': 'disgust',
            'neutral': 'neutral', 'boredom': 'neutral',
            'love': 'love',
            'curious to dive deeper': 'neutral'  # For topical chat dataset
        }
    
    def load_dataset(self, file_path=None):
        """
        Load emotion dataset from CSV file or create sample data
        
        Args:
            file_path: Path to CSV file with 'text' and 'emotion' columns
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        if file_path and Path(file_path).exists():
            print(f"Loading dataset from {file_path}")
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'message': 'text', 'sentiment': 'emotion',
                'label': 'emotion', 'category': 'emotion'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure we have the required columns
            if 'text' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("Dataset must have 'text' and 'emotion' columns")
            
            # Standardize emotions
            df['emotion'] = df['emotion'].str.lower().str.strip()
            df['emotion'] = df['emotion'].map(self.emotion_mapping).fillna(df['emotion'])
            
            # Remove invalid entries
            df = df.dropna(subset=['text', 'emotion'])
            df = df[df['text'].str.len() > 0]
            
        else:
            print("Creating sample dataset for demonstration...")
            sample_data = {
                'text': [
                    "I'm absolutely thrilled about this opportunity!",
                    "Feeling really down and melancholy today",
                    "This situation is making me furious and irritated",
                    "I'm genuinely worried about what might happen next",
                    "What a shocking and unexpected turn of events!",
                    "This is absolutely revolting and disgusting",
                    "Just a regular day, nothing particularly noteworthy",
                    "I absolutely adore this, it brings me so much joy",
                    "The weather is nice today, feeling content",
                    "I'm devastated by this terrible news"
                ],
                'emotion': [
                    'happy', 'sad', 'angry', 'fear', 'surprise', 
                    'disgust', 'neutral', 'love', 'neutral', 'sad'
                ]
            }
            df = pd.DataFrame(sample_data)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
        
        return df
    
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs and mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Keep only alphabetic characters and basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word not in stop_words and len(word) > 1])
        except:
            # Fallback if NLTK fails
            pass
        
        return text
    
    def prepare_training_data(self, df):
        """
        Prepare dataset for model training
        
        Args:
            df: DataFrame with 'text' and 'emotion' columns
            
        Returns:
            tuple: (X, y, processed_df)
        """
        print("Preprocessing text data...")
        
        # Clean text
        df['text'] = df['text'].apply(self.clean_text)
        df = df[df['text'].str.len() > 0]  # Remove empty texts
        
        # Initialize and fit tokenizer
        print("Creating text tokenizer...")
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<UNK>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(df['text'])
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=self.sequence_length, padding='post', truncating='post')
        
        # Encode emotion labels
        print("Encoding emotion labels...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(df['emotion'])
        y = to_categorical(y_encoded)
        
        print(f"Processed {len(X)} samples")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Number of emotion classes: {len(self.label_encoder.classes_)}")
        print(f"Emotion classes: {list(self.label_encoder.classes_)}")
        
        return X, y, df
    
    def build_model(self, num_classes):
        """
        Build hybrid BiLSTM + CNN model architecture
        
        Args:
            num_classes: Number of emotion classes
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        print("Building neural network architecture...")
        
        text_input = Input(shape=(self.sequence_length,), name='text_input')
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.sequence_length,
            mask_zero=True,
            name='embedding'
        )(text_input)
        
        # BiLSTM branch
        lstm_out = Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm'
        )(embedding)
        lstm_pooled = GlobalMaxPooling1D(name='lstm_pooling')(lstm_out)
        
        # CNN branch with multiple filter sizes
        conv_outputs = []
        for filter_size in [3, 4, 5]:
            conv = Conv1D(
                filters=64,
                kernel_size=filter_size,
                activation='relu',
                padding='same',
                name=f'conv_{filter_size}'
            )(embedding)
            conv = BatchNormalization()(conv)
            conv_pooled = GlobalMaxPooling1D()(conv)
            conv_outputs.append(conv_pooled)
        
        # Combine all features
        if len(conv_outputs) > 1:
            combined_features = Concatenate(name='feature_fusion')([lstm_pooled] + conv_outputs)
        else:
            combined_features = Concatenate(name='feature_fusion')([lstm_pooled, conv_outputs[0]])
        
        # Classification layers
        dense = Dense(128, activation='relu', name='dense_1')(combined_features)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        
        dense = Dense(64, activation='relu', name='dense_2')(dense)
        dense = Dropout(0.3)(dense)
        
        # Output layer
        output = Dense(num_classes, activation='softmax', name='emotion_output')(dense)
        
        model = Model(inputs=text_input, outputs=output, name='emotion_classifier')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=30, batch_size=64):
        """
        Train the emotion classification model
        
        Args:
            X: Input features (tokenized sequences)
            y: Target labels (one-hot encoded)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            tuple: (X_val, y_val) validation data
        """
        print("Starting model training...")
        
        # Calculate class weights for balanced training
        y_integers = np.argmax(y, axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y_integers
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        num_classes = y.shape[1]
        self.model = self.build_model(num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        return X_val, y_val
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate model performance on validation data
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            float: Validation accuracy
        """
        print("\nEvaluating model performance...")
        
        # Make predictions
        predictions = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Create confusion matrix visualization
        self._plot_confusion_matrix(y_true_classes, y_pred_classes)
        
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        return accuracy
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Emotion Classification', fontsize=16, pad=20)
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Visualize training progress"""
        if self.training_history is None:
            print("No training history available to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot accuracy
        epochs = range(1, len(self.training_history.history['accuracy']) + 1)
        ax1.plot(epochs, self.training_history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, self.training_history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(epochs, self.training_history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, self.training_history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_emotion(self, text):
        """
        Predict emotion for a single text
        
        Args:
            text: Input text string
            
        Returns:
            tuple: (predicted_emotion, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.sequence_length)
        
        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Get emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return emotion, confidence
    
    def save_model(self, model_dir='emotion_model'):
        """
        Save the complete trained model and preprocessing components
        
        Args:
            model_dir: Directory to save model files
        """
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        print(f"Saving model to {model_path}")
        
        # Save the neural network model
        self.model.save(model_path / 'emotion_model.h5')
        
        # Save preprocessing components
        with open(model_path / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(model_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save configuration
        config = {
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'emotion_classes': list(self.label_encoder.classes_)
        }
        with open(model_path / 'config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print("Model saved successfully!")
        print(f"Files created in {model_path}:")
        for file in model_path.glob('*'):
            print(f"  - {file.name}")
    
    def test_predictions(self, test_sentences=None):
        """
        Test the model with sample sentences
        
        Args:
            test_sentences: List of test sentences (optional)
        """
        if test_sentences is None:
            test_sentences = [
                "I'm absolutely thrilled about this amazing opportunity!",
                "Feeling really depressed and hopeless about everything",
                "This situation is making me incredibly angry and frustrated",
                "I'm terrified about what might happen in the future",
                "Wow, I never saw that coming! What a surprise!",
                "This food tastes absolutely disgusting and revolting",
                "It's just another ordinary day, nothing special happening",
                "I love spending time with my family, they mean everything to me"
            ]
        
        print("\n" + "="*60)
        print("TESTING MODEL PREDICTIONS")
        print("="*60)
        
        for i, sentence in enumerate(test_sentences, 1):
            emotion, confidence = self.predict_emotion(sentence)
            print(f"\n{i}. Text: \"{sentence}\"")
            print(f"   Predicted Emotion: {emotion.upper()}")
            print(f"   Confidence: {confidence:.3f}")
            print("-" * 50)

def main():
    """Main function to train the emotion classification model"""
    print("🎭 EMOTION CLASSIFICATION MODEL TRAINER")
    print("="*50)
    
    # Enable GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU acceleration enabled")
        except RuntimeError as e:
            print(f"⚠️  GPU setup failed: {e}")
    else:
        print("💻 Running on CPU")
    
    # Initialize the classifier
    classifier = EmotionClassifier(
        vocab_size=8000,
        sequence_length=80,
        embedding_dim=100
    )
    
    # Load and prepare data
    try:
        # Try to load from file first
        dataset = classifier.load_dataset('topical_chat.csv')
    except:
        # Fall back to sample data
        print("Could not load topical_chat.csv, using sample data")
        dataset = classifier.load_dataset()
    
    # Prepare training data
    X, y, processed_df = classifier.prepare_training_data(dataset)
    
    # Train the model
    X_val, y_val = classifier.train(
        X, y,
        validation_split=0.2,
        epochs=25,
        batch_size=64
    )
    
    # Evaluate performance
    final_accuracy = classifier.evaluate(X_val, y_val)
    
    # Visualize training progress
    classifier.plot_training_history()
    
    # Save the trained model
    classifier.save_model()
    
    # Test with sample sentences
    classifier.test_predictions()
    
    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Final Accuracy: {final_accuracy:.1%}")
    print("\n📁 Generated Files:")
    print("   - emotion_model/ (complete saved model)")
    print("   - confusion_matrix.png")
    print("   - training_history.png")

if __name__ == "__main__":
    # Set random seeds for reproducible results
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the training pipeline
    main()