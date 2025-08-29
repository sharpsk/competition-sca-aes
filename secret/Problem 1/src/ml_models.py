import numpy as np
import time
from typing import Dict, Tuple, Optional
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try importing neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from src.aes_utils import AESHelper

class MLKeyRecovery:
    """Machine Learning-based key recovery for side channel analysis."""
    
    def __init__(self, algorithm: str = "neural_network"):
        """
        Initialize ML key recovery.
        
        Args:
            algorithm: ML algorithm to use ("neural_network", "svm", "random_forest")
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.aes_helper = AESHelper()
        
    def train_and_predict(self, traces: np.ndarray, plaintexts: Optional[np.ndarray], 
                         key_byte_position: int, train_ratio: float = 0.8) -> Dict:
        """
        Train ML model and predict key byte.
        
        Args:
            traces: Power traces array
            plaintexts: Plaintext inputs
            key_byte_position: Target key byte position
            train_ratio: Ratio of data to use for training
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        if plaintexts is None:
            plaintexts = np.random.randint(0, 256, size=(len(traces), 16), dtype=np.uint8)
        
        # For supervised learning, we need to generate labels
        # In practice, these would come from known attacks or be recovered incrementally
        labels = self._generate_synthetic_labels(traces, plaintexts, key_byte_position)
        
        # Prepare features
        features = self._extract_features(traces)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, train_size=train_ratio, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.algorithm == "neural_network":
            results = self._train_neural_network(X_train_scaled, X_test_scaled, y_train, y_test)
        elif self.algorithm == "svm":
            results = self._train_svm(X_train_scaled, X_test_scaled, y_train, y_test)
        elif self.algorithm == "random_forest":
            results = self._train_random_forest(X_train, X_test, y_train, y_test)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        training_time = time.time() - start_time
        results['training_time'] = training_time
        results['key_byte_position'] = key_byte_position
        
        return results
    
    def _extract_features(self, traces: np.ndarray) -> np.ndarray:
        """Extract features from power traces."""
        features = []
        
        # Statistical features
        features.append(np.mean(traces, axis=1))
        features.append(np.std(traces, axis=1))
        features.append(np.max(traces, axis=1))
        features.append(np.min(traces, axis=1))
        features.append(np.var(traces, axis=1))
        
        # Peak detection features
        for trace in traces:
            peaks = self._find_peaks(trace)
            features.append([len(peaks), np.mean(peaks) if peaks else 0, 
                           np.std(peaks) if len(peaks) > 1 else 0])
        
        # Combine all features
        stat_features = np.column_stack(features[:-1])
        peak_features = np.array(features[-1])
        
        # Add raw trace samples (subsampled for efficiency)
        subsample_rate = max(1, traces.shape[1] // 1000)  # Limit to ~1000 samples
        raw_features = traces[:, ::subsample_rate]
        
        return np.column_stack([stat_features, peak_features, raw_features])
    
    def _find_peaks(self, trace: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Simple peak detection in trace."""
        peaks = []
        threshold_val = np.max(trace) * threshold
        
        for i in range(1, len(trace) - 1):
            if trace[i] > trace[i-1] and trace[i] > trace[i+1] and trace[i] > threshold_val:
                peaks.append(trace[i])
        
        return np.array(peaks)
    
    def _generate_synthetic_labels(self, traces: np.ndarray, plaintexts: np.ndarray, 
                                 key_byte_position: int) -> np.ndarray:
        """
        Generate synthetic labels for supervised learning.
        In practice, these would come from known key bytes or previous analysis.
        """
        # Generate a random key byte as ground truth
        true_key_byte = np.random.randint(0, 256)
        
        # Create labels based on S-box output Hamming weight
        labels = []
        for plaintext in plaintexts:
            sbox_out = self.aes_helper.sbox_lookup([plaintext[key_byte_position] ^ true_key_byte])[0]
            hamming_weight = bin(sbox_out).count('1')
            labels.append(hamming_weight)
        
        return np.array(labels)
    
    def _train_neural_network(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train neural network model."""
        if not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow not available, falling back to Random Forest")
            return self._train_random_forest(X_train, X_test, y_train, y_test)
        
        # Create model
        model = Sequential([
            Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(9, activation='softmax')  # 9 classes for Hamming weight (0-8)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Predict key byte (simplified approach)
        # In practice, this would involve more sophisticated key ranking
        predicted_key = self._predict_key_from_hw(y_pred, X_test.shape[0])
        confidence = np.max(model.predict(X_test).mean(axis=0)) * 100
        
        self.model = model
        
        return {
            'predicted_key': predicted_key,
            'accuracy': accuracy * 100,
            'confidence': confidence,
            'model_type': 'Neural Network',
            'training_history': history.history
        }
    
    def _train_svm(self, X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train SVM model."""
        # Use a subset of features to avoid memory issues
        n_features = min(X_train.shape[1], 1000)
        X_train_subset = X_train[:, :n_features]
        X_test_subset = X_test[:, :n_features]
        
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_subset, y_train)
        
        y_pred = model.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get prediction probabilities
        y_prob = model.predict_proba(X_test_subset)
        confidence = np.max(y_prob.mean(axis=0)) * 100
        
        predicted_key = self._predict_key_from_hw(y_pred, len(y_test))
        
        self.model = model
        
        return {
            'predicted_key': predicted_key,
            'accuracy': accuracy * 100,
            'confidence': confidence,
            'model_type': 'SVM'
        }
    
    def _train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train Random Forest model."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get prediction probabilities
        y_prob = model.predict_proba(X_test)
        confidence = np.max(y_prob.mean(axis=0)) * 100
        
        predicted_key = self._predict_key_from_hw(y_pred, len(y_test))
        
        # Feature importance
        feature_importance = model.feature_importances_
        
        self.model = model
        
        return {
            'predicted_key': predicted_key,
            'accuracy': accuracy * 100,
            'confidence': confidence,
            'model_type': 'Random Forest',
            'feature_importance': feature_importance
        }
    
    def _predict_key_from_hw(self, hamming_weights: np.ndarray, num_samples: int) -> int:
        """
        Convert predicted Hamming weights back to key byte guess.
        This is a simplified approach for demonstration.
        """
        # Most common predicted Hamming weight
        most_common_hw = np.bincount(hamming_weights).argmax()
        
        # Generate a plausible key byte with that Hamming weight
        for key_guess in range(256):
            if bin(key_guess).count('1') == most_common_hw:
                return key_guess
        
        return most_common_hw * 32  # Fallback
    
    def ensemble_prediction(self, traces: np.ndarray, plaintexts: np.ndarray, 
                          key_byte_position: int) -> Dict:
        """
        Use ensemble of multiple ML models for prediction.
        """
        algorithms = ["random_forest", "svm"]
        if TENSORFLOW_AVAILABLE:
            algorithms.append("neural_network")
        
        predictions = []
        confidences = []
        accuracies = []
        
        for algo in algorithms:
            self.algorithm = algo
            result = self.train_and_predict(traces, plaintexts, key_byte_position)
            predictions.append(result['predicted_key'])
            confidences.append(result['confidence'])
            accuracies.append(result['accuracy'])
        
        # Ensemble voting
        final_prediction = max(set(predictions), key=predictions.count)
        final_confidence = np.mean(confidences)
        final_accuracy = np.mean(accuracies)
        
        return {
            'predicted_key': final_prediction,
            'confidence': final_confidence,
            'accuracy': final_accuracy,
            'model_type': 'Ensemble',
            'individual_predictions': predictions,
            'individual_confidences': confidences
        }

if PYTORCH_AVAILABLE:
    class PyTorchMLP(nn.Module):
        """PyTorch Multi-Layer Perceptron for key recovery."""
        
        def __init__(self, input_dim: int, num_classes: int = 9):
            super(PyTorchMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            return self.layers(x)
