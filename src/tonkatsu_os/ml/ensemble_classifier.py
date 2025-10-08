"""
Machine Learning Models for Raman Spectral Classification

This module provides ensemble ML models for molecular identification from Raman spectra,
including Random Forest, SVM, and Neural Network classifiers with confidence scoring.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """
    Ensemble classifier combining multiple ML models for robust
    molecular identification from Raman spectra.
    """

    def __init__(self, use_pca: bool = True, n_components: int = 50, use_pls: bool = True):
        """
        Initialize ensemble classifier.

        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
            use_pls: Whether to include PLS regression in ensemble (NIPALS algorithm)
        """
        self.use_pca = use_pca
        self.use_pls = use_pls
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.label_encoder = LabelEncoder()

        # Initialize individual models
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        self.svm_model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)

        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=500,
            random_state=42,
        )

        # PLS model (NIPALS algorithm) - excellent for spectroscopic data with multicollinearity
        self.pls_model = None
        if use_pls:
            self.pls_model = OneVsRestClassifier(
                PLSRegression(
                    n_components=min(15, n_components),  # Typically fewer components needed for PLS
                    scale=False,  # We handle scaling separately
                )
            )

        # Ensemble model
        self.ensemble_model = None
        self.is_trained = False
        self.class_names = None

    def preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features with scaling and optional PCA.

        Args:
            X: Feature matrix
            fit: Whether to fit the preprocessors

        Returns:
            Preprocessed features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            if self.use_pca:
                X_processed = self.pca.fit_transform(X_scaled)
            else:
                X_processed = X_scaled
        else:
            X_scaled = self.scaler.transform(X)
            if self.use_pca:
                X_processed = self.pca.transform(X_scaled)
            else:
                X_processed = X_scaled

        return X_processed

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Train the ensemble model with cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            validation_split: Fraction of data for validation

        Returns:
            Training results dictionary
        """
        logger.info("Starting ensemble model training...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded
        )

        # Preprocess features
        X_train_processed = self.preprocess_features(X_train, fit=True)
        X_val_processed = self.preprocess_features(X_val, fit=False)

        # Train individual models
        logger.info("Training Random Forest...")
        self.rf_model.fit(X_train_processed, y_train)

        logger.info("Training SVM...")
        self.svm_model.fit(X_train_processed, y_train)

        logger.info("Training Neural Network...")
        self.nn_model.fit(X_train_processed, y_train)

        if self.use_pls:
            logger.info("Training PLS Regression (NIPALS algorithm)...")
            self.pls_model.fit(X_train_processed, y_train)

        # Create calibrated ensemble
        logger.info("Creating ensemble model...")
        estimators = [("rf", self.rf_model), ("svm", self.svm_model), ("nn", self.nn_model)]

        if self.use_pls:
            estimators.append(("pls", self.pls_model))

        self.ensemble_model = VotingClassifier(estimators=estimators, voting="soft")

        # Calibrate for better probability estimates
        class_counts = np.bincount(y_train)
        min_class_samples = np.min(class_counts)
        if min_class_samples >= 2:
            cv = min(3, int(min_class_samples))
            self.ensemble_model = CalibratedClassifierCV(self.ensemble_model, method="isotonic", cv=cv)
            self.ensemble_model.fit(X_train_processed, y_train)
        else:
            logger.warning(
                "Skipping ensemble calibration due to limited samples per class (min=%s)",
                min_class_samples,
            )
            self.ensemble_model.fit(X_train_processed, y_train)

        self.is_trained = True

        # Evaluate on validation set
        val_predictions = self.ensemble_model.predict(X_val_processed)
        val_probabilities = self.ensemble_model.predict_proba(X_val_processed)

        # Calculate individual model accuracies
        rf_acc = accuracy_score(y_val, self.rf_model.predict(X_val_processed))
        svm_acc = accuracy_score(y_val, self.svm_model.predict(X_val_processed))
        nn_acc = accuracy_score(y_val, self.nn_model.predict(X_val_processed))
        ensemble_acc = accuracy_score(y_val, val_predictions)

        results = {
            "rf_accuracy": rf_acc,
            "svm_accuracy": svm_acc,
            "nn_accuracy": nn_acc,
            "ensemble_accuracy": ensemble_acc,
            "classification_report": classification_report(
                y_val, val_predictions, target_names=self.class_names, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_val, val_predictions).tolist(),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "n_features": X_train_processed.shape[1],
            "n_classes": len(self.class_names),
        }

        if self.use_pls:
            pls_acc = accuracy_score(y_val, self.pls_model.predict(X_val_processed))
            results["pls_accuracy"] = pls_acc

        logger.info(f"Training completed. Ensemble accuracy: {ensemble_acc:.3f}")
        return results

    def predict(self, X: np.ndarray) -> Dict:
        """
        Predict compound from spectrum features.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_processed = self.preprocess_features(X, fit=False)

        # Get predictions from ensemble
        ensemble_pred = self.ensemble_model.predict(X_processed)
        ensemble_proba = self.ensemble_model.predict_proba(X_processed)

        # Get individual model predictions
        rf_pred = self.rf_model.predict(X_processed)
        rf_proba = self.rf_model.predict_proba(X_processed)

        svm_pred = self.svm_model.predict(X_processed)
        svm_proba = self.svm_model.predict_proba(X_processed)

        nn_pred = self.nn_model.predict(X_processed)
        nn_proba = self.nn_model.predict_proba(X_processed)

        pls_pred = None
        pls_proba = None
        if self.use_pls:
            pls_pred = self.pls_model.predict(X_processed)
            pls_proba = self.pls_model.predict_proba(X_processed)

        results = []
        for i in range(len(X)):
            # Ensemble prediction
            pred_class_idx = ensemble_pred[i]
            pred_class = self.class_names[pred_class_idx]
            confidence = np.max(ensemble_proba[i])

            # Top 3 predictions
            top_indices = np.argsort(ensemble_proba[i])[-3:][::-1]
            top_predictions = [
                {"compound": self.class_names[idx], "probability": ensemble_proba[i][idx]}
                for idx in top_indices
            ]

            # Model agreement
            individual_preds = [rf_pred[i], svm_pred[i], nn_pred[i]]
            if self.use_pls:
                individual_preds.append(pls_pred[i])
            agreement = np.mean(np.array(individual_preds) == pred_class_idx)

            # Uncertainty quantification
            entropy = -np.sum(ensemble_proba[i] * np.log(ensemble_proba[i] + 1e-10))
            max_entropy = np.log(len(self.class_names))
            uncertainty = entropy / max_entropy

            result = {
                "predicted_compound": pred_class,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "model_agreement": agreement,
                "top_predictions": top_predictions,
                "individual_predictions": {
                    "random_forest": {
                        "compound": self.class_names[rf_pred[i]],
                        "confidence": np.max(rf_proba[i]),
                    },
                    "svm": {
                        "compound": self.class_names[svm_pred[i]],
                        "confidence": np.max(svm_proba[i]),
                    },
                    "neural_network": {
                        "compound": self.class_names[nn_pred[i]],
                        "confidence": np.max(nn_proba[i]),
                    },
                },
            }

            # Add PLS predictions if enabled
            if self.use_pls:
                result["individual_predictions"]["pls_regression"] = {
                    "compound": self.class_names[pls_pred[i]],
                    "confidence": np.max(pls_proba[i]),
                }

            results.append(result)

        return results

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize hyperparameters using grid search.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Best parameters for each model
        """
        logger.info("Starting hyperparameter optimization...")

        y_encoded = self.label_encoder.fit_transform(y)
        X_processed = self.preprocess_features(X, fit=True)

        best_params = {}

        # Random Forest hyperparameter tuning
        rf_param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        rf_grid.fit(X_processed, y_encoded)
        best_params["random_forest"] = rf_grid.best_params_

        # SVM hyperparameter tuning
        svm_param_grid = {
            "C": [1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01],
            "kernel": ["rbf", "poly"],
        }

        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=42),
            svm_param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        svm_grid.fit(X_processed, y_encoded)
        best_params["svm"] = svm_grid.best_params_

        # Neural Network hyperparameter tuning
        nn_param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01, 0.1],
        }

        nn_grid = GridSearchCV(
            MLPClassifier(random_state=42, max_iter=500),
            nn_param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        nn_grid.fit(X_processed, y_encoded)
        best_params["neural_network"] = nn_grid.best_params_

        logger.info("Hyperparameter optimization completed")
        return best_params

    def evaluate_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation evaluation.

        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds

        Returns:
            Cross-validation results
        """
        y_encoded = self.label_encoder.fit_transform(y)
        X_processed = self.preprocess_features(X, fit=True)

        # Individual model CV scores
        rf_scores = cross_val_score(self.rf_model, X_processed, y_encoded, cv=cv_folds)
        svm_scores = cross_val_score(self.svm_model, X_processed, y_encoded, cv=cv_folds)
        nn_scores = cross_val_score(self.nn_model, X_processed, y_encoded, cv=cv_folds)

        # Ensemble CV scores
        ensemble = VotingClassifier(
            [("rf", self.rf_model), ("svm", self.svm_model), ("nn", self.nn_model)], voting="soft"
        )

        ensemble_scores = cross_val_score(ensemble, X_processed, y_encoded, cv=cv_folds)

        return {
            "random_forest": {
                "mean_score": np.mean(rf_scores),
                "std_score": np.std(rf_scores),
                "scores": rf_scores.tolist(),
            },
            "svm": {
                "mean_score": np.mean(svm_scores),
                "std_score": np.std(svm_scores),
                "scores": svm_scores.tolist(),
            },
            "neural_network": {
                "mean_score": np.mean(nn_scores),
                "std_score": np.std(nn_scores),
                "scores": nn_scores.tolist(),
            },
            "ensemble": {
                "mean_score": np.mean(ensemble_scores),
                "std_score": np.std(ensemble_scores),
                "scores": ensemble_scores.tolist(),
            },
        }

    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "ensemble_model": self.ensemble_model,
            "scaler": self.scaler,
            "pca": self.pca,
            "label_encoder": self.label_encoder,
            "class_names": self.class_names,
            "use_pca": self.use_pca,
            "n_components": self.n_components,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.ensemble_model = model_data["ensemble_model"]
        self.scaler = model_data["scaler"]
        self.pca = model_data["pca"]
        self.label_encoder = model_data["label_encoder"]
        self.class_names = model_data["class_names"]
        self.use_pca = model_data["use_pca"]
        self.n_components = model_data["n_components"]
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")


class ConfidenceScorer:
    """
    Advanced confidence scoring system for molecular identification predictions.
    """

    def __init__(self):
        self.calibration_data = None

    def calculate_confidence_score(
        self,
        probabilities: np.ndarray,
        peak_match_score: float,
        model_agreement: float,
        spectral_quality: float = 1.0,
    ) -> Dict:
        """
        Calculate comprehensive confidence score.

        Args:
            probabilities: Model prediction probabilities
            peak_match_score: Score from peak matching algorithm
            model_agreement: Agreement between different models
            spectral_quality: Quality of the input spectrum

        Returns:
            Confidence analysis dictionary
        """
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        normalized_entropy = entropy / max_entropy

        # Weighted confidence score
        confidence_components = {
            "probability_score": max_prob,
            "entropy_score": 1 - normalized_entropy,
            "peak_match_score": peak_match_score,
            "model_agreement_score": model_agreement,
            "spectral_quality_score": spectral_quality,
        }

        # Weighted combination
        weights = {
            "probability_score": 0.3,
            "entropy_score": 0.2,
            "peak_match_score": 0.25,
            "model_agreement_score": 0.15,
            "spectral_quality_score": 0.1,
        }

        overall_confidence = sum(
            weights[key] * score for key, score in confidence_components.items()
        )

        # Risk assessment
        risk_level = self._assess_risk(overall_confidence, confidence_components)

        return {
            "overall_confidence": overall_confidence,
            "confidence_components": confidence_components,
            "risk_level": risk_level,
            "recommendation": self._get_recommendation(overall_confidence, risk_level),
        }

    def _assess_risk(self, overall_confidence: float, components: Dict) -> str:
        """Assess the risk level of the prediction."""
        if overall_confidence > 0.8 and components["model_agreement_score"] > 0.8:
            return "low"
        elif overall_confidence > 0.6 and components["peak_match_score"] > 0.5:
            return "medium"
        else:
            return "high"

    def _get_recommendation(self, confidence: float, risk_level: str) -> str:
        """Provide recommendation based on confidence and risk."""
        if risk_level == "low":
            return "High confidence identification. Results can be trusted."
        elif risk_level == "medium":
            return "Moderate confidence. Consider additional validation."
        else:
            return "Low confidence. Manual verification strongly recommended."
