"""Model evaluation functionality for SDM models."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

def evaluate_model_cv(
    model: Any,
    occurrence_gdf: pd.DataFrame,
    feature_columns: List[str],
    n_cv_folds: int = 3,
    random_state: int = 42
) -> Tuple[List[Any], np.ndarray]:
    """Evaluate a model using k-fold cross-validation.
    
    Args:
        model: The model to evaluate
        occurrence_gdf: DataFrame containing occurrence data
        feature_columns: List of feature column names
        n_cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        Tuple containing:
        - List of trained models from each fold
        - Array of AUC scores from each fold
    """
    kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    cv_models = []
    cv_scores = np.zeros(n_cv_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(occurrence_gdf)):
        # Split data
        X_train = occurrence_gdf.iloc[train_idx][feature_columns]
        y_train = occurrence_gdf.iloc[train_idx]["class"]
        X_val = occurrence_gdf.iloc[val_idx][feature_columns]
        y_val = occurrence_gdf.iloc[val_idx]["class"]
        
        # Train model
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train, y_train)
        cv_models.append(model_copy)
        
        # Evaluate
        y_pred = model_copy.predict_proba(X_val)[:, 1]
        cv_scores[fold_idx] = roc_auc_score(y_val, y_pred)
        
        logger.info(f"Fold {fold_idx + 1}/{n_cv_folds} AUC: {cv_scores[fold_idx]:.4f}")
    
    return cv_models, cv_scores

def evaluate_model_performance(
    model: Any,
    test_data: pd.DataFrame,
    feature_columns: List[str]
) -> Dict[str, float]:
    """Evaluate model performance on test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        feature_columns: List of feature column names
        
    Returns:
        Dictionary containing performance metrics
    """
    X_test = test_data[feature_columns]
    y_test = test_data["class"]
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate confusion matrix metrics
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        "auc": auc_score,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score
    }

def save_evaluation_results(
    results: Dict[str, Any],
    output_dir: Path,
    species_name: str,
    activity_type: str
) -> Path:
    """Save model evaluation results to a CSV file.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save results
        species_name: Name of the species
        activity_type: Type of activity
        
    Returns:
        Path to the saved results file
    """
    # Create results directory if it doesn't exist
    results_dir = output_dir / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results DataFrame
    results_df = pd.DataFrame([results])
    
    # Add metadata
    results_df["species"] = species_name
    results_df["activity_type"] = activity_type
    results_df["timestamp"] = pd.Timestamp.now()
    
    # Save to CSV
    output_file = results_dir / f"{species_name}_{activity_type}_evaluation.csv"
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved evaluation results to: {output_file}")
    return output_file
