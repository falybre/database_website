import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import joblib
import sys
import os

# Add the project root to path to import modules
sys.path.append(os.path.abspath('.'))

# Import both model approaches
from improved_model_approach import load_and_preprocess, prepare_features
import joblib

def load_original_model():
    """Load the original model and related files"""
    try:
        model = joblib.load('traffic_incident_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError:
        print("Original model files not found. Make sure to run the original model first.")
        return None, None

def load_improved_model():
    """Load the improved model and related files"""
    try:
        model = joblib.load('improved_traffic_incident_model.pkl')
        label_encoder = joblib.load('improved_label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError:
        print("Improved model files not found. Make sure to run the improved model first.")
        return None, None

def compare_models(data_path='data_mmda_traffic_spatial.csv'):
    """Compare the performance of the original and improved models"""
    # Load data
    data = load_and_preprocess(data_path)
    
    # Prepare features
    X, y, label_encoder, categorical_features, numeric_features = prepare_features(data)
    
    # Load models
    orig_model, orig_encoder = load_original_model()
    impr_model, impr_encoder = load_improved_model()
    
    if orig_model is None or impr_model is None:
        print("One or both models could not be loaded. Exiting comparison.")
        return
    
    # Split data for consistent testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Make predictions
    orig_preds = orig_model.predict(X_test)
    impr_preds = impr_model.predict(X_test)
    
    # Calculate accuracies
    orig_accuracy = accuracy_score(y_test, orig_preds)
    impr_accuracy = accuracy_score(y_test, impr_preds)
    
    print(f"Original Model Accuracy: {orig_accuracy:.4f}")
    print(f"Improved Model Accuracy: {impr_accuracy:.4f}")
    print(f"Accuracy Improvement: {(impr_accuracy - orig_accuracy) * 100:.2f}%")
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    models = ['Original Model', 'Improved Model']
    accuracies = [orig_accuracy, impr_accuracy]
    
    bar_plot = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    
    # Add exact values on top of bars
    for bar in bar_plot:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    
    # Generate classification reports
    class_names = label_encoder.classes_
    
    print("\nOriginal Model Classification Report:")
    print(classification_report(y_test, orig_preds, target_names=class_names))
    
    print("\nImproved Model Classification Report:")
    print(classification_report(y_test, impr_preds, target_names=class_names))
    
    # Compare confusion matrices
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    cm_orig = confusion_matrix(y_test, orig_preds)
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Original Model Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.subplot(1, 2, 2)
    cm_impr = confusion_matrix(y_test, impr_preds)
    sns.heatmap(cm_impr, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Improved Model Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png')
    
    # Per-class performance improvement
    orig_report = classification_report(y_test, orig_preds, target_names=class_names, output_dict=True)
    impr_report = classification_report(y_test, impr_preds, target_names=class_names, output_dict=True)
    
    # Create dataframe for comparison
    comparison_data = []
    for cls in class_names:
        orig_f1 = orig_report[cls]['f1-score']
        impr_f1 = impr_report[cls]['f1-score']
        improvement = (impr_f1 - orig_f1) * 100  # percentage points
        
        comparison_data.append({
            'Class': cls,
            'Original F1': orig_f1,
            'Improved F1': impr_f1,
            'Improvement (%)': improvement
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nPer-class F1 Score Improvement:")
    print(comparison_df)
    
    # Plot per-class improvement
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['Original F1'], width, label='Original Model', color='#1f77b4')
    plt.bar(x + width/2, comparison_df['Improved F1'], width, label='Improved Model', color='#ff7f0e')
    
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison by Class')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('f1_score_comparison.png')
    
    # Return comparison summary
    return {
        'original_accuracy': orig_accuracy,
        'improved_accuracy': impr_accuracy,
        'improvement': impr_accuracy - orig_accuracy,
        'original_report': orig_report,
        'improved_report': impr_report,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    compare_models()
