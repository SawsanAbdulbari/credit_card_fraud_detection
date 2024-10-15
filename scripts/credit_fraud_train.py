import sys
import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from tabulate import tabulate

# Add project directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utils
from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model

def main(args):
    # Load data
    train_data, val_data = load_data(args.train_path, args.val_path)
    
    # Separate features and target
    X_train, y_train = preprocess_data(train_data)
    X_val, y_val = preprocess_data(val_data)
    
    # Train the best model (Random Forest Classifier)
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(random_state=42)
    
    # Evaluate the model
    model, best_threshold, f1, pr_auc, roc_auc, report = evaluate_model(rf_model, X_train, y_train, X_val, y_val)
    
    # Prepare results for display
    results = [
        ["Model", "Random Forest Classifier"],
        ["Best Threshold", f"{best_threshold:.4f}"],
        ["F1-Score", f"{f1:.4f}"],
        ["PR-AUC", f"{pr_auc:.4f}"],
        ["ROC-AUC", f"{roc_auc:.4f}"]
    ]
    
    # Display results
    print("\nModel Evaluation Results:")
    print(tabulate(results, headers="firstrow", tablefmt="grid"))
    
    print("\nClassification Report:")
    print(report)
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    best_model_path = os.path.join('models', 'best_random_forest_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nBest model saved to {best_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the best model (Random Forest Classifier) for credit card fraud detection.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the validation data CSV file')
    args = parser.parse_args()
    main(args)
