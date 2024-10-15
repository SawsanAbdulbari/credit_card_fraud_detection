import sys
import os
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle

# Add project directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utils
from credit_fraud_utils_data import load_data, preprocess_data, resample_data
from credit_fraud_utils_eval import evaluate_model

def main(args):
    # Load data
    train_data, val_data = load_data(args.train_path, args.val_path)
    
    # Separate features and target
    X_train, y_train = preprocess_data(train_data)
    X_val, y_val = preprocess_data(val_data)
    
    # Resampling techniques
    resampling_methods = {
        'Original': (X_train, y_train),
        'RandomOverSampler': RandomOverSampler(random_state=42).fit_resample(X_train, y_train),
        'RandomUnderSampler': RandomUnderSampler(random_state=42).fit_resample(X_train, y_train),
        'SMOTE': SMOTE(random_state=42).fit_resample(X_train, y_train)
    }

    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42)
    }

    results = {}
    best_f1_score = 0
    best_model_info = None

    # Evaluate each model with each resampling technique
    for method, (X_res, y_res) in resampling_methods.items():
        for model_name, model in models.items():
            model, best_threshold, f1, pr_auc, roc_auc, report = evaluate_model(model, X_res, y_res, X_val, y_val)
            results[(method, model_name)] = {
                'Model': model,
                'Best Threshold': best_threshold,
                'F1-Score': f1,
                'PR-AUC': pr_auc,
                'ROC-AUC': roc_auc,
                'Classification Report': report
            }
            # Debugging print
            print(f"Method: {method}, Model: {model_name}, F1-Score: {f1}, PR-AUC: {pr_auc}, ROC-AUC: {roc_auc}")

            # Update the best model if this one is better
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_info = {
                    'model': model,
                    'best_threshold': best_threshold,
                    'method': method,
                    'model_name': model_name
                }


    # Evaluate Voting Classifier with each resampling technique
    for method, (X_res, y_res) in resampling_methods.items():
        voting_clf = VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(random_state=42))
        ], voting='soft')
        voting_clf, best_threshold, f1, pr_auc, roc_auc, report = evaluate_model(voting_clf, X_res, y_res, X_val, y_val)
        results[(method, 'VotingClassifier')] = {
            'Model': voting_clf,
            'Best Threshold': best_threshold,
            'F1-Score': f1,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'Classification Report': report
        }
        # Debugging print
        print(f"Method: {method}, Model: VotingClassifier, F1-Score: {f1}, PR-AUC: {pr_auc}, ROC-AUC: {roc_auc}")

        # Update the best model if this one is better
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_info = {
                'model': voting_clf,
                'best_threshold': best_threshold,
                'method': method,
                'model_name': 'VotingClassifier'
            }

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the evaluation results
    output_results_path = os.path.join('models', args.output_results)
    with open(output_results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_results_path}")

    # Save the best model separately
    best_model_path = os.path.join('models', 'new_best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_info, f)
    print(f"Best model ({best_model_info['model_name']} with {best_model_info['method']}) saved to {best_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models to detect credit card fraud and evaluate them.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the validation data CSV file')
    parser.add_argument('--output_results', type=str, required=True, help='Filename to save the evaluation results')
    args = parser.parse_args()
    main(args)