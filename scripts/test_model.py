import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, classification_report
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_on_test_data(model, test_data_path):
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    # Scale features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:,1]
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate predictions
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    
    # Print evaluation results
    print(f"F1-Score: {f1}")
    print(f"PR-AUC: {pr_auc}")
    print(f"ROC-AUC: {roc_auc}")
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on test data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data CSV file')
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    evaluate_on_test_data(model, args.test_data_path)
