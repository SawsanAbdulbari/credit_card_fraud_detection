import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report, roc_auc_score

def evaluate_model(model, X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_val_scaled)[:,1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, best_threshold, f1, pr_auc, roc_auc, classification_report(y_val, y_pred)

def evaluate_voting_classifier(model, X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_val_scaled)[:,1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    return model, best_threshold, f1, pr_auc, roc_auc, classification_report(y_val, y_pred)
