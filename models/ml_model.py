import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class MLModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with validation if provided"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=eval_set,
                verbose=100,
                early_stopping_rounds=20
            )
        else:
            self.model.fit(X_train_scaled, y_train)
            
        # Save feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1': f1_score(y, predictions)
        }
        
        return metrics
    
    def save_model(self, path='models/saved'):
        """Save the model and scaler"""
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join(path, f'xgboost_model_{timestamp}.joblib')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, f'scaler_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature importance
        importance_path = os.path.join(path, f'feature_importance_{timestamp}.csv')
        self.feature_importance.to_csv(importance_path, index=False)
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'importance_path': importance_path
        }
    
    def load_model(self, model_path, scaler_path):
        """Load a saved model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def get_top_features(self, n=10):
        """Get top n most important features"""
        return self.feature_importance.head(n)
    
    def cross_validate(self, X, y, n_splits=5):
        """Perform cross-validation"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train on this fold
            self.train(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            metrics = self.evaluate(X_val_fold, y_val_fold)
            metrics['fold'] = fold
            metrics_list.append(metrics)
        
        # Calculate average metrics
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        if 'fold' in avg_metrics:
            del avg_metrics['fold']
            
        return avg_metrics
