import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import joblib
import sys
import os
from datetime import datetime
import shap
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class MLModel:
    def __init__(self, algorithm='xgboost', params=None):
        """Initialize the model with a chosen algorithm
        
        Args:
            algorithm: Model type ('xgboost', 'rf', 'lgbm')
            params: Dictionary of model parameters
        """
        self.algorithm = algorithm
        self.version = str(uuid.uuid4())[:8]  # Generate a unique model version ID
        
        # Default parameters
        if params is None:
            params = {}
            
        # Initialize the selected algorithm
        if algorithm == 'xgboost':
            from xgboost import XGBClassifier
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            # Update defaults with provided params
            default_params.update(params)
            self.model = XGBClassifier(**default_params)
            
        elif algorithm == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': RANDOM_STATE
            }
            default_params.update(params)
            self.model = RandomForestClassifier(**default_params)
            
        elif algorithm == 'lgbm':
            import lightgbm as lgb
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.01,
                'random_state': RANDOM_STATE
            }
            default_params.update(params)
            self.model = lgb.LGBMClassifier(**default_params)
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_params = None
        self.selected_features = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_selection=False):
        """Train the model with validation if provided"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # Apply feature selection if requested
        if feature_selection:
            self._select_features(X_train_df, y_train)
            X_train_df = X_train_df[self.selected_features]
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            
            if feature_selection and self.selected_features is not None:
                X_val_df = X_val_df[self.selected_features]
                
            eval_set = [(X_train_df, y_train), (X_val_df, y_val)]
            
            if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
                self.model.fit(
                    X_train_df, 
                    y_train,
                    eval_set=eval_set,
                    verbose=100,
                    early_stopping_rounds=20
                )
            else:
                self.model.fit(X_train_df, y_train)
        else:
            self.model.fit(X_train_df, y_train)
            
        # Save feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            features = self.selected_features if self.selected_features is not None else X_train.columns
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
    def _select_features(self, X, y, threshold='median'):
        """Select important features using the model's feature importance"""
        selector = SelectFromModel(self.model, threshold=threshold)
        selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support()
        self.selected_features = X.columns[selected_indices].tolist()
        self.feature_selector = selector
        
        return self.selected_features
        
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Use selected features if available
        if self.selected_features is not None:
            X_df = X_df[self.selected_features]
            
        return self.model.predict(X_df)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        X_scaled = self.scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Use selected features if available
        if self.selected_features is not None:
            X_df = X_df[self.selected_features]
            
        return self.model.predict_proba(X_df)
    
    def evaluate(self, X, y, detailed=False):
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1': f1_score(y, predictions)
        }
        
        if detailed:
            # Add confusion matrix
            cm = confusion_matrix(y, predictions)
            metrics['confusion_matrix'] = cm
            
            # Add class probabilities if available
            if hasattr(self, 'predict_proba'):
                try:
                    proba = self.predict_proba(X)
                    metrics['avg_confidence'] = np.mean(np.max(proba, axis=1))
                except:
                    pass
        
        return metrics
    
    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring='f1'):
        """Find optimal hyperparameters using grid search"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.cols)
        
        # Use selected features if available
        if self.selected_features is not None:
            X_df = X_df[self.selected_features]
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_df, y)
        
        # Save best parameters and update model
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_
        }
    
    def plot_learning_curves(self, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
        """Generate learning curves to diagnose bias/variance issues"""
        X_scaled = self.scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_df, y, cv=cv, train_sizes=train_sizes, scoring='f1'
        )
        
        # Calculate mean and std for training and test scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot the learning curves
        plt.figure(figsize=(10, 6))
        plt.title("Learning Curves")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.grid()
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="r")
        plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="r", label="Cross-validation score")
        plt.legend(loc="best")
        
        # Save the plot
        os.makedirs('models/plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f'models/plots/learning_curve_{self.algorithm}_{timestamp}.png'
        plt.savefig(plot_path)
        
        return plot_path
        
    def explain_model(self, X, max_display=10):
        """Generate SHAP values to explain model predictions"""
        try:
            X_scaled = self.scaler.transform(X)
            X_df = pd.DataFrame(X_scaled, columns=X.columns)
            
            if self.selected_features is not None:
                X_df = X_df[self.selected_features]
                
            # Create explainer
            explainer = shap.Explainer(self.model, X_df)
            shap_values = explainer(X_df)
            
            # Create and save summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=max_display, show=False)
            
            os.makedirs('models/explanations', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f'models/explanations/shap_summary_{self.algorithm}_{timestamp}.png'
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            return {
                'shap_values': shap_values,
                'plot_path': plot_path
            }
            
        except Exception as e:
            print(f"Error generating SHAP values: {e}")
            return None
    
    def save_model(self, path='models/saved'):
        """Save the model, scaler, and metadata"""
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join(path, f'{self.algorithm}_model_{self.version}_{timestamp}.joblib')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, f'scaler_{self.version}_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature selector if available
        feature_selector_path = None
        if self.feature_selector is not None:
            feature_selector_path = os.path.join(path, f'feature_selector_{self.version}_{timestamp}.joblib')
            joblib.dump(self.feature_selector, feature_selector_path)
        
        # Save feature importance if available
        importance_path = None
        if hasattr(self, 'feature_importance'):
            importance_path = os.path.join(path, f'feature_importance_{self.version}_{timestamp}.csv')
            self.feature_importance.to_csv(importance_path, index=False)
        
        # Save metadata
        metadata = {
            'algorithm': self.algorithm,
            'version': self.version,
            'timestamp': timestamp,
            'selected_features': self.selected_features,
            'best_params': self.best_params
        }
        metadata_path = os.path.join(path, f'metadata_{self.version}_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f)
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'feature_selector_path': feature_selector_path,
            'importance_path': importance_path,
            'metadata_path': metadata_path
        }
    
    def load_model(self, model_path, scaler_path, feature_selector_path=None, metadata_path=None):
        """Load a saved model, scaler and metadata"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        if feature_selector_path:
            self.feature_selector = joblib.load(feature_selector_path)
            
        if metadata_path:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.algorithm = metadata.get('algorithm', 'xgboost')
                self.version = metadata.get('version', str(uuid.uuid4())[:8])
                self.selected_features = metadata.get('selected_features')
                self.best_params = metadata.get('best_params')
        
    def get_top_features(self, n=10):
        """Get top n most important features"""
        if hasattr(self, 'feature_importance'):
            return self.feature_importance.head(n)
        return None
    
    def cross_validate(self, X, y, n_splits=5):
        """Perform cross-validation"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train on this fold
            self.train(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            metrics = self.evaluate(X_val_fold, y_val_fold, detailed=True)
            metrics['fold'] = fold
            metrics_list.append({k: v for k, v in metrics.items() if k != 'confusion_matrix'})
            
            # Print fold results
            print(f"Fold {fold}/{n_splits} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Calculate average metrics
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        if 'fold' in avg_metrics:
            del avg_metrics['fold']
            
        return avg_metrics
