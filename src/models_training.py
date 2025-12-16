import logging
import os
import pickle
import optuna
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

class RFModel():
    def train(self, X_train, y_train, verbose=True, **kwargs):
        if verbose:
            logging.info('Initialize Random Forest Model')
        model = RandomForestClassifier(
            random_state=42, criterion="log_loss", class_weight="balanced", **kwargs)
        model.fit(X_train, y_train)
        
        if verbose:
            logging.info('Random Forest Model Training Completed Successfully')
        return model
    
    def optimize(self, trial, X_train, X_test, y_train, y_test):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        model = self.train(
            X_train, y_train,
            verbose=False,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap
        )
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy

class LGBMModel():
    def train(self, X_train, y_train, verbose=True, **kwargs):
        if verbose:
            logging.info('Initialize LightGBM Model')
        model = LGBMClassifier(
            random_state=42, objective='multiclass', num_class=3, 
            metric="multi_logloss", is_unbalance=True, class_weight="balanced", **kwargs)
        model.fit(X_train, y_train)
        
        if verbose:
            logging.info('LightGBM Model Training Completed Successfully')
        return model
    
    def optimize(self, trial, X_train, X_test, y_train, y_test):
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        min_child_samples = trial.suggest_int('min_child_samples', 10, 100)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True)  # L1 regularization
        reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)  # L2 regularization
        
        model = self.train(
            X_train, y_train,
            verbose=False,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda
        )
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy

class HyperparameterTuner:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(
            lambda trial: self.model.optimize(trial, self.X_train, self.X_test, self.y_train, self.y_test), 
            n_trials=n_trials,
            show_progress_bar=True
        )
        return study.best_trial.params

def get_model_trainer(model_name, X_train, y_train, X_test, y_test):
    try:
        if model_name == 'random_forest':
            model = RFModel()
        elif model_name == 'lightgbm':
            model = LGBMModel()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        logging.info(f'Starting hyperparameter optimization for {model_name}...')
        tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)
        best_params = tuner.optimize(n_trials=100)
        
        logging.info(f'Best hyperparameters for {model_name}: {best_params}')
        
        trained_model = model.train(X_train, y_train, verbose=True, **best_params)
        
        return trained_model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

def save_model(model, file_path: str) -> None:
    try:
        # Ensure the artifacts directory exists
        if os.path.exists(r'.\artifacts') is False:
            os.makedirs(r'.\artifacts')
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.debug('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise