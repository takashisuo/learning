import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None

import optuna
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
import warnings
import logging
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Create logger
logger = logging.getLogger(__name__)

def load_data():
    # Load Titanic dataset
    df = sns.load_dataset('titanic')
    
    # Simple preprocessing
    # Drop columns that are not useful or require complex encoding for this demo
    drop_cols = ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'embarked']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Encode categorical variables
    # sex: male=0, female=1
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    
    # alone: boolean to int
    if 'alone' in df.columns:
        df['alone'] = df['alone'].astype(int)
    
    # Fill missing values
    # age: fill with mean
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].mean())
    
    # Drop remaining rows with missing values
    df = df.dropna()
    
    return df

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("Loaded Configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Set MLflow tracking URI to the original working directory
    # Hydra changes the working directory to the output directory, so we need to use get_original_cwd()
    import pathlib
    original_cwd = hydra.utils.get_original_cwd()
    mlflow_tracking_uri = pathlib.Path(original_cwd).joinpath('mlruns').as_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    
    # Set MLflow experiment
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Data loaded. Shape: {df.shape}")
    
    X = df.drop(columns=[cfg.data.target])
    y = df[cfg.data.target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def objective(trial):
        with mlflow.start_run(nested=True):
            model_name = cfg.model.name
            mlflow.log_param("model_name", model_name)
            
            if model_name == "lgbm":
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                }
                
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test)
                
                model = lgb.train(params, train_data, valid_sets=[valid_data], callbacks=[lgb.log_evaluation(0)])
                preds = model.predict(X_test)
                
            elif model_name == "catboost":
                if cb is None:
                    raise ImportError("CatBoost is not installed. Please install it to use this model.")
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'loss_function': 'Logloss',
                    'verbose': 0
                }
                
                train_pool = cb.Pool(X_train, y_train)
                test_pool = cb.Pool(X_test, y_test)
                
                model = cb.CatBoostClassifier(**params)
                model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20)
                preds = model.predict_proba(X_test)[:, 1]
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            auc = roc_auc_score(y_test, preds)
            
            # Log params and metrics to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("auc", auc)
            
            return auc

    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    
    logger.info("\nOptimization finished.")
    logger.info(f"Best trial value (AUC): {study.best_trial.value}")
    logger.info(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()
