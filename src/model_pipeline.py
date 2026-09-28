import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_preprocess_data(filepath):
    """
    Loads data and performs initial preprocessing:
    1. Handles missing values via KNN Imputation
    2. Performs basic feature engineering (ratios)
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Define feature categories
    categorical_cols = ['sex']
    numeric_cols = [c for c in df.columns if c not in categorical_cols + ['patient_id', 'diabetes_status']]
    
    # 1. Feature Engineering
    # Calculate Neutrophil-to-Lymphocyte Ratio (NLR)
    if 'neutrophils' in df.columns and 'lymphocytes' in df.columns:
        df['NLR'] = df['neutrophils'] / (df['lymphocytes'] + 1e-9)
        numeric_cols.append('NLR')
    
    # Calculate TG to HDL ratio
    if 'triglycerides' in df.columns and 'hdl' in df.columns:
        df['TG_HDL_ratio'] = df['triglycerides'] / (df['hdl'] + 1e-9)
        numeric_cols.append('TG_HDL_ratio')

    # 2. Imputation for missing values
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # 3. Encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    X = df.drop(columns=['patient_id', 'diabetes_status'])
    y = df['diabetes_status']
    
    return X, y

def train_and_evaluate_models(X, y):
    """
    Trains multiple ML models using cross-validation and SMOTE for class imbalance.
    Evaluates and returns performance metrics.
    """
    # Train-test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Handle Class Imbalance on training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_auc = 0
    best_model_instance = None
    
    print("Training and evaluating models...")
    for name, model in models.items():
        # Train model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        # Evaluate
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'AUC': auc,
            'Sensitivity': report['1']['recall'],
            'Specificity': report['0']['recall'],
            'F1': report['1']['f1-score'],
            'Model': model
        }
        print(f"{name} - AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_instance = model
            
    print(f"\nBest Model: {best_model_name} with AUC: {best_auc:.4f}")
    
    return best_model_instance, X_train_resampled, X_test_scaled, y_test, results

def interpret_model(model, X_train, X_test):
    """
    Uses SHAP to interpret the trained model and generate feature importance plots.
    """
    print("\nGenerating SHAP explanations for the best model...")
    
    # Note: TreeExplainer is faster for XGBoost/RandomForest
    if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_test)

    # Plot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Feature Importance Summary')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    plt.close()
    print("Saved SHAP summary plot as 'shap_summary.png'")

if __name__ == "__main__":
    # Placeholder for actual data path
    data_path = "../data/simulated_patient_data.csv"
    
    try:
        print("Starting ML Pipeline...")
        X, y = load_and_preprocess_data(data_path)
        best_model, X_train_resampled, X_test, y_test, all_results = train_and_evaluate_models(X, y)
        interpret_model(best_model, X_train_resampled, X_test)
        
        # Save best model
        joblib.dump(best_model, 'best_t2dm_screening_model.pkl')
        print("Pipeline complete. Model saved.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please place your data there and rerun.")
