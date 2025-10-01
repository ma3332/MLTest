import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time
import gc
import torch
import warnings
warnings.filterwarnings('ignore')
import shap

def clear_gpu_cache():
    """
    Clear GPU cache and free memory
    """
    try:
        # Force garbage collection
        gc.collect()
        
        # Try to clear CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared successfully")
        except ImportError:
            pass
            
        # XGBoost doesn't have explicit GPU cache clearing,
        # but we can force garbage collection
        print("🧹 Memory cache cleared")
        
    except Exception as e:
        print(f"⚠️  Could not clear cache: {e}")

def get_gpu_memory_info():
    """
    Get GPU memory information if available
    """
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
            
            print(f"🎮 GPU Memory Info:")
            print(f"   Total: {gpu_memory:.2f} GB")
            print(f"   Allocated: {gpu_allocated:.2f} GB")
            print(f"   Cached: {gpu_cached:.2f} GB")
            print(f"   Free: {gpu_memory - gpu_allocated:.2f} GB")
            
            return gpu_allocated, gpu_cached
    except ImportError:
        print("ℹ️  PyTorch not available for memory monitoring")
    except Exception as e:
        print(f"⚠️  Could not get GPU memory info: {e}")
    
    return None, None


def check_gpu_availability():
    """
    Check if GPU is available for XGBoost training
    """
    try:
        # Try to create a simple DMatrix to test GPU availability
        test_data = np.array([[1, 2], [3, 4]])
        test_label = np.array([1, 2])
        dtest = xgb.DMatrix(test_data, label=test_label)
        
        # Try GPU training with minimal parameters
        params_gpu = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'reg:squarederror'
        }
        
        # This will raise an exception if GPU is not available
        xgb.train(params_gpu, dtest, num_boost_round=1, verbose_eval=False)
        return True
    except Exception as e:
        print(f"⚠️  GPU not available: {str(e)}")
        return False

def create_sample_dataset(n_samples=1000, n_features=10, noise=0.1):
    """
    Create a sample regression dataset for demonstration
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42
    )
    
    # Convert to DataFrame for better visualization
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, X, y

def train_xgboost_model(X_train, y_train, X_val, y_val, use_gpu=True):
    """
    Train XGBoost regression model with early stopping
    Supports both GPU and CPU training
    """
    # Check GPU availability
    gpu_available = check_gpu_availability() if use_gpu else False
    
    if gpu_available:
        print("🚀 Using GPU acceleration for training...")
        # XGBoost parameters optimized for GPU
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',  # GPU-accelerated histogram method
            'gpu_id': 0,                # Use first GPU
            'max_depth': 8,             # Slightly deeper for GPU
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'predictor': 'gpu_predictor',  # GPU prediction
            'gpu_hist_dtype': 'float32',   # Use float32 for better GPU performance
        }
    else:
        print("💻 Using CPU for training...")
        # XGBoost parameters optimized for CPU
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',      # CPU histogram method
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Create DMatrix for XGBoost with feature names
    feature_names = [f'feature_{i+1}' for i in range(X_train.shape[1])]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    return model, gpu_available

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate the model and calculate RMSE and other metrics
    """
    # Make predictions
    if feature_names is None:
        feature_names = [f'feature_{i+1}' for i in range(X_test.shape[1])]
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    return y_pred, rmse, r2, mae

def plot_results(y_test, y_pred, rmse, r2):
    """
    Create visualization plots for model performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def feature_importance(model, feature_names):
    """
    Plot feature importance from XGBoost model
    """
    print(f"🔍 Attempting to get feature importance...")
    
    # Try different importance types in order of preference
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    importance = {}
    
    for imp_type in importance_types:
        try:
            importance = model.get_score(importance_type=imp_type)
            print(f"✅ Successfully got importance using '{imp_type}': {len(importance)} features")
            if importance:  # If we got non-empty results, use this type
                break
        except Exception as e:
            print(f"❌ Failed to get importance using '{imp_type}': {e}")
            continue
    
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"🔍 Final importance scores: {sorted_features}")
    
 
def analyze_with_shap(model, X_train, X_test, feature_names, sample_size=1000):
    
    print("\n🔍 Performing SHAP analysis...")
    
    # Sample data for SHAP (to avoid memory issues with large datasets)
    if len(X_train) > sample_size:
        print(f"📊 Sampling {sample_size} instances for SHAP analysis...")
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[sample_indices]
    else:
        X_train_sample = X_train
    
    if len(X_test) > sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_test_sample = X_test[sample_indices]
    else:
        X_test_sample = X_test
    
    # Create SHAP explainer
    print("🤖 Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("📈 Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test_sample)
    
    # Create feature importance from SHAP
    shap_importance = np.abs(shap_values).mean(0)
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    print(f"\n🎯 SHAP Feature Importance (Top 10):")
    print(shap_importance_df.head(10).to_string(index=False))
    
    # Create SHAP visualizations
    print("\n📊 Creating SHAP visualizations...")
    
    # 1. Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Predictions')
    plt.tight_layout()
    plt.show()
    
    # 2. Feature importance plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                        plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Mean Absolute Impact)')
    plt.tight_layout()
    plt.show()
    
    # 3. Waterfall plot for a single prediction
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test_sample[0], 
                        feature_names=feature_names, show=False)
    plt.title('SHAP Waterfall Plot - Individual Prediction Explanation')
    plt.tight_layout()
    plt.show()
    
    # 4. Partial dependence plot for top features
    top_features = shap_importance_df.head(3)['feature'].tolist()
    for i, feature in enumerate(top_features):
        if i < 3:  # Limit to top 3 features
            plt.figure(figsize=(8, 6))
            feature_idx = feature_names.index(feature)
            shap.partial_dependence_plot(
                feature_idx, model.predict, X_test_sample, 
                ice=False, model_expected_value=True, 
                feature_expected_value=True, show=False
            )
            plt.title(f'Partial Dependence Plot - {feature}')
            plt.tight_layout()
            plt.show()
    
    return shap_importance_df, shap_values, explainer
        

def main():
    """
    Main function to run the XGBoost regression pipeline
    """
    print("🚀 Starting XGBoost Regression Model Training")
    print("=" * 50)
    
    # Check system capabilities
    print("\n🔍 Checking system capabilities...")
    gpu_available = check_gpu_availability()
    if gpu_available:
        print("✅ GPU acceleration available!")
    else:
        print("ℹ️  GPU not available, will use CPU")
    
    # 1. Create sample dataset (larger for GPU demonstration)
    print("\n📊 Creating sample dataset...")
    # Use larger dataset to better showcase GPU benefits
    dataset_size = 50000 if gpu_available else 1000
    df, X, y = create_sample_dataset(n_samples=dataset_size, n_features=20, noise=0.1)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target statistics:")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Std: {y.std():.4f}")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    
    # 2. Split data
    print("\n🔄 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train XGBoost model
    print("\n🤖 Training XGBoost model...")
    
    # Check GPU memory before training
    if gpu_available:
        print("\n📊 GPU Memory before training:")
        get_gpu_memory_info()
    
    start_time = time.time()
    model, gpu_used = train_xgboost_model(X_train, y_train, X_val, y_val, use_gpu=True)
    training_time = time.time() - start_time
    print(f"✅ Model training completed! {'(GPU accelerated)' if gpu_used else '(CPU)'}")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    
    # Check GPU memory after training
    if gpu_used:
        print("\n📊 GPU Memory after training:")
        get_gpu_memory_info()
    
    # Clear GPU cache after training
    print("\n🧹 Cleaning up GPU memory...")
    clear_gpu_cache()
    
    # 4. Evaluate model
    print("\n📈 Evaluating model on test set...")
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    y_pred, rmse, r2, mae = evaluate_model(model, X_test, y_test, feature_names)
    
    print(f"\n🎯 Model Performance Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # 5. Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Performance plots
    plot_results(y_test, y_pred, rmse, r2)
    
    # Feature importance plot
    feature_importance(model, feature_names)
    
    # SHAP analysis
    shap_importance_df, shap_values, explainer = analyze_with_shap(
        model, X_train, X_test, feature_names, sample_size=1000
    )
     
    # 6. Additional analysis
    print(f"\n📋 Additional Analysis:")
    print(f"  Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  Actual range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  Mean absolute percentage error: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%")
    
    print(f"\n✅ XGBoost regression model training and evaluation completed!")
    print(f"   Final RMSE: {rmse:.4f}")
    
    # Final cleanup
    print("\n🧹 Final memory cleanup...")
    clear_gpu_cache()
    
    return model, rmse, r2, mae

if __name__ == "__main__":
    # Run the main pipeline
    model, rmse, r2, mae = main()
