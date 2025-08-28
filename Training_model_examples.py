import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import glob
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

print("="*60)
print("CHUNKED DATA MODEL TRAINING")
print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS): 2025-08-28 15:23:44")
print(f"Current User's Login: Alysoffar")
print("="*60)

# Paths
CHUNKED_DATA_DIR = "/kaggle/working/chunked_data"
OUTPUT_DIR = "/kaggle/working/"

# Model training configuration
TARGET_COLUMN = 'PM25_ug_m3'  # Target variable
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# METHOD 1: LOAD ALL CHUNKS INTO SINGLE DATAFRAME (MEMORY PERMITTING)
# =============================================================================

def load_all_chunks_combined(chunk_dir, file_pattern="*_chunk_*.parquet"):
    """
    Load all chunk files and combine into single DataFrame.
    Best for: Medium datasets that fit in memory after combining.
    """
    print(f"\n📥 METHOD 1: Loading all chunks combined")
    print(f"Searching for pattern: {file_pattern} in {chunk_dir}")
    
    # Find all chunk files
    chunk_files = glob.glob(os.path.join(chunk_dir, file_pattern))
    chunk_files.sort()  # Ensure consistent ordering
    
    if not chunk_files:
        print(f"❌ No chunk files found matching pattern: {file_pattern}")
        return None
    
    print(f"✅ Found {len(chunk_files)} chunk files")
    
    # Load and combine all chunks
    dataframes = []
    total_rows = 0
    
    for i, file_path in enumerate(chunk_files):
        try:
            chunk_df = pd.read_parquet(file_path)
            dataframes.append(chunk_df)
            total_rows += len(chunk_df)
            print(f"   📄 Loaded chunk {i+1}: {len(chunk_df):,} rows")
            
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
            continue
    
    if not dataframes:
        print(f"❌ No chunks loaded successfully")
        return None
    
    # Combine all chunks
    print(f"\n🔄 Combining {len(dataframes)} chunks...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"✅ Combined dataset shape: {combined_df.shape}")
    print(f"   Total rows: {total_rows:,}")
    
    return combined_df

# =============================================================================
# METHOD 2: INCREMENTAL TRAINING ON CHUNKS
# =============================================================================

def train_model_incremental(chunk_dir, target_column, file_pattern="*_chunk_*.parquet"):
    """
    Train model incrementally on chunks (online learning).
    Best for: Very large datasets that don't fit in memory.
    """
    print(f"\n🎯 METHOD 2: Incremental training on chunks")
    
    # Find chunk files
    chunk_files = glob.glob(os.path.join(chunk_dir, file_pattern))
    chunk_files.sort()
    
    if not chunk_files:
        print(f"❌ No chunk files found")
        return None, None
    
    print(f"✅ Found {len(chunk_files)} chunks for incremental training")
    
    # Use models that support incremental learning
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    
    model = SGDRegressor(random_state=RANDOM_STATE, max_iter=1000)
    scaler = StandardScaler()
    
    # Process first chunk to initialize scaler and get feature names
    first_chunk = pd.read_parquet(chunk_files[0])
    
    if target_column not in first_chunk.columns:
        print(f"❌ Target column '{target_column}' not found in chunks")
        return None, None
    
    # Identify feature columns (exclude metadata)
    metadata_cols = ['time', 'lat', 'lon']
    feature_cols = [col for col in first_chunk.columns 
                   if col != target_column and col not in metadata_cols]
    
    print(f"📊 Feature columns identified: {len(feature_cols)}")
    print(f"   Features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"   Features: {feature_cols}")
    
    # Initialize scaler with first chunk
    X_first = first_chunk[feature_cols].fillna(0)  # Handle any NaN values
    y_first = first_chunk[target_column].fillna(first_chunk[target_column].mean())
    
    scaler.fit(X_first)
    X_first_scaled = scaler.transform(X_first)
    model.fit(X_first_scaled, y_first)
    
    print(f"   🏁 Initialized with chunk 1: {len(X_first):,} samples")
    
    # Train incrementally on remaining chunks
    total_samples = len(X_first)
    
    for i, chunk_file in enumerate(chunk_files[1:], 2):
        try:
            chunk_df = pd.read_parquet(chunk_file)
            
            X_chunk = chunk_df[feature_cols].fillna(0)
            y_chunk = chunk_df[target_column].fillna(chunk_df[target_column].mean())
            
            # Scale features using pre-fitted scaler
            X_chunk_scaled = scaler.transform(X_chunk)
            
            # Partial fit (incremental training)
            model.partial_fit(X_chunk_scaled, y_chunk)
            total_samples += len(X_chunk)
            
            print(f"   📈 Trained on chunk {i}: {len(X_chunk):,} samples (Total: {total_samples:,})")
            
        except Exception as e:
            print(f"   ❌ Error processing chunk {i}: {e}")
            continue
    
    print(f"✅ Incremental training completed on {total_samples:,} total samples")
    
    return model, scaler, feature_cols

# =============================================================================
# METHOD 3: BATCH LOADING WITH MEMORY MANAGEMENT
# =============================================================================

def train_model_batch_chunks(chunk_dir, target_column, batch_size=3, file_pattern="*_chunk_*.parquet"):
    """
    Load and train on batches of chunks to manage memory.
    Best for: Large datasets where you want to control memory usage.
    """
    print(f"\n📦 METHOD 3: Batch processing chunks (batch_size={batch_size})")
    
    # Find chunk files
    chunk_files = glob.glob(os.path.join(chunk_dir, file_pattern))
    chunk_files.sort()
    
    if not chunk_files:
        print(f"❌ No chunk files found")
        return None, None
    
    print(f"✅ Found {len(chunk_files)} chunks")
    print(f"   Processing in batches of {batch_size} chunks")
    
    # Collect all data in batches
    all_X = []
    all_y = []
    feature_cols = None
    
    for i in range(0, len(chunk_files), batch_size):
        batch_files = chunk_files[i:i+batch_size]
        print(f"\n🔄 Processing batch {i//batch_size + 1}: {len(batch_files)} files")
        
        batch_data = []
        for file_path in batch_files:
            try:
                chunk_df = pd.read_parquet(file_path)
                batch_data.append(chunk_df)
                print(f"   📄 Loaded: {os.path.basename(file_path)} ({len(chunk_df):,} rows)")
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
        
        if not batch_data:
            continue
        
        # Combine batch
        batch_combined = pd.concat(batch_data, ignore_index=True)
        print(f"   📊 Batch combined shape: {batch_combined.shape}")
        
        # Identify features on first batch
        if feature_cols is None:
            if target_column not in batch_combined.columns:
                print(f"❌ Target column '{target_column}' not found")
                return None, None
            
            metadata_cols = ['time', 'lat', 'lon']
            feature_cols = [col for col in batch_combined.columns 
                           if col != target_column and col not in metadata_cols]
            print(f"   🎯 Features identified: {len(feature_cols)}")
        
        # Extract features and target
        X_batch = batch_combined[feature_cols].fillna(0)
        y_batch = batch_combined[target_column].fillna(batch_combined[target_column].mean())
        
        all_X.append(X_batch)
        all_y.append(y_batch)
        
        print(f"   ✅ Batch processed: {len(X_batch):,} samples")
        
        # Clear memory
        del batch_data, batch_combined, X_batch, y_batch
    
    # Combine all batches
    print(f"\n🔄 Combining all batches...")
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    print(f"✅ Final combined data shape: {X_combined.shape}")
    print(f"   Total samples: {len(X_combined):,}")
    
    return X_combined, y_combined, feature_cols

# =============================================================================
# METHOD 4: SMART CHUNK LOADING WITH MANIFEST
# =============================================================================

def load_chunks_with_manifest(chunk_dir):
    """
    Use chunk manifest for intelligent loading.
    Best for: When you have chunk metadata and want optimal loading.
    """
    print(f"\n📋 METHOD 4: Loading with manifest file")
    
    manifest_path = os.path.join(chunk_dir, "chunk_manifest.json")
    
    if not os.path.exists(manifest_path):
        print(f"⚠️  Manifest file not found: {manifest_path}")
        print(f"   Falling back to pattern-based loading...")
        return load_all_chunks_combined(chunk_dir)
    
    # Load manifest
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"✅ Manifest loaded successfully")
        
        # Display manifest info
        chunk_info = manifest.get('chunk_info', [])
        normalized_files = manifest.get('chunk_files', {}).get('normalized_data', [])
        pca_files = manifest.get('chunk_files', {}).get('pca_data', [])
        
        print(f"   📊 Chunks info: {len(chunk_info)} chunks")
        print(f"   📁 Normalized files: {len(normalized_files)}")
        print(f"   🎯 PCA files: {len(pca_files)}")
        
        # Load files based on manifest
        print(f"\n🔄 Loading files based on manifest...")
        
        # Choose which files to load (normalized vs PCA)
        files_to_load = normalized_files  # Use normalized data for training
        
        dataframes = []
        for file_path in files_to_load:
            if os.path.exists(file_path):
                chunk_df = pd.read_parquet(file_path)
                dataframes.append(chunk_df)
                print(f"   ✅ Loaded: {os.path.basename(file_path)} ({len(chunk_df):,} rows)")
            else:
                print(f"   ⚠️  File not found: {file_path}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"✅ Combined dataset from manifest: {combined_df.shape}")
            return combined_df
        else:
            print(f"❌ No files loaded from manifest")
            return None
            
    except Exception as e:
        print(f"❌ Error reading manifest: {e}")
        return None

# =============================================================================
# COMPLETE MODEL TRAINING EXAMPLE
# =============================================================================

def complete_model_training_example():
    """
    Complete example showing how to train a model on chunked data.
    """
    print(f"\n" + "="*60)
    print("COMPLETE MODEL TRAINING EXAMPLE")
    print("="*60)
    
    # Step 1: Check what chunk files exist
    print(f"\n1️⃣ Checking available chunk files...")
    
    if not os.path.exists(CHUNKED_DATA_DIR):
        print(f"❌ Chunked data directory not found: {CHUNKED_DATA_DIR}")
        print(f"   The dataset was likely processed as a single file.")
        print(f"   Check for these files instead:")
        
        single_files = [
            "normalized_air_quality_data_comprehensive.parquet",
            "pca_air_quality_data_comprehensive.parquet"
        ]
        
        for file in single_files:
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                print(f"   ✅ {file}: {df.shape}")
            else:
                print(f"   ❌ {file}: Not found")
        return
    
    # List available chunk files
    chunk_files = glob.glob(os.path.join(CHUNKED_DATA_DIR, "*.parquet"))
    normalized_chunks = [f for f in chunk_files if "normalized_chunk" in f]
    pca_chunks = [f for f in chunk_files if "pca_chunk" in f]
    
    print(f"   📁 Chunked data directory: {CHUNKED_DATA_DIR}")
    print(f"   📊 Normalized chunks: {len(normalized_chunks)}")
    print(f"   🎯 PCA chunks: {len(pca_chunks)}")
    
    if not normalized_chunks and not pca_chunks:
        print(f"   ❌ No chunk files found")
        return
    
    # Step 2: Choose loading method and load data
    print(f"\n2️⃣ Loading data for model training...")
    
    # Method choice based on available data
    if normalized_chunks:
        print(f"   Using normalized chunks for training...")
        # Try manifest-based loading first
        data = load_chunks_with_manifest(CHUNKED_DATA_DIR)
        if data is None:
            # Fallback to pattern-based loading
            data = load_all_chunks_combined(CHUNKED_DATA_DIR, "*normalized_chunk*.parquet")
    elif pca_chunks:
        print(f"   Using PCA chunks for training...")
        data = load_all_chunks_combined(CHUNKED_DATA_DIR, "*pca_chunk*.parquet")
    else:
        print(f"   ❌ No suitable chunks found")
        return
    
    if data is None:
        print(f"   ❌ Failed to load data")
        return
    
    print(f"✅ Data loaded successfully: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Step 3: Prepare features and target
    print(f"\n3️⃣ Preparing features and target...")
    
    # Check if target column exists
    if TARGET_COLUMN not in data.columns:
        print(f"❌ Target column '{TARGET_COLUMN}' not found in data")
        print(f"   Available columns: {list(data.columns)}")
        
        # Suggest alternative targets
        potential_targets = [col for col in data.columns if 'PM' in col.upper()]
        if potential_targets:
            print(f"   Potential PM-related targets: {potential_targets}")
            TARGET_COLUMN = potential_targets[0]
            print(f"   Using {TARGET_COLUMN} as target instead")
        else:
            # Use PC1 as target for demonstration
            pc_columns = [col for col in data.columns if col.startswith('PC')]
            if pc_columns:
                TARGET_COLUMN = pc_columns[0]
                print(f"   Using {TARGET_COLUMN} as demonstration target")
            else:
                print(f"   No suitable target found")
                return
    
    # Separate features and target
    metadata_cols = ['time', 'lat', 'lon']
    feature_cols = [col for col in data.columns 
                   if col != TARGET_COLUMN and col not in metadata_cols]
    
    print(f"   🎯 Target column: {TARGET_COLUMN}")
    print(f"   📊 Feature columns: {len(feature_cols)}")
    print(f"   📋 Features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"   Features: {feature_cols}")
    
    # Extract features and target
    X = data[feature_cols].fillna(0)  # Handle missing values
    y = data[TARGET_COLUMN].fillna(data[TARGET_COLUMN].mean())
    
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   ✅ Target shape: {y.shape}")
    print(f"   📈 Target statistics:")
    print(f"      Mean: {y.mean():.3f}")
    print(f"      Std:  {y.std():.3f}")
    print(f"      Min:  {y.min():.3f}")
    print(f"      Max:  {y.max():.3f}")
    
    # Step 4: Train-test split
    print(f"\n4️⃣ Creating train-test split...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"   📊 Training set: {X_train.shape}")
    print(f"   📊 Test set: {X_test.shape}")
    
    # Step 5: Train models
    print(f"\n5️⃣ Training models...")
    
    # Model 1: Random Forest
    print(f"   🌲 Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"      📈 Random Forest Results:")
    print(f"         MSE: {rf_mse:.6f}")
    print(f"         R²:  {rf_r2:.6f}")
    
    # Model 2: Linear Regression
    print(f"   📏 Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    lr_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    print(f"      📈 Linear Regression Results:")
    print(f"         MSE: {lr_mse:.6f}")
    print(f"         R²:  {lr_r2:.6f}")
    
    # Step 6: Feature importance (Random Forest)
    print(f"\n6️⃣ Feature importance analysis...")
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   🔝 Top 10 most important features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"      {i+1:2d}. {row['feature']}: {row['importance']:.6f}")
    
    # Step 7: Save trained models
    print(f"\n7️⃣ Saving trained models...")
    
    model_files = {}
    
    # Save Random Forest
    rf_path = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
    joblib.dump(rf_model, rf_path)
    model_files['random_forest'] = rf_path
    print(f"   💾 Random Forest saved: {rf_path}")
    
    # Save Linear Regression
    lr_path = os.path.join(OUTPUT_DIR, "linear_regression_model.joblib")
    joblib.dump(lr_model, lr_path)
    model_files['linear_regression'] = lr_path
    print(f"   💾 Linear Regression saved: {lr_path}")
    
    # Save feature importance
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feature_importance.to_csv(fi_path, index=False)
    print(f"   💾 Feature importance saved: {fi_path}")
    
    # Save model metadata
    model_metadata = {
        'timestamp': '2025-08-28 15:23:44',
        'user': 'Alysoffar',
        'target_column': TARGET_COLUMN,
        'feature_columns': feature_cols,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'models': {
            'random_forest': {
                'file': rf_path,
                'mse': float(rf_mse),
                'r2': float(rf_r2),
                'n_estimators': 100
            },
            'linear_regression': {
                'file': lr_path,
                'mse': float(lr_mse),
                'r2': float(lr_r2)
            }
        },
        'data_source': 'chunked_data' if len(normalized_chunks) > 0 or len(pca_chunks) > 0 else 'single_file'
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "model_training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"   💾 Metadata saved: {metadata_path}")
    
    # Final summary
    print(f"\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\n📊 Training Summary:")
    print(f"   • Data source: {'Chunked data' if normalized_chunks or pca_chunks else 'Single file'}")
    print(f"   • Total samples: {len(data):,}")
    print(f"   • Features used: {len(feature_cols)}")
    print(f"   • Target variable: {TARGET_COLUMN}")
    print(f"   • Models trained: 2 (Random Forest, Linear Regression)")
    
    print(f"\n🏆 Best Model Performance:")
    if rf_r2 > lr_r2:
        print(f"   • Random Forest (R² = {rf_r2:.6f})")
    else:
        print(f"   • Linear Regression (R² = {lr_r2:.6f})")
    
    print(f"\n💾 Files Created:")
    print(f"   • {os.path.basename(rf_path)}")
    print(f"   • {os.path.basename(lr_path)}")
    print(f"   • {os.path.basename(fi_path)}")
    print(f"   • {os.path.basename(metadata_path)}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Download model files from Kaggle Output tab")
    print(f"   2. Load models using: joblib.load('model_file.joblib')")
    print(f"   3. Make predictions on new data")
    print(f"   4. Fine-tune hyperparameters if needed")

# =============================================================================
# UTILITY FUNCTION: LOAD TRAINED MODEL FOR PREDICTION
# =============================================================================

def load_and_predict_example():
    """
    Example of how to load trained model and make predictions.
    """
    print(f"\n" + "="*60)
    print("LOAD TRAINED MODEL EXAMPLE")
    print("="*60)
    
    # Load model
    model_path = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
    metadata_path = os.path.join(OUTPUT_DIR, "model_training_metadata.json")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"   Run complete_model_training_example() first")
        return
    
    # Load model and metadata
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"   • Model type: Random Forest")
    print(f"   • Features expected: {len(metadata['feature_columns'])}")
    print(f"   • Target variable: {metadata['target_column']}")
    print(f"   • Training R²: {metadata['models']['random_forest']['r2']:.6f}")
    
    # Example prediction (using dummy data)
    print(f"\n🔮 Making example predictions...")
    
    # Create dummy data with same features
    feature_names = metadata['feature_columns']
    dummy_data = np.random.randn(5, len(feature_names))  # 5 samples
    dummy_df = pd.DataFrame(dummy_data, columns=feature_names)
    
    predictions = model.predict(dummy_df)
    
    print(f"   📊 Predictions on dummy data:")
    for i, pred in enumerate(predictions):
        print(f"      Sample {i+1}: {pred:.3f}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the complete training example
    complete_model_training_example()
    
    # Optionally run the prediction example
    print(f"\n" + "="*40)
    print("Running prediction example...")
    load_and_predict_example()