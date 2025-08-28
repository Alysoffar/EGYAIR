def load_and_train_on_chunks(chunk_dir="/kaggle/working/chunked_data", target_col='PM25_ug_m3'):
    """
    Loader function to train model on chunks without combining them.
    """
    import pandas as pd
    import glob
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Find chunks
    chunk_files = glob.glob(f"{chunk_dir}/*normalized_chunk*.parquet")
    chunk_files.sort()
    
    if not chunk_files:
        print(" No chunks found")
        return None
    
    # Initialize model
    model = SGDRegressor(random_state=42)
    scaler = StandardScaler()
    
    # Train on chunks one by one
    for i, chunk_file in enumerate(chunk_files):
        chunk = pd.read_parquet(chunk_file)
        
        # Prepare features and target
        X = chunk.drop([target_col, 'time', 'lat', 'lon'], axis=1, errors='ignore')
        y = chunk[target_col]
        
        if i == 0:
            # First chunk: fit scaler and model
            scaler.fit(X)
            X_scaled = scaler.transform(X)
            model.fit(X_scaled, y)
        else:
            # Subsequent chunks: continue training
            X_scaled = scaler.transform(X)
            model.partial_fit(X_scaled, y)
        
        print(f" Trained on chunk {i+1}: {len(X)} samples")
    
    return model, scaler

# Usage:
model, scaler = load_and_train_on_chunks()