#Uses CSV historical stock price data manually retrieved from Nasdaq.com
#LSTM chart generator that predicts range of stock price within next few days
#Number of debugging runs: 95
#Working, this script is a back-up
#Results are suggested/intended to be fed into LLMs for analysis
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define a PyTorch Dataset for time series data with additional features
class EnhancedStockDataset(Dataset):
    def __init__(self, prices, seq_length=10, prediction_days=3):
        self.prices = prices
        self.seq_length = seq_length
        self.prediction_days = prediction_days
        
    def __len__(self):
        return len(self.prices) - self.seq_length - self.prediction_days + 1
    
    def __getitem__(self, idx):
        # Input sequence (past prices)
        x = self.prices[idx:idx+self.seq_length].flatten()
        
        # Target price (3 days in the future)
        y = self.prices[idx+self.seq_length+self.prediction_days-1].item()
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

# Define an enhanced LSTM Neural Network Model
class EnhancedLSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.3):
        super(EnhancedLSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # Bidirectional LSTM for better capturing patterns in both directions
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Reshape input to (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, seq_length, self.input_size)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        output, _ = self.lstm(x, (h0, c0))  # output shape: (batch, seq_len, hidden_size*2)
        
        # Get attention weights
        attention_weights = torch.softmax(self.attention(output), dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * output, dim=1)
        
        # Feed through fully connected layers
        out = self.fc(context)
        
        return out

# Function to create additional technical indicators
def add_technical_indicators(df):
    """Add technical analysis indicators to the dataframe"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # 1. Moving averages
    df_copy['MA5'] = df_copy['Close/Last'].rolling(window=5).mean()
    df_copy['MA10'] = df_copy['Close/Last'].rolling(window=10).mean()
    df_copy['MA20'] = df_copy['Close/Last'].rolling(window=20).mean()
    
    # 2. Exponential moving averages
    df_copy['EMA5'] = df_copy['Close/Last'].ewm(span=5, adjust=False).mean()
    df_copy['EMA10'] = df_copy['Close/Last'].ewm(span=10, adjust=False).mean()
    
    # 3. Relative Strength Index (RSI)
    delta = df_copy['Close/Last'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD (Moving Average Convergence Divergence)
    ema12 = df_copy['Close/Last'].ewm(span=12, adjust=False).mean()
    ema26 = df_copy['Close/Last'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = ema12 - ema26
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. Price rate of change
    df_copy['ROC'] = df_copy['Close/Last'].pct_change(periods=5) * 100
    
    # 6. Bollinger Bands
    df_copy['20d_std'] = df_copy['Close/Last'].rolling(window=20).std()
    df_copy['Upper_Band'] = df_copy['MA20'] + (df_copy['20d_std'] * 2)
    df_copy['Lower_Band'] = df_copy['MA20'] - (df_copy['20d_std'] * 2)
    
    # 7. Momentum
    df_copy['Momentum'] = df_copy['Close/Last'].diff(periods=4)
    
    # 8. Percentage distance from 5-day moving average
    df_copy['Dist_From_MA5%'] = ((df_copy['Close/Last'] - df_copy['MA5']) / df_copy['MA5']) * 100
    
    # 9. Trend direction (simple indicator)
    df_copy['Trend_Direction'] = np.where(df_copy['Close/Last'] > df_copy['MA10'], 1, -1)
    
    # Fill NaN values created by indicators that use windows
    df_copy.bfill()

    
    return df_copy

# Function to load and prepare the data with enhanced features
def prepare_enhanced_data(csv_path, seq_length=15, prediction_days=3, batch_size=32, test_size=0.2):
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Convert date to datetime and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Clean the 'Close/Last' column by removing '$' and converting to float
    if df['Close/Last'].dtype == object:
        df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)
    
    # Add technical indicators
    enhanced_df = add_technical_indicators(df)
    
    # Extract the features
    feature_columns = [
        'Close/Last', 'MA5', 'MA10', 'EMA5', 'RSI', 'MACD', 
        'ROC', 'Momentum', 'Dist_From_MA5%', 'Trend_Direction'
    ]
    features = enhanced_df[feature_columns].values
    
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # Extract only the normalized closing prices for predictions
    close_price_idx = feature_columns.index('Close/Last')
    prices_scaled = features_scaled[:, close_price_idx].reshape(-1, 1)
    
    # Create a price-only scaler for later prediction conversion
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(df['Close/Last'].values.reshape(-1, 1))
    
    # Prepare dataset
    dataset = EnhancedStockDataset(prices_scaled, seq_length, prediction_days)
    
    # Create chronological train/validation split (no shuffling)
    # For time series, we want the most recent data for validation
    dataset_size = len(dataset)
    train_size = int((1 - test_size) * dataset_size)
    val_size = dataset_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, price_scaler, df, enhanced_df

# Training function with learning rate scheduler and early stopping
def train_enhanced_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay for regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    
    # Early stopping initialization
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# Function to predict future prices with trend-aware logic
def predict_future_enhanced(model, df, last_sequence, price_scaler, days=3, device='cuda'):
    model.eval()
    
    # Get the last actual price point
    last_actual_price = df['Close/Last'].values[-1]
    last_scaled_value = price_scaler.transform([[last_actual_price]])[0][0]
    
    # Analyze recent trend (more weight on recent days)
    # Use exponential weighting to give more importance to recent price movements
    recent_prices = df['Close/Last'].values[-30:]
    weights = np.exp(np.linspace(0, 1, len(recent_prices)))
    weights /= weights.sum()
    
    # Calculate weighted trend using last 30 days
    x = np.arange(len(recent_prices))
    trend_slope = np.sum(weights * (recent_prices - recent_prices.mean()) * (x - x.mean())) / np.sum(weights * (x - x.mean())**2)
    
    # Calculate recent volatility based on last 10 days (more relevant)
    recent_volatility = np.std(np.diff(df['Close/Last'].values[-10:]))
    
    # Make a copy of the last sequence
    current_sequence = last_sequence.copy()
    predictions = []
    
    # Generate predictions one by one
    for i in range(days):
        # Convert to tensor and reshape
        seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(seq_tensor)
        
        # Get the raw prediction
        next_pred = pred.item()
        
        # Special handling for first prediction to ensure continuity
        if i == 0:
            # Ensure the first prediction is anchored to the last actual value
            # Blend between model prediction and continuation of recent trend
            continuation = last_scaled_value + (trend_slope / last_actual_price)  # Normalized trend
            # More weight (80%) on continuation for first prediction to ensure smoothness
            next_pred = 0.2 * next_pred + 0.8 * continuation
        else:
            # For subsequent predictions, incorporate trend information more subtly
            trend_weight = 0.03 if trend_slope < 0 else 0.02  # Asymmetric weighting
            normalized_trend = trend_slope / last_actual_price  # Normalize trend to price scale
            trend_adjustment = normalized_trend * trend_weight
            next_pred += trend_adjustment
        
        # Add minimal noise to make predictions look natural
        noise_level = recent_volatility * 0.02 / last_actual_price  # Scale noise to price level
        noise = np.random.normal(0, noise_level)
        next_pred += noise
        
        # Get the previous prediction or last actual value
        prev_value = predictions[-1] if i > 0 else last_scaled_value
        
        # Constraint changes based on trend direction and strength
        # In a strong downtrend, allow slightly larger downward moves but restrict upward moves
        if trend_slope < -0.5:  # Strong downtrend
            max_down_change = 0.015  # Allow 1.5% down
            max_up_change = 0.005    # Restrict up to 0.5%
            
            if next_pred > prev_value:
                # Limit upward move in downtrend
                next_pred = min(next_pred, prev_value + max_up_change)
            elif next_pred < prev_value:
                # Allow reasonable downward move
                next_pred = max(next_pred, prev_value - max_down_change)
        else:  # Normal or uptrend
            # Standard limit on daily changes (1%)
            max_change = 0.01
            if abs(next_pred - prev_value) > max_change:
                direction = 1 if next_pred > prev_value else -1
                next_pred = prev_value + direction * max_change
        
        predictions.append(next_pred)
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], [next_pred])
    
    # Inverse transform the predictions to get actual prices
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = price_scaler.inverse_transform(predictions)
    
    return predictions.flatten()

# Main function to run the complete workflow
def run_enhanced_stock_prediction_with_improvements(csv_path, seq_length=15, prediction_days=3, batch_size=32, epochs=100, ensemble_size=5):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare enhanced data
    train_loader, val_loader, price_scaler, df, enhanced_df = prepare_enhanced_data(
        csv_path, seq_length, prediction_days, batch_size
    )
    
    # Create an ensemble of models with slightly different configurations
    ensemble_models = []
    ensemble_predictions = []
    
    for i in range(ensemble_size):
        print(f"\nTraining model {i+1}/{ensemble_size} for ensemble...")
        
        # Vary hyperparameters slightly for each model in the ensemble
        # In your hyperparameter optimization or ensemble training loop:

        hidden_size = int(np.random.choice([64, 96, 128, 160, 192]))
        num_layers = int(np.random.choice([1, 2, 3, 4]))
        dropout = float(np.random.uniform(0.1, 0.5))
        learning_rate = float(np.random.choice([0.0001, 0.0003, 0.001, 0.003]))

        params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate
        }

        model = EnhancedLSTMPredictor(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        
        # Train model with early stopping
        model, train_losses, val_losses = train_enhanced_model(
            model, train_loader, val_loader, device, epochs, patience=15
        )
        
        ensemble_models.append(model)
        
        # Get the last sequence from the data for prediction
        prices_scaled = price_scaler.transform(df['Close/Last'].values.reshape(-1, 1))
        last_sequence = prices_scaled[-seq_length:].flatten()
        
        # Predict future prices with the current model
        model_predictions = predict_future_enhanced(model, df, last_sequence, price_scaler, prediction_days, device)
        ensemble_predictions.append(model_predictions)
    
    # Calculate the ensemble prediction (median to be robust to outliers)
    ensemble_predictions = np.array(ensemble_predictions)
    median_predictions = np.median(ensemble_predictions, axis=0)
    
    # Calculate confidence intervals (75% and 25% percentiles)
    upper_bound = np.percentile(ensemble_predictions, 75, axis=0)
    lower_bound = np.percentile(ensemble_predictions, 25, axis=0)
    
    # Get the dates for prediction
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(prediction_days)]
    
    # Create a DataFrame for the predictions with confidence intervals
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': median_predictions,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })
    
    print("\nEnsemble predicted stock prices for the next 3 days:")
    print(predictions_df[['Date', 'Predicted_Close', 'Lower_Bound', 'Upper_Bound']])
    
    # Plot historical prices and predictions with confidence interval
    plt.figure(figsize=(12, 6))
    
    # Plot the historical data
    plt.plot(df['Date'][-30:], df['Close/Last'][-30:], color='purple', linewidth=2.5, label='Historical Prices')
    
    # Plot the median prediction
    plt.plot(predictions_df['Date'], predictions_df['Predicted_Close'], 'r-', linewidth=2, 
             marker='o', markersize=6, label='Predicted Prices (Ensemble Median)')
    
    # Plot the confidence interval
    plt.fill_between(predictions_df['Date'], predictions_df['Lower_Bound'], predictions_df['Upper_Bound'], 
                      color='red', alpha=0.2, label='Prediction 50% Confidence Interval')
    
    # Highlight the last actual price point
    plt.plot(df['Date'].iloc[-1], df['Close/Last'].iloc[-1], 'o', color='purple', 
             markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    
    # Connect the last actual point to the first prediction
    plt.plot([df['Date'].iloc[-1], predictions_df['Date'].iloc[0]], 
             [df['Close/Last'].iloc[-1], predictions_df['Predicted_Close'].iloc[0]], 
             'k:', linewidth=1.5, alpha=0.7)
    
    # Improve axis labels and formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title('Stock Price Prediction with Ensemble Learning', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    # Ensure y-axis includes both historical and prediction data with some padding
    all_prices = np.concatenate([
        df['Close/Last'][-30:].values, 
        predictions_df['Predicted_Close'].values,
        predictions_df['Upper_Bound'].values,
        predictions_df['Lower_Bound'].values
    ])
    y_min = np.min(all_prices) * 0.99
    y_max = np.max(all_prices) * 1.01
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('ensemble_stock_prediction.png', dpi=300)
    plt.show()
    
    # Save the best model from the ensemble (one with lowest validation loss)
    best_model = ensemble_models[0]  # Default to first model
    lowest_val_loss = float('inf')
    
    # Get the model with lowest validation loss
    for model in ensemble_models:
        # Run a quick validation to get loss
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = nn.MSELoss()(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = model
    
    # Save best model for future use
    torch.save(best_model.state_dict(), 'best_stock_prediction_model.pth')
    
    return best_model, predictions_df, df, ensemble_models, price_scaler
def update_model_with_new_data(model, csv_path, new_data_csv=None, seq_length=15, prediction_days=3, batch_size=32, epochs=20):
    """
    Update the model with new market data as it becomes available.
    
    Args:
        model: Existing trained model
        csv_path: Original data path
        new_data_csv: Path to new data CSV with recent market data
        seq_length: Sequence length for prediction
        prediction_days: Number of days to predict
        batch_size: Batch size for training
        epochs: Number of epochs for fine-tuning
    
    Returns:
        Updated model and new data including the most recent points
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the original data
    original_df = pd.read_csv(csv_path)
    original_df['Date'] = pd.to_datetime(original_df['Date'])
    original_df = original_df.sort_values('Date')
    
    # Clean the 'Close/Last' column if needed
    if original_df['Close/Last'].dtype == object:
        original_df['Close/Last'] = original_df['Close/Last'].str.replace('$', '').astype(float)
    
    # If new data is provided, merge it with the original data
    if new_data_csv:
        new_df = pd.read_csv(new_data_csv)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # Clean the new data 'Close/Last' column if needed
        if new_df['Close/Last'].dtype == object:
            new_df['Close/Last'] = new_df['Close/Last'].str.replace('$', '').astype(float)
        
        # Merge, keeping only new dates that aren't in the original data
        combined_df = pd.concat([original_df, new_df])
        combined_df = combined_df.drop_duplicates(subset=['Date']).sort_values('Date')
    else:
        combined_df = original_df
    
    # Add technical indicators to the combined data
    enhanced_df = add_technical_indicators(combined_df)
    
    # Prepare the updated data for training
    train_loader, val_loader, price_scaler, df, enhanced_df = prepare_enhanced_data(
        None, seq_length, prediction_days, batch_size, test_size=0.1, 
        prepared_df=enhanced_df
    )
    
    # Fine-tune the model with the new data
    print("Fine-tuning model with new data...")
    model, train_losses, val_losses = train_enhanced_model(
        model, train_loader, val_loader, device, epochs=epochs, patience=5
    )
    
    print("Model fine-tuning complete!")
    
    return model, enhanced_df

def optimize_hyperparameters(csv_path, num_trials=20, seq_length=15, prediction_days=3, batch_size=32, epochs=50):
    """
    Use Bayesian Optimization to find optimal hyperparameters for the model.
    
    Args:
        csv_path: Path to the stock data CSV
        num_trials: Number of optimization trials
        seq_length: Sequence length for prediction
        prediction_days: Number of days to predict
        batch_size: Batch size for training
        epochs: Maximum epochs for each trial
        
    Returns:
        Dictionary of best hyperparameters
    """
    from sklearn.model_selection import KFold
    import random
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for hyperparameter optimization")
    
    # Prepare data once for all trials
    train_loader, val_loader, price_scaler, df, enhanced_df = prepare_enhanced_data(
        csv_path, seq_length, prediction_days, batch_size
    )
    
    # Combine train and validation data for k-fold cross-validation
    dataset = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset])
    
    best_val_loss = float('inf')
    best_params = {}
    
    results = []
    
    for trial in range(num_trials):
        # Sample hyperparameters
        hidden_size = random.choice([64, 96, 128, 160, 192])
        num_layers = random.choice([1, 2, 3, 4])
        dropout = random.uniform(0.1, 0.5)
        learning_rate = random.choice([0.0001, 0.0003, 0.001, 0.003])
        
        # Current hyperparameter set
        params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate
        }
        
        print(f"\nTrial {trial+1}/{num_trials} with params: {params}")
        
        # K-fold cross-validation
        k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(k_fold.split(dataset)):
            print(f"  Fold {fold+1}/3")
            
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader_fold = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            val_loader_fold = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=val_subsampler)
            
            # Initialize model with current hyperparameters
            model = EnhancedLSTMPredictor(
                input_size=1, 
                hidden_size=hidden_size, 
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
            
            # Train with early stopping
            model, train_losses, val_losses = train_enhanced_model(
                model, train_loader_fold, val_loader_fold, device, 
                epochs=epochs, lr=learning_rate, patience=10
            )
            
            # Record final validation loss
            cv_val_losses.append(val_losses[-1])
        
        # Average validation loss across folds
        avg_val_loss = sum(cv_val_losses) / len(cv_val_losses)
        print(f"  Average validation loss: {avg_val_loss:.6f}")
        
        # Record the results
        results.append({
            'params': params,
            'val_loss': avg_val_loss
        })
        
        # Update best parameters if this trial is better
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = params
            print(f"  New best params found with loss: {best_val_loss:.6f}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'])
    
    print("\nHyperparameter optimization results:")
    for i, result in enumerate(results[:5]):
        print(f"Rank {i+1}: Params: {result['params']}, Loss: {result['val_loss']:.6f}")
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return best_params

def backtest_model(model, df, price_scaler, seq_length=15, prediction_days=3, backtest_periods=5, device='cuda'):
    """
    Backtest the model on historical data by making predictions at different points in time
    and comparing with actual prices.
    
    Args:
        model: Trained model
        df: DataFrame with historical data
        price_scaler: Fitted scaler for prices
        seq_length: Sequence length used for predictions
        prediction_days: Number of days to predict
        backtest_periods: Number of historical points to test from
        device: Computation device
        
    Returns:
        DataFrame with backtest results
    """
    backtest_results = []
    
    # Get indices for backtest starting points
    total_points = len(df)
    available_points = total_points - seq_length - prediction_days
    
    if available_points < backtest_periods:
        backtest_periods = available_points
        print(f"Warning: Reduced backtest periods to {backtest_periods} due to data constraints")
    
    # Calculate step size to distribute backtest points evenly
    step = max(1, available_points // backtest_periods)
    
    # Get scaled prices
    prices_scaled = price_scaler.transform(df['Close/Last'].values.reshape(-1, 1))
    
    for i in range(backtest_periods):
        # Calculate the index for this backtest point
        end_idx = total_points - (backtest_periods - i) * step
        
        # Ensure we have enough data
        if end_idx - seq_length < 0 or end_idx + prediction_days > total_points:
            continue
            
        # Get the sequence for this backtest point
        sequence = prices_scaled[end_idx - seq_length:end_idx].flatten()
        
        # Make predictions
        predictions = predict_future_enhanced(model, df.iloc[:end_idx], sequence, price_scaler, prediction_days, device)
        
        # Get actual prices for the predicted period
        actual_prices = df['Close/Last'].iloc[end_idx:end_idx + prediction_days].values
        
        # Calculate prediction error
        errors = predictions - actual_prices
        mape = np.mean(np.abs(errors / actual_prices)) * 100  # Mean Absolute Percentage Error
        
        # Record results
        result = {
            'Backtest_Date': df['Date'].iloc[end_idx],
            'Predicted_Prices': predictions,
            'Actual_Prices': actual_prices,
            'MAPE': mape
        }
        backtest_results.append(result)
    
    # Calculate overall backtest performance
    all_predicted = []
    all_actual = []
    
    for result in backtest_results:
        all_predicted.extend(result['Predicted_Prices'])
        all_actual.extend(result['Actual_Prices'])
    
    overall_mape = np.mean(np.abs(np.array(all_predicted) - np.array(all_actual)) / np.array(all_actual)) * 100
    
    print(f"\nBacktest Results:")
    print(f"Overall Mean Absolute Percentage Error (MAPE): {overall_mape:.2f}%")
    
    # Create a DataFrame for detailed backtest results
    backtest_df = pd.DataFrame(backtest_results)
    
    # Visualize backtest results
    plt.figure(figsize=(14, 7))
    
    for i, result in enumerate(backtest_results):
        # Get dates for this backtest period
        start_date = result['Backtest_Date']
        dates = [start_date + pd.Timedelta(days=j) for j in range(prediction_days)]
        
        # Plot predicted vs actual
        if i == 0:  # Only add to legend once
            plt.plot(dates, result['Predicted_Prices'], 'r--', alpha=0.7, label='Predicted')
            plt.plot(dates, result['Actual_Prices'], 'b-', alpha=0.7, label='Actual')
        else:
            plt.plot(dates, result['Predicted_Prices'], 'r--', alpha=0.7)
            plt.plot(dates, result['Actual_Prices'], 'b-', alpha=0.7)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title('Backtest Results: Predicted vs Actual Prices', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300)
    plt.show()
    
    return backtest_df, overall_mape

def run_advanced_stock_prediction_pipeline(csv_path, optimize=True, ensemble=True, backtest=True, update_data=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Hyperparameter Optimization
    if optimize:
        print("\n=== Running Hyperparameter Optimization ===")
        best_params = optimize_hyperparameters(csv_path, num_trials=10)
        hidden_size = int(best_params['hidden_size'])
        num_layers = int(best_params['num_layers'])
        dropout = float(best_params['dropout'])
        learning_rate = float(best_params['learning_rate'])
    else:
        hidden_size = 128
        num_layers = 3
        dropout = 0.3
        learning_rate = 0.001

    # Step 2: Ensemble or single model training
    if ensemble:
        print("\n=== Training Ensemble Model ===")
        # FIX HERE: Properly unpack returned values (5 expected)
        model, predictions_df, df, ensemble_models, price_scaler = run_enhanced_stock_prediction_with_improvements(
            csv_path, ensemble_size=5
        )


    else:
        print("\n=== Training Single Model ===")
        train_loader, val_loader, price_scaler, df, enhanced_df = prepare_enhanced_data(
            csv_path, seq_length=15, prediction_days=3, batch_size=32
        )

        model = EnhancedLSTMPredictor(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        model, train_losses, val_losses = train_enhanced_model(
            model, train_loader, val_loader, device, epochs=100, lr=learning_rate, patience=15
        )

        prices_scaled = price_scaler.transform(df['Close/Last'].values.reshape(-1, 1))
        last_sequence = prices_scaled[-15:].flatten()

        predictions = predict_future_enhanced(model, df, last_sequence, price_scaler, days=3, device=device)

        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(3)]
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        })

        ensemble_models = None  # Set explicitly if not used

    # Step 3: Backtesting (optional)
    if backtest:
        print("\n=== Running Backtesting ===")
        backtest_results, backtest_mape = backtest_model(
            model, df, price_scaler, backtest_periods=5, device=device
        )

    # Step 4: Model Update (optional)
    if update_data:
        print("\n=== Updating Model with New Data ===")
        model, df = update_model_with_new_data(
            model, csv_path, new_data_csv=update_data
        )

    # FIX HERE: Consistent return statement
    return best_model, predictions_df, df, ensemble_models, price_scaler

# Example usage (Corrected unpacking to match function's returned values)
if __name__ == "__main__":
    csv_path = "/kaggle/input/spyhpd/SPY.csv"
    
    model, predictions_df, df, ensemble_models, price_scaler = run_advanced_stock_prediction_pipeline(
        csv_path,
        optimize=True,
        ensemble=True,
        backtest=True
    )
    
    print("\nEnhanced prediction complete!")
