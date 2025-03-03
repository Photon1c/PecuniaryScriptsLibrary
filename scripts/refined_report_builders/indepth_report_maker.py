#A lot of debugging needed, reaching a thousand lines of code
import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

def analyze_stock_tickers(tickers_csv_path, stocks_dir, options_dir):
    """
    Process stock tickers from a CSV file, analyze historical data and options data,
    and generate a sentiment report based on directional momentum.
    
    Args:
        tickers_csv_path (str): Path to CSV file containing stock tickers
        stocks_dir (str): Path to directory containing historical stock data CSVs
        options_dir (str): Path to directory containing options data
    
    Returns:
        pd.DataFrame: Summary report with ticker sentiment analysis
    """
    # Step 1: Load the list of stock tickers
    try:
        tickers_df = pd.read_csv(tickers_csv_path)
        tickers = tickers_df.iloc[:, 0].tolist()  # Assuming tickers are in the first column
        print(f"Loaded {len(tickers)} tickers from {tickers_csv_path}")
    except Exception as e:
        print(f"Error loading tickers CSV: {e}")
        return None
    
    # Initialize results dictionary
    results = {
        'Ticker': [],
        'Stock_Data_Available': [],
        'Options_Data_Available': [],
        'Analysis_Status': [],
        'Momentum_Score': [],
        'Volume_Trend': [],
        'Price_Trend': [],
        'RSI_Status': [],
        'MACD_Signal': [],
        'Bollinger_Position': [],
        'Volatility': [],
        'Put_Call_Ratio': [],
        'Short_Term_Forecast': [],
        'Sentiment': [],
        'Confidence': []
    }
    
    # Create a detailed analysis report dictionary
    detailed_reports = {}
    
    # Step 2: Process each ticker
    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        stock_data_available = False
        options_data_available = False
        
        # Step 2A: Check for stock historical data
        stock_file_path = os.path.join(stocks_dir, f"{ticker}.csv")

        if os.path.exists(stock_file_path):
            stock_data_available = True
            print(f"Found historical data for {ticker}")
        else:
            print(f"No historical data found for {ticker}")
        
        # Step 2B: Check for options data
        ticker_options_dir = os.path.join(options_dir, ticker.lower())
        if os.path.exists(ticker_options_dir):
            # Look for option chain directories (organized by date)
            date_dirs = glob.glob(os.path.join(ticker_options_dir, "*"))
            if date_dirs:
                for date_dir in date_dirs:
                    option_files = glob.glob(os.path.join(date_dir, f"{ticker.lower()}_quotedata.csv"))
                    if option_files:
                        options_data_available = True
                        print(f"Found options data for {ticker}")
                        break
            
            if not options_data_available:
                print(f"No options data found for {ticker}")
        else:
            print(f"No options directory found for {ticker}")
        
        # Determine if we should analyze this ticker
        if stock_data_available:
            if options_data_available:
                analysis_status = "Full Analysis"
            else:
                analysis_status = "Stock-Only Analysis"
        else:
            analysis_status = "Blacklisted"
        
        # Add to results dictionary
        results['Ticker'].append(ticker)
        results['Stock_Data_Available'].append(stock_data_available)
        results['Options_Data_Available'].append(options_data_available)
        results['Analysis_Status'].append(analysis_status)
        
        # Step 3: Perform analysis if data is available
        momentum_score = None
        volume_trend = None
        price_trend = None
        rsi_status = None
        macd_signal = None
        bollinger_position = None
        volatility = None
        put_call_ratio = None
        short_term_forecast = None
        sentiment = "N/A"
        confidence = None
        
        if analysis_status != "Blacklisted":
            analysis_results = analyze_ticker(ticker, stock_file_path, 
                                             ticker_options_dir if options_data_available else None)
            
            if analysis_results and 'error' not in analysis_results:
                momentum_score = analysis_results['momentum_score']
                print(f"DEBUG: Analysis results for {ticker} -> {analysis_results}") # <-- Add this
                volume_trend = analysis_results['volume_trend']
                price_trend = analysis_results['price_trend']
                rsi_status = analysis_results['rsi_status']
                macd_signal = analysis_results['macd_signal']
                bollinger_position = analysis_results.get('bollinger_position', "Unavailable")
                volatility = analysis_results['volatility']
                print(f"DEBUG: Analysis results for {ticker} -> {analysis_results}")  # Debug print
                put_call_ratio = analysis_results['put_call_ratio']
                short_term_forecast = analysis_results['short_term_forecast']
                sentiment = analysis_results['sentiment']
                confidence = analysis_results['confidence']
                
                # Save detailed report
                detailed_reports[ticker] = analysis_results['detailed_report']
            else:
                error_msg = analysis_results.get('error', 'Unknown error') if analysis_results else 'Analysis failed'
                sentiment = f"Unable to perform analysis: {error_msg}"
        
        results['Momentum_Score'].append(momentum_score)
        results['Volume_Trend'].append(volume_trend)
        results['Price_Trend'].append(price_trend)
        results['RSI_Status'].append(rsi_status)
        results['MACD_Signal'].append(macd_signal)
        results['Bollinger_Position'].append(bollinger_position)
        results['Volatility'].append(volatility)
        results['Put_Call_Ratio'].append(put_call_ratio)
        results['Short_Term_Forecast'].append(short_term_forecast)
        results['Sentiment'].append(sentiment)
        results['Confidence'].append(confidence)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate and print summary report
    print("\n--- Stock Analysis Summary Report ---")
    print(f"Total tickers analyzed: {len(tickers)}")
    print(f"Tickers with full analysis: {results_df[results_df['Analysis_Status'] == 'Full Analysis'].shape[0]}")
    print(f"Tickers with stock-only analysis: {results_df[results_df['Analysis_Status'] == 'Stock-Only Analysis'].shape[0]}")
    print(f"Blacklisted tickers: {results_df[results_df['Analysis_Status'] == 'Blacklisted'].shape[0]}")
    
    # Generate detailed HTML report
    generate_detailed_html_report(results_df, detailed_reports)
    
    return results_df

def analyze_ticker(ticker, stock_file_path, ticker_options_dir):
    """
    Perform comprehensive directional momentum analysis on a ticker's data.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_file_path (str): Path to historical stock data CSV
        ticker_options_dir (str): Path to options data directory (or None if not available)
    
    Returns:
        dict: Dictionary containing analysis results and detailed report
    """
    try:
        # Load historical price data
        stock_df = pd.read_csv(stock_file_path)
        
        # Convert column names to lowercase for consistency
        stock_df.columns = [col.lower().replace('/', '_') for col in stock_df.columns]
        
        # Check for necessary columns
        price_col = next((col for col in stock_df.columns if 'close' in col or 'last' in col), None)
        volume_col = next((col for col in stock_df.columns if 'volume' in col), None)
        date_col = next((col for col in stock_df.columns if 'date' in col), None)
        
        if not all([price_col, volume_col, date_col]):
            print(f"Missing required columns in {ticker} data")
            return {'error': 'Insufficient Data - Missing required columns'}
        
        # Ensure date is in datetime format and sort
        stock_df[date_col] = pd.to_datetime(stock_df[date_col])
        stock_df = stock_df.sort_values(by=date_col)
        
        # Check if we have enough data points
        if len(stock_df) < 50:
            return {'error': 'Insufficient Data - Need at least 50 data points'}
        
        # Calculate technical indicators
        stock_df['ma_20'] = stock_df[price_col].rolling(window=20).mean()
        stock_df['ma_50'] = stock_df[price_col].rolling(window=50).mean()
        stock_df['ma_200'] = stock_df[price_col].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = stock_df[price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        stock_df['ema_12'] = stock_df[price_col].ewm(span=12, adjust=False).mean()
        stock_df['ema_26'] = stock_df[price_col].ewm(span=26, adjust=False).mean()
        stock_df['macd'] = stock_df['ema_12'] - stock_df['ema_26']
        stock_df['signal'] = stock_df['macd'].ewm(span=9, adjust=False).mean()
        stock_df['macd_hist'] = stock_df['macd'] - stock_df['signal']

        # Volatility (ATR)
        stock_df['high'] = stock_df.get('high', stock_df[price_col])
        stock_df['low'] = stock_df.get('low', stock_df[price_col])
        stock_df['true_range'] = (stock_df['high'] - stock_df['low']).rolling(window=14).mean()
        stock_df['atr_pct'] = stock_df['true_range'] / stock_df[price_col] * 100
        
        # Stationarity test (ADF)
        recent_prices = stock_df[price_col].tail(90).dropna()
        try:
            adf_result = adfuller(recent_prices)
            is_stationary = adf_result[1] < 0.05
        except:
            is_stationary = None

        # Forecast using ARIMA
        short_term_forecast = None
        forecast_pct_change = None
        try:
            if len(recent_prices) >= 30:
                model = ARIMA(recent_prices, order=(5,1,0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=5)
                current_price = recent_prices.iloc[-1]
                predicted_price = forecast[-1]
                forecast_pct_change = (predicted_price - current_price) / current_price * 100

                if forecast_pct_change > 3:
                    short_term_forecast = "Strongly Bullish"
                elif forecast_pct_change > 1:
                    short_term_forecast = "Moderately Bullish"
                elif forecast_pct_change > 0:
                    short_term_forecast = "Slightly Bullish"
                elif forecast_pct_change > -1:
                    short_term_forecast = "Slightly Bearish"
                elif forecast_pct_change > -3:
                    short_term_forecast = "Moderately Bearish"
                else:
                    short_term_forecast = "Strongly Bearish"
        except Exception as e:
            short_term_forecast = "Forecast Unavailable"
            forecast_pct_change = None
            print(f"Error in ARIMA forecast for {ticker}: {e}")

        # Construct final report
        result = {
            'momentum_score': round(forecast_pct_change, 2) if forecast_pct_change else None,
            'volatility': round(stock_df['atr_pct'].iloc[-1], 2) if 'atr_pct' in stock_df else None,
            'short_term_forecast': short_term_forecast,
            'sentiment': "Bullish" if forecast_pct_change and forecast_pct_change > 0 else "Bearish",
            'confidence': "High" if forecast_pct_change and abs(forecast_pct_change) > 3 else "Moderate",
            'volume_trend': None,  # Ensures key exists
            'price_trend': None,   # Ensures key exists
            'rsi_status': None,    # Ensures key exists
            'macd_signal': None,   # Ensures key exists
            'bollinger_position': None,  # 🔧 Fix: Ensures key exists
            'put_call_ratio': None,
            'detailed_report': {
                'ticker': ticker,
                'current_price': stock_df[price_col].iloc[-1],
                'short_term_forecast': short_term_forecast,
                'forecast_pct_change': forecast_pct_change,
                'stationary_prices': is_stationary,
                'arima_parameters': '(5,1,0)'
            }
        }






        return result  # ✅ Correctly inside try block

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return {'error': str(e)}  # ✅ Properly catches all failures



def get_key_strengths(price_trend_score, ma_cross_score, rsi_score, macd_score, 
                     volume_score, bb_score, high_low_score, options_score, forecast_score):
    """
    Identify the key strength factors based on individual component scores.
    Returns a list of strength descriptions.
    """
    strengths = []
    
    if price_trend_score > 0:
        strengths.append("Positive price trend")
    
    if ma_cross_score > 0:
        strengths.append("Bullish moving average crossovers")
    
    if rsi_score > 0:
        strengths.append("Favorable RSI conditions")
    
    if macd_score > 0:
        strengths.append("Bullish MACD signal")
    
    if volume_score > 0:
        strengths.append("Strong volume confirmation")
    
    if bb_score > 0:
        strengths.append("Favorable Bollinger Band position")
    
    if high_low_score > 0:
        strengths.append("Strong position relative to 52-week range")
    
    if options_score > 0:
        strengths.append("Positive options sentiment")
    
    if forecast_score > 0:
        strengths.append("Bullish short-term forecast")
    
    return strengths if strengths else ["No significant strengths identified"]

def get_key_weaknesses(price_trend_score, ma_cross_score, rsi_score, macd_score, 
                      volume_score, bb_score, high_low_score, options_score, forecast_score):
    """
    Identify the key weakness factors based on individual component scores.
    Returns a list of weakness descriptions.
    """
    weaknesses = []
    
    if price_trend_score < 0:
        weaknesses.append("Negative price trend")
    
    if ma_cross_score < 0:
        weaknesses.append("Bearish moving average crossovers")
    
    if rsi_score < 0:
        weaknesses.append("Unfavorable RSI conditions")
    
    if macd_score < 0:
        weaknesses.append("Bearish MACD signal")
    
    if volume_score < 0:
        weaknesses.append("Weak volume confirmation")
    
    if bb_score < 0:
        weaknesses.append("Unfavorable Bollinger Band position")
    
    if high_low_score < 0:
        weaknesses.append("Weak position relative to 52-week range")
    
    if options_score < 0:
        weaknesses.append("Negative options sentiment")
    
    if forecast_score < 0:
        weaknesses.append("Bearish short-term forecast")
    
    return weaknesses if weaknesses else ["No significant weaknesses identified"]

def analyze_options_data(ticker_options_dir, ticker):
    """
    Analyze options data to extract insights about market sentiment.
    
    Args:
        ticker_options_dir (str): Path to the directory containing options data
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary containing options analysis results
    """
    try:
        # Find the most recent options data directory
        date_dirs = sorted(glob.glob(os.path.join(ticker_options_dir, "*")), reverse=True)
        
        if not date_dirs:
            return {'error': 'No options data directories found'}
        
        recent_dir = date_dirs[0]
        option_files = glob.glob(os.path.join(recent_dir, f"{ticker.lower()}_quotedata.csv"))
        
        if not option_files:
            return {'error': 'No options data files found'}
        
        # Load options data
        options_df = pd.read_csv(option_files[0])
        
        # Convert column names to lowercase for consistency
        options_df.columns = [col.lower() for col in options_df.columns]
        
        # Extract key options metrics
        total_call_volume = 0
        total_put_volume = 0
        near_calls = 0
        near_puts = 0
        call_oi_sum = 0
        put_oi_sum = 0
        
        # Check for expected columns
        expected_cols = ['option_type', 'volume', 'open_interest', 'strike', 'expiration']
        column_map = {}
        
        for expected in expected_cols:
            for col in options_df.columns:
                if expected in col:
                    column_map[expected] = col
                    break
        
        # If we're missing critical columns, return error
        missing_cols = [col for col in expected_cols if col not in column_map]
        if missing_cols:
            return {'error': f'Missing critical options columns: {missing_cols}'}
            
        # Process options data
        if 'last_price' not in options_df.columns and 'last_trade_price' in options_df.columns:
            options_df['last_price'] = options_df['last_trade_price']
            
        # Calculate put-call ratio based on volume
        call_data = options_df[options_df[column_map['option_type']].str.lower().str.contains('call')]
        put_data = options_df[options_df[column_map['option_type']].str.lower().str.contains('put')]
        
        if not call_data.empty:
            total_call_volume = call_data[column_map['volume']].sum()
            call_oi_sum = call_data[column_map['open_interest']].sum() if column_map['open_interest'] in call_data.columns else 0
        
        if not put_data.empty:
            total_put_volume = put_data[column_map['volume']].sum()
            put_oi_sum = put_data[column_map['open_interest']].sum() if column_map['open_interest'] in put_data.columns else 0
        
        # Calculate put-call ratios
        volume_put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        oi_put_call_ratio = put_oi_sum / call_oi_sum if call_oi_sum > 0 else 0
        
        # Determine sentiment score based on put-call ratio
        # Typically a high put-call ratio (> 1) is bearish, low (< 0.7) is bullish
        if volume_put_call_ratio > 1.2:
            sentiment_score = -3  # Very bearish
        elif volume_put_call_ratio > 1:
            sentiment_score = -2  # Bearish
        elif volume_put_call_ratio > 0.8:
            sentiment_score = -1  # Slightly bearish
        elif volume_put_call_ratio > 0.6:
            sentiment_score = 1   # Slightly bullish
        elif volume_put_call_ratio > 0.4:
            sentiment_score = 2   # Bullish
        else:
            sentiment_score = 3   # Very bullish
        
        # Return options analysis
        return {
            'put_call_ratio': round(volume_put_call_ratio, 2),
            'oi_put_call_ratio': round(oi_put_call_ratio, 2),
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'call_open_interest': call_oi_sum,
            'put_open_interest': put_oi_sum,
            'sentiment_score': sentiment_score
        }
    except Exception as e:
        print(f"Error analyzing options data for {ticker}: {e}")
        return {'error': str(e)}

def generate_detailed_html_report(results_df, detailed_reports):
    """
    Generate a detailed HTML report with analysis results and visualizations.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing summary results
        detailed_reports (dict): Dictionary of detailed analysis reports by ticker
    """
    try:
        # Create HTML file
        with open('stock_analysis_report.html', 'w') as f:
            # Write HTML header and style
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stock Analysis Report</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        color: #333;
                    }
                    h1, h2, h3 {
                        color: #0066cc;
                    }
                    table {
                        border-collapse: collapse;
                        width: 100%;
                        margin-bottom: 20px;
                    }
                    th, td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                    tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                    .bullish {
                        color: green;
                        font-weight: bold;
                    }
                    .bearish {
                        color: red;
                        font-weight: bold;
                    }
                    .neutral {
                        color: #666;
                    }
                    .section {
                        margin-bottom: 30px;
                        border: 1px solid #eee;
                        padding: 15px;
                        border-radius: 5px;
                    }
                    .ticker-card {
                        border: 1px solid #ddd;
                        margin-bottom: 20px;
                        padding: 15px;
                        border-radius: 5px;
                    }
                    .summary-box {
                        background-color: #f8f8f8;
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 15px;
                    }
                    .indicator-group {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                    }
                    .indicator {
                        flex: 1;
                        min-width: 250px;
                        border: 1px solid #eee;
                        padding: 10px;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
                <h1>Stock Analysis Report</h1>
                <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            """)
            
            # Add summary section
            f.write("""
                <div class="section">
                    <h2>Analysis Summary</h2>
                    <table>
                        <tr>
                            <th>Total Tickers Analyzed</th>
                            <td>""" + str(len(results_df)) + """</td>
                        </tr>
                        <tr>
                            <th>Tickers with Full Analysis</th>
                            <td>""" + str(results_df[results_df['Analysis_Status'] == 'Full Analysis'].shape[0]) + """</td>
                        </tr>
                        <tr>
                            <th>Tickers with Stock-Only Analysis</th>
                            <td>""" + str(results_df[results_df['Analysis_Status'] == 'Stock-Only Analysis'].shape[0]) + """</td>
                        </tr>
                        <tr>
                            <th>Blacklisted Tickers</th>
                            <td>""" + str(results_df[results_df['Analysis_Status'] == 'Blacklisted'].shape[0]) + """</td>
                        </tr>
                    </table>
                </div>
            """)
            
            # Add sentiment distribution
            sentiment_counts = results_df['Sentiment'].value_counts()
            f.write("""
                <div class="section">
                    <h2>Sentiment Distribution</h2>
                    <table>
                        <tr>
                            <th>Sentiment</th>
                            <th>Count</th>
                        </tr>
            """)
            
            for sentiment, count in sentiment_counts.items():
                sentiment_class = ''
                if 'Bullish' in str(sentiment):
                    sentiment_class = 'bullish'
                elif 'Bearish' in str(sentiment):
                    sentiment_class = 'bearish'
                else:
                    sentiment_class = 'neutral'
                    
                f.write(f"""
                        <tr>
                            <td class="{sentiment_class}">{sentiment}</td>
                            <td>{count}</td>
                        </tr>
                """)
            
            f.write("""
                    </table>
                </div>
            """)
            
            # Add top bullish and bearish stocks
            if 'Momentum_Score' in results_df.columns:
                valid_scores = results_df[results_df['Momentum_Score'].notna()]
                
                if len(valid_scores) > 0:
                    top_bullish = valid_scores.nlargest(5, 'Momentum_Score')
                    top_bearish = valid_scores.nsmallest(5, 'Momentum_Score')
                    
                    f.write("""
                        <div class="section">
                            <h2>Top Bullish Stocks</h2>
                            <table>
                                <tr>
                                    <th>Ticker</th>
                                    <th>Momentum Score</th>
                                    <th>Sentiment</th>
                                    <th>Confidence</th>
                                </tr>
                    """)
                    
                    for _, row in top_bullish.iterrows():
                        f.write(f"""
                                <tr>
                                    <td>{row['Ticker']}</td>
                                    <td class="bullish">{row['Momentum_Score']}</td>
                                    <td class="bullish">{row['Sentiment']}</td>
                                    <td>{row['Confidence']}</td>
                                </tr>
                        """)
                    
                    f.write("""
                            </table>
                        </div>
                        
                        <div class="section">
                            <h2>Top Bearish Stocks</h2>
                            <table>
                                <tr>
                                    <th>Ticker</th>
                                    <th>Momentum Score</th>
                                    <th>Sentiment</th>
                                    <th>Confidence</th>
                                </tr>
                    """)
                    
                    for _, row in top_bearish.iterrows():
                        f.write(f"""
                                <tr>
                                    <td>{row['Ticker']}</td>
                                    <td class="bearish">{row['Momentum_Score']}</td>
                                    <td class="bearish">{row['Sentiment']}</td>
                                    <td>{row['Confidence']}</td>
                                </tr>
                        """)
                    
                    f.write("""
                            </table>
                        </div>
                    """)
            
            # Add detailed ticker reports
            f.write("""
                <div class="section">
                    <h2>Detailed Ticker Analysis</h2>
            """)
            
            for ticker, report in detailed_reports.items():
                if not report:
                    continue
                
                # Determine sentiment class
                sentiment = report['summary']['sentiment']
                sentiment_class = ''
                if 'Bullish' in sentiment:
                    sentiment_class = 'bullish'
                elif 'Bearish' in sentiment:
                    sentiment_class = 'bearish'
                else:
                    sentiment_class = 'neutral'
                
                f.write(f"""
                    <div class="ticker-card">
                        <h3>{ticker} - <span class="{sentiment_class}">{sentiment}</span></h3>
                        
                        <div class="summary-box">
                            <h4>Summary</h4>
                            <table>
                                <tr>
                                    <th>Current Price</th>
                                    <td>${report['current_price']:.2f}</td>
                                    <th>Momentum Score</th>
                                    <td class="{sentiment_class}">{report['summary']['momentum_score']}</td>
                                </tr>
                                <tr>
                                    <th>Analysis Date</th>
                                    <td>{report['analysis_date']}</td>
                                    <th>Confidence</th>
                                    <td>{report['summary']['confidence']}</td>
                                </tr>
                                <tr>
                                    <th>Short-Term Forecast</th>
                                    <td>{report['summary']['short_term_forecast']}</td>
                                    <th></th>
                                    <td></td>
                                </tr>
                            </table>
                        </div>
                        
                        <div class="indicator-group">
                            <div class="indicator">
                                <h4>Key Strengths</h4>
                                <ul>
                """)
                
                for strength in report['summary']['key_strengths']:
                    f.write(f"<li>{strength}</li>")
                
                f.write("""
                                </ul>
                            </div>
                            <div class="indicator">
                                <h4>Key Weaknesses</h4>
                                <ul>
                """)
                
                for weakness in report['summary']['key_weaknesses']:
                    f.write(f"<li>{weakness}</li>")
                
                f.write("""
                                </ul>
                            </div>
                        </div>
                        
                        <h4>Technical Indicators</h4>
                        <div class="indicator-group">
                            <div class="indicator">
                                <h5>Price Action</h5>
                                <table>
                """)
                
                price_data = report['technical_indicators']['price_action']
                for key, value in price_data.items():
                    if key in ['change_1d', 'change_5d', 'change_20d', 'pct_from_52w_high', 'pct_from_52w_low']:
                        if value is not None:
                            css_class = 'bullish' if value > 0 else 'bearish'
                            f.write(f"""
                                    <tr>
                                        <th>{key.replace('_', ' ').title()}</th>
                                        <td class="{css_class}">{value:.2f}%</td>
                                    </tr>
                            """)
                    else:
                        if value is not None and value != 'N/A':
                            f.write(f"""
                                    <tr>
                                        <th>{key.replace('_', ' ').title()}</th>
                                        <td>{value if isinstance(value, str) else f'${value:.2f}' if 'price' in key.lower() or 'high' in key.lower() or 'low' in key.lower() else value}</td>
                                    </tr>
                            """)
                
                f.write("""
                                </table>
                            </div>
                            <div class="indicator">
                                <h5>Momentum Indicators</h5>
                                <table>
                """)
                
                momentum_data = report['technical_indicators']['momentum_indicators']
                for key, value in momentum_data.items():
                    if value is not None:
                        css_class = ''
                        if key == 'rsi':
                            if value > 70:
                                css_class = 'bearish'
                            elif value < 30:
                                css_class = 'bullish'
                        elif key == 'macd_direction':
                            if 'Bullish' in str(value):
                                css_class = 'bullish'
                            elif 'Bearish' in str(value):
                                css_class = 'bearish'
                                
                        f.write(f"""
                                <tr>
                                    <th>{key.replace('_', ' ').title()}</th>
                                    <td class="{css_class}">{value}</td>
                                </tr>
                        """)
                
                f.write("""
                                </table>
                            </div>
                        </div>
                        
                        <div class="indicator-group">
                            <div class="indicator">
                                <h5>Volatility Indicators</h5>
                                <table>
                """)
                
                volatility_data = report['technical_indicators']['volatility_indicators']
                for key, value in volatility_data.items():
                    if value is not None and value != 'N/A':
                        f.write(f"""
                                <tr>
                                    <th>{key.replace('_', ' ').title()}</th>
                                    <td>{value if isinstance(value, str) else f'${value:.2f}' if 'price' in key.lower() else f'{value:.2f}' if isinstance(value, float) else value}</td>
                                </tr>
                        """)
                
                f.write("""
                                </table>
                            </div>
                            <div class="indicator">
                                <h5>Volume Analysis</h5>
                                <table>
                """)
                
                volume_data = report['technical_indicators']['volume_analysis']
                for key, value in volume_data.items():
                    if value is not None:
                        f.write(f"""
                                <tr>
                                    <th>{key.replace('_', ' ').title()}</th>
                                    <td>{value if isinstance(value, str) else f'{value:,.0f}' if 'volume' in key.lower() else f'{value:.2f}'}</td>
                                </tr>
                        """)
                
                f.write("""
                                </table>
                            </div>
                        </div>
                    </div>
                """)
            
            f.write("""
                </div>
            </body>
            </html>
            """)
        
        print(f"HTML report generated: stock_analysis_report.html")
    except Exception as e:
        print(f"Error generating HTML report: {e}")

# Main execution block
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze stock tickers with directional momentum')
    parser.add_argument('--tickers', required=True, help='Path to CSV file containing stock tickers')
    parser.add_argument('--stocks', required=True, help='Path to directory containing historical stock data')
    parser.add_argument('--options', required=True, help='Path to directory containing options data')
    
    args = parser.parse_args()
    
    # Run the analysis
    results = analyze_stock_tickers(args.tickers, args.stocks, args.options)
    
    if results is not None:
        # Save results to CSV
        results.to_csv('stock_analysis_results.csv', index=False)
        print(f"Results saved to stock_analysis_results.csv")
