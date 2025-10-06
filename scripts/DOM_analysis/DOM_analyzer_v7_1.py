#!/usr/bin/env python3
"""
CBOE Direct Book Viewer Data Collector v7.1
============================================

A comprehensive real-time DOM (Depth of Market) data collector with live visualization.

Features:
- Direct access to CBOE book viewer
- Real-time data collection with customizable parameters
- Live heatmap visualization via HTTP server
- Comprehensive error handling and logging
- User-friendly interface with validation
- Configurable data collection settings

Author: AI Assistant
Version: 7.1
Date: 2025-01-31
"""

import csv
import json
import time
import os
import shutil
import threading
from datetime import datetime, timedelta
from collections import deque
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Selenium imports for web scraping
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Data visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class DOMDataCollector:
    """
    Main class for collecting DOM (Depth of Market) data from CBOE.
    
    This class handles:
    - Web scraping of CBOE book viewer
    - Real-time data collection
    - Data storage and management
    - Error handling and recovery
    """
    
    def __init__(self, ticker="SPY", poll_interval=3, max_cycles=100, 
                 output_file=None, enable_visualization=True, port=8080):
        """
        Initialize the DOM data collector.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'SPY', 'AAPL')
            poll_interval (int): Polling interval in seconds
            max_cycles (int): Maximum number of data collection cycles
            output_file (str): CSV output file path (auto-generated if None)
            enable_visualization (bool): Enable live visualization server
            port (int): HTTP server port for visualization
        """
        self.ticker = ticker.upper()
        self.poll_interval = poll_interval
        self.max_cycles = max_cycles
        self.enable_visualization = enable_visualization
        self.port = port
        
        # Data storage
        self.data_buffer = deque(maxlen=1000)  # Keep last 1000 data points
        self.collected_data = []
        self.is_collecting = False
        
        # File management
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"dom_data_{self.ticker}_{timestamp}.csv"
        else:
            self.output_file = output_file
            
        # Browser setup
        self.driver = None
        self.setup_browser()
        
        # Visualization server
        if self.enable_visualization:
            self.start_visualization_server()
    
    def setup_browser(self):
        """Configure Chrome browser with optimal settings for web scraping."""
        print("🔧 Setting up Chrome browser...")
        
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Additional stability options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # Strategy:
        # 1) Try Selenium Manager (built into Selenium 4.6+) to resolve the correct driver automatically
        # 2) Fallback to webdriver-manager
        # 3) If we hit a corrupted driver (e.g., WinError 193), clear cache and retry once
        try:
            # Prefer Selenium Manager (no explicit Service path)
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            print("✅ Browser setup completed successfully (Selenium Manager)")
            return
        except Exception as first_error:
            print(f"⚠️ Selenium Manager failed: {first_error}")
        
        # Fallback: webdriver-manager
        try:
            driver_path = ChromeDriverManager().install()
            self.driver = webdriver.Chrome(
                service=Service(driver_path),
                options=options
            )
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            print("✅ Browser setup completed successfully (WebDriver Manager)")
            return
        except Exception as second_error:
            print(f"⚠️ WebDriver Manager failed: {second_error}")
            
            # Attempt to clear webdriver-manager cache and retry once
            try:
                wdm_cache = os.path.join(os.path.expanduser("~"), ".wdm")
                if os.path.isdir(wdm_cache):
                    shutil.rmtree(wdm_cache, ignore_errors=True)
                    print("🧹 Cleared webdriver cache; retrying driver install...")
                driver_path = ChromeDriverManager().install()
                self.driver = webdriver.Chrome(
                    service=Service(driver_path),
                    options=options
                )
                self.driver.execute_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                )
                print("✅ Browser setup completed successfully after cache reset")
                return
            except Exception as third_error:
                print(f"❌ Retry after cache reset failed: {third_error}")
                raise RuntimeError("Chrome browser setup failed. See logs above for details.")
    
    def construct_url(self):
        """Construct the direct CBOE book viewer URL for the ticker."""
        return f"https://www.cboe.com/us/equities/market_statistics/book/{self.ticker.lower()}/"
    
    def validate_ticker(self):
        """Validate that the ticker symbol is reasonable."""
        if not self.ticker or len(self.ticker) > 5:
            raise ValueError("Ticker symbol must be 1-5 characters")
        
        # Common ticker validation
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.')
        if not all(c in valid_chars for c in self.ticker):
            raise ValueError("Ticker contains invalid characters")
        
        return True
    
    def load_book_viewer(self):
        """Load the CBOE book viewer page for the specified ticker."""
        url = self.construct_url()
        print(f"🌐 Loading book viewer: {url}")
        
        try:
            self.driver.get(url)
            print(f"📊 Loading book viewer for {self.ticker}...")
            
            # Wait for page to load
            time.sleep(5)
            
            # Wait for book viewer to be present
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "book-viewer"))
            )
            
            print(f"✅ Book viewer loaded successfully for {self.ticker}")
            
            # Wait for data to populate
            WebDriverWait(self.driver, 10).until(
                lambda d: len(d.find_elements(By.CLASS_NAME, "book-viewer__bid-price")) > 0 or
                         len(d.find_elements(By.CLASS_NAME, "book-viewer__ask-price")) > 0
            )
            
            print(f"📈 Data is now available for {self.ticker}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load book viewer: {e}")
            return False
    
    def extract_dom_data(self):
        """
        Extract DOM data from the current page state.
        
        Returns:
            dict: Dictionary containing bid/ask data and metadata
        """
        try:
            # Find bid data
            bid_prices = self.driver.find_elements(By.CLASS_NAME, "book-viewer__bid-price")
            bid_sizes = self.driver.find_elements(By.CLASS_NAME, "book-viewer__bid-shares")
            
            # Find ask data
            ask_prices = self.driver.find_elements(By.CLASS_NAME, "book-viewer__ask-price")
            ask_sizes = self.driver.find_elements(By.CLASS_NAME, "book-viewer__ask-shares")
            
            # Find trade data
            trade_prices = self.driver.find_elements(By.CLASS_NAME, "book-viewer__trades-price")
            trade_sizes = self.driver.find_elements(By.CLASS_NAME, "book-viewer__trades-shares")
            
            # Get last updated timestamp
            last_updated = ""
            try:
                last_updated_elem = self.driver.find_element(By.CLASS_NAME, "last-updated")
                last_updated = last_updated_elem.text.strip()
            except:
                pass
            
            # Extract top of book data
            bid_price = bid_prices[0].text.strip() if bid_prices else ""
            bid_size = bid_sizes[0].text.strip() if bid_sizes else ""
            ask_price = ask_prices[0].text.strip() if ask_prices else ""
            ask_size = ask_sizes[0].text.strip() if ask_sizes else ""
            trade_price = trade_prices[0].text.strip() if trade_prices else ""
            trade_size = trade_sizes[0].text.strip() if trade_sizes else ""
            
            # Create data record
            data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'symbol': self.ticker,
                'bid_price': bid_price,
                'bid_size': bid_size,
                'ask_price': ask_price,
                'ask_size': ask_size,
                'trade_price': trade_price,
                'trade_size': trade_size,
                'last_updated': last_updated,
                'spread': self._calculate_spread(bid_price, ask_price),
                'mid_price': self._calculate_mid_price(bid_price, ask_price)
            }
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error extracting DOM data: {e}")
            return None
    
    def _calculate_spread(self, bid_price, ask_price):
        """Calculate the bid-ask spread."""
        try:
            bid = float(bid_price) if bid_price else 0
            ask = float(ask_price) if ask_price else 0
            return round(ask - bid, 4) if bid > 0 and ask > 0 else 0
        except:
            return 0
    
    def _calculate_mid_price(self, bid_price, ask_price):
        """Calculate the mid price."""
        try:
            bid = float(bid_price) if bid_price else 0
            ask = float(ask_price) if ask_price else 0
            return round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else 0
        except:
            return 0
    
    def save_to_csv(self, data):
        """Save data to CSV file."""
        try:
            file_exists = os.path.exists(self.output_file)
            
            with open(self.output_file, mode="a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writerow([
                        "timestamp", "symbol", "bid_price", "bid_size",
                        "ask_price", "ask_size", "trade_price", "trade_size",
                        "last_updated", "spread", "mid_price"
                    ])
                    print(f"📝 Created new data file: {self.output_file}")
                
                # Write data row
                writer.writerow([
                    data['timestamp'], data['symbol'], data['bid_price'],
                    data['bid_size'], data['ask_price'], data['ask_size'],
                    data['trade_price'], data['trade_size'], data['last_updated'],
                    data['spread'], data['mid_price']
                ])
                
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")
    
    def collect_data(self):
        """Main data collection loop."""
        print(f"🔄 Starting data collection for {self.ticker}")
        print(f"⏱️  Polling every {self.poll_interval} seconds for {self.max_cycles} cycles")
        
        # Load the book viewer
        if not self.load_book_viewer():
            print("❌ Failed to load book viewer. Exiting.")
            return
        
        # Create CSV file header
        self.save_to_csv({
            'timestamp': 'timestamp', 'symbol': 'symbol', 'bid_price': 'bid_price',
            'bid_size': 'bid_size', 'ask_price': 'ask_price', 'ask_size': 'ask_size',
            'trade_price': 'trade_price', 'trade_size': 'trade_size',
            'last_updated': 'last_updated', 'spread': 'spread', 'mid_price': 'mid_price'
        })
        
        self.is_collecting = True
        cycle = 0
        
        try:
            while cycle < self.max_cycles and self.is_collecting:
                cycle += 1
                
                # Extract data
                data = self.extract_dom_data()
                
                if data:
                    # Store data
                    self.collected_data.append(data)
                    self.data_buffer.append(data)
                    
                    # Save to CSV
                    self.save_to_csv(data)
                    
                    # Log progress
                    print(f"📈 Cycle {cycle}/{self.max_cycles}: {self.ticker} - "
                          f"Bid: {data['bid_price']} ({data['bid_size']}) | "
                          f"Ask: {data['ask_price']} ({data['ask_size']}) | "
                          f"Spread: {data['spread']}")
                
                # Wait for next poll
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Data collection stopped by user")
        except Exception as e:
            print(f"❌ Error during data collection: {e}")
        finally:
            self.is_collecting = False
            print(f"✅ Data collection completed. Total cycles: {cycle}")
            print(f"📁 Data saved to: {self.output_file}")
    
    def start_visualization_server(self):
        """Start the HTTP server for live visualization."""
        if not self.enable_visualization:
            return
        
        def start_server():
            try:
                server = HTTPServer(('localhost', self.port), VisualizationHandler)
                server.data_collector = self
                print(f"🌐 Live visualization server started on http://localhost:{self.port}")
                server.serve_forever()
            except Exception as e:
                print(f"❌ Failed to start visualization server: {e}")
        
        # Start server in background thread
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open(f'http://localhost:{self.port}')
                print(f"🌐 Browser opened to visualization dashboard")
            except:
                print(f"💡 Please manually open: http://localhost:{self.port}")
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            print("🔚 Browser session closed")
        
        self.is_collecting = False


class VisualizationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for serving live visualization."""
    
    def do_GET(self):
        """Handle GET requests for visualization."""
        try:
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                
                html_content = self.generate_dashboard()
                self.wfile.write(html_content.encode('utf-8'))
                
            elif self.path == '/data':
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                data = self.get_latest_data()
                self.wfile.write(json.dumps(data).encode('utf-8'))
                
            else:
                self.send_response(404)
                self.end_headers()
                
        except Exception as e:
            print(f"❌ HTTP server error: {e}")
            self.send_response(500)
            self.end_headers()
    
    def generate_dashboard(self):
        """Generate the main dashboard HTML."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset=\"UTF-8\">
            <title>DOM Data Visualization - {self.server.data_collector.ticker}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .header h1 {{ color: #333; margin-bottom: 10px; }}
                .status {{ padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                .status.collecting {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
                .status.stopped {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
                .chart-container {{ margin-bottom: 30px; }}
                .controls {{ margin-bottom: 20px; }}
                button {{ padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }}
                .btn-primary {{ background-color: #007bff; color: white; }}
                .btn-secondary {{ background-color: #6c757d; color: white; }}
                .btn-danger {{ background-color: #dc3545; color: white; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .data-table th, .data-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                .data-table th {{ background-color: #f8f9fa; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 DOM Data Visualization</h1>
                    <h2>{self.server.data_collector.ticker} - Real-time Market Depth</h2>
                </div>
                
                <div id="status" class="status collecting">
                    <strong>Status:</strong> <span id="status-text">Collecting data...</span>
                </div>
                
                <div class="controls">
                    <button class="btn-primary" onclick="refreshData()">🔄 Refresh</button>
                    <button class="btn-secondary" onclick="toggleAutoRefresh()">⏸️ Toggle Auto-refresh</button>
                    <button class="btn-danger" onclick="stopCollection()">⏹️ Stop Collection</button>
                </div>
                
                <div class="chart-container">
                    <div id="price-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <div id="heatmap-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <div id="spread-chart" style="height: 300px;"></div>
                </div>
                
                <div>
                    <h3>📋 Latest Data</h3>
                    <table class="data-table" id="data-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Bid Price</th>
                                <th>Bid Size</th>
                                <th>Ask Price</th>
                                <th>Ask Size</th>
                                <th>Spread</th>
                                <th>Mid Price</th>
                            </tr>
                        </thead>
                        <tbody id="data-body">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                let autoRefresh = true;
                let priceChart, heatmapChart, spreadChart;
                
                // Initialize charts
                function initCharts() {{
                    // Price chart
                    priceChart = document.getElementById('price-chart');
                    
                    // Heatmap chart
                    heatmapChart = document.getElementById('heatmap-chart');
                    
                    // Spread chart
                    spreadChart = document.getElementById('spread-chart');
                    
                    updateCharts();
                }}
                
                // Update charts with latest data
                function updateCharts() {{
                    fetch('/data')
                        .then(response => response.json())
                        .then(data => {{
                            updateStatus(data.is_collecting);
                            updateDataTable(data.latest_data);
                            updatePriceChart(data.price_data);
                            updateHeatmapChart(data.heatmap_data);
                            updateSpreadChart(data.spread_data);
                        }})
                        .catch(error => console.error('Error fetching data:', error));
                }}
                
                // Update status display
                function updateStatus(isCollecting) {{
                    const statusDiv = document.getElementById('status');
                    const statusText = document.getElementById('status-text');
                    
                    if (isCollecting) {{
                        statusDiv.className = 'status collecting';
                        statusText.textContent = 'Collecting data...';
                    }} else {{
                        statusDiv.className = 'status stopped';
                        statusText.textContent = 'Data collection stopped';
                    }}
                }}
                
                // Update data table
                function updateDataTable(data) {{
                    const tbody = document.getElementById('data-body');
                    tbody.innerHTML = '';
                    
                    if (data && data.length > 0) {{
                        const latest = data[data.length - 1];
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${{latest.timestamp}}</td>
                            <td>${{latest.bid_price}}</td>
                            <td>${{latest.bid_size}}</td>
                            <td>${{latest.ask_price}}</td>
                            <td>${{latest.ask_size}}</td>
                            <td>${{latest.spread}}</td>
                            <td>${{latest.mid_price}}</td>
                        `;
                    }}
                }}
                
                // Update price chart
                function updatePriceChart(data) {{
                    if (!data || data.length === 0) return;
                    
                    const traces = [
                        {{
                            x: data.map(d => d.timestamp),
                            y: data.map(d => parseFloat(d.bid_price) || 0),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Bid Price',
                            line: {{color: 'green'}},
                            marker: {{size: 6}}
                        }},
                        {{
                            x: data.map(d => d.timestamp),
                            y: data.map(d => parseFloat(d.ask_price) || 0),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Ask Price',
                            line: {{color: 'red'}},
                            marker: {{size: 6}}
                        }},
                        {{
                            x: data.map(d => d.timestamp),
                            y: data.map(d => parseFloat(d.mid_price) || 0),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Mid Price',
                            line: {{color: 'blue'}},
                            marker: {{size: 4}}
                        }}
                    ];
                    
                    const layout = {{
                        title: 'Real-time Price Movement',
                        xaxis: {{title: 'Time'}},
                        yaxis: {{title: 'Price ($)'}},
                        height: 400,
                        showlegend: true
                    }};
                    
                    Plotly.newPlot(priceChart, traces, layout);
                }}
                
                // Update heatmap chart
                function updateHeatmapChart(data) {{
                    if (!data || data.length === 0) return;
                    
                    // Create heatmap data from bid/ask sizes
                    const timestamps = data.map(d => d.timestamp);
                    const bidSizes = data.map(d => parseInt(d.bid_size.replace(/,/g, '')) || 0);
                    const askSizes = data.map(d => parseInt(d.ask_size.replace(/,/g, '')) || 0);
                    
                    const heatmapData = [
                        {{
                            z: [bidSizes, askSizes],
                            x: timestamps,
                            y: ['Bid Size', 'Ask Size'],
                            type: 'heatmap',
                            colorscale: 'Viridis'
                        }}
                    ];
                    
                    const layout = {{
                        title: 'Order Size Heatmap',
                        xaxis: {{title: 'Time'}},
                        yaxis: {{title: 'Order Type'}},
                        height: 400
                    }};
                    
                    Plotly.newPlot(heatmapChart, heatmapData, layout);
                }}
                
                // Update spread chart
                function updateSpreadChart(data) {{
                    if (!data || data.length === 0) return;
                    
                    const trace = {{
                        x: data.map(d => d.timestamp),
                        y: data.map(d => parseFloat(d.spread) || 0),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Bid-Ask Spread',
                        line: {{color: 'orange'}},
                        marker: {{size: 6}}
                    }};
                    
                    const layout = {{
                        title: 'Bid-Ask Spread Over Time',
                        xaxis: {{title: 'Time'}},
                        yaxis: {{title: 'Spread ($)'}},
                        height: 300
                    }};
                    
                    Plotly.newPlot(spreadChart, [trace], layout);
                }}
                
                // Control functions
                function refreshData() {{
                    updateCharts();
                }}
                
                function toggleAutoRefresh() {{
                    autoRefresh = !autoRefresh;
                    const btn = event.target;
                    if (autoRefresh) {{
                        btn.textContent = '⏸️ Pause Auto-refresh';
                        btn.className = 'btn-secondary';
                    }} else {{
                        btn.textContent = '▶️ Resume Auto-refresh';
                        btn.className = 'btn-primary';
                    }}
                }}
                
                function stopCollection() {{
                    if (confirm('Are you sure you want to stop data collection?')) {{
                        fetch('/stop', {{method: 'POST'}})
                            .then(() => updateCharts());
                    }}
                }}
                
                // Auto-refresh every 3 seconds
                setInterval(() => {{
                    if (autoRefresh) {{
                        updateCharts();
                    }}
                }}, 3000);
                
                // Initialize on page load
                window.onload = function() {{
                    initCharts();
                }};
            </script>
        </body>
        </html>
        """
    
    def get_latest_data(self):
        """Get the latest data for visualization."""
        collector = self.server.data_collector
        
        # Get recent data for charts
        recent_data = list(collector.data_buffer)[-50:] if collector.data_buffer else []
        
        # Prepare data for visualization
        price_data = recent_data
        heatmap_data = recent_data
        spread_data = recent_data
        
        return {
            'is_collecting': collector.is_collecting,
            'latest_data': recent_data,
            'price_data': price_data,
            'heatmap_data': heatmap_data,
            'spread_data': spread_data
        }


def get_user_input():
    """Get user input for configuration."""
    print("🚀 CBOE Direct Book Viewer Data Collector v7.1")
    print("=" * 60)
    
    # Get ticker symbol
    while True:
        ticker = input("Enter ticker symbol (e.g., SPY, AAPL, TSLA): ").strip().upper()
        if ticker and len(ticker) <= 5:
            break
        else:
            print("❌ Please enter a valid ticker symbol (1-5 characters)")
    
    # Get polling settings
    try:
        poll_interval = int(input("Enter polling interval in seconds (default 3): ") or "3")
        max_cycles = int(input("Enter number of polling cycles (default 100): ") or "100")
    except ValueError:
        print("⚠️ Using default values due to invalid input")
        poll_interval, max_cycles = 3, 100
    
    # Get visualization preference
    viz_choice = input("Enable live visualization? (y/n, default y): ").strip().lower()
    enable_visualization = viz_choice != 'n'
    
    # Get port for visualization
    port = 8080
    if enable_visualization:
        try:
            port_input = input("Enter HTTP server port (default 8080): ") or "8080"
            port = int(port_input)
        except ValueError:
            print("⚠️ Using default port 8080")
    
    return ticker, poll_interval, max_cycles, enable_visualization, port


def main():
    """Main function to run the DOM data collector."""
    try:
        # Get user configuration
        ticker, poll_interval, max_cycles, enable_visualization, port = get_user_input()
        
        print(f"\n🎯 Configuration:")
        print(f"   Ticker: {ticker}")
        print(f"   Poll Interval: {poll_interval} seconds")
        print(f"   Max Cycles: {max_cycles}")
        print(f"   Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
        if enable_visualization:
            print(f"   Server Port: {port}")
        print(f"\n⏳ Starting data collection...")
        
        # Create and run data collector
        collector = DOMDataCollector(
            ticker=ticker,
            poll_interval=poll_interval,
            max_cycles=max_cycles,
            enable_visualization=enable_visualization,
            port=port
        )
        
        # Start data collection
        collector.collect_data()
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        if 'collector' in locals():
            collector.cleanup()


if __name__ == "__main__":
    main() 