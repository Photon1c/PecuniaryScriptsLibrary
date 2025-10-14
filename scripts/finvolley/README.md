# Gamma Flip Ping Pong 🏓

A fun, interactive visualization of gamma flip dynamics in the stock market, gamified as a ping pong simulation! Watch real market prices battle it out in a unique backtesting environment.

> **✨ New Features:** Position tracking with P&L, performance reports, CSV export, and data inspection tools!

## 🎮 What is This?

Gamma Flip Ping Pong transforms complex options market dynamics into an engaging visual experience. The game simulates the concept of "gamma flip" - a critical level in options markets where dealer hedging behavior changes from negative to positive gamma exposure.

### Gameplay Mechanics

- **Spot Price (Green Paddle)**: Represents the actual stock price
- **Flip Price (Red Paddle)**: Represents the gamma flip level
- **Gamma Line (Yellow)**: The center line where gamma flips
- **Ball Movement**: Simulates price action across the gamma flip threshold
- **Position Tracking**: Monitor your $100 position's P&L in real-time

### Scoring System

- **Flip Score**: Increments when the flip paddle hits the ball AND your position is profitable
- **Total Hits**: Tracks all paddle interactions
- **P&L**: Live profit/loss calculation based on price movement

## 🚀 Features

- **Live Mode**: Uses the latest stock price with simulated drift
- **Backtesting Mode**: Replay historical price data with date range filters
- **Position Tracking**: Default $100 position with customizable size
- **Performance Reports**: Automatic report generation on session end
- **CSV Export**: Detailed performance logs saved to CSV
- **Multi-Ticker Support**: Works with any ticker in your data directory

## 📋 Requirements

```bash
pip install pygame pandas
```

## 🛠️ Installation

1. Ensure your stock data is organized in the following structure:
   ```
   F:/inputs/stocks/
   ├── SPY.csv
   ├── AAPL.csv
   └── ...
   ```

2. Stock CSV files should contain at minimum:
   - `Date` column
   - `Close/Last` or `Close` column (price data)

3. Clone/download this project and navigate to the finvolley directory

## 💻 Usage

### Quick Start

1. **First, inspect your data to see available dates:**
   ```bash
   python finvolley.py --ticker SPY --debug-data
   ```

2. **Then run a backtest with the correct dates:**
   ```bash
   python finvolley.py --ticker SPY --start-date 10-1-2025 --end-date 10-13-2025 --position-size 500
   ```

### Basic Usage (Live Mode)

Run with latest price data:
```bash
python finvolley.py
```

### Backtesting with Date Range

Test historical performance:
```bash
python finvolley.py --start-date 10-1-2025 --end-date 10-13-2025
```

### Different Ticker

Analyze any stock:
```bash
python finvolley.py --ticker AAPL
```

### Custom Position Size

Start with a different position value:
```bash
python finvolley.py --position-size 500
```

### Custom Data Directory

Use a different data source:
```bash
python finvolley.py --base-dir /path/to/your/data
```

### Debug Mode

Inspect all loaded data points:
```bash
python finvolley.py --ticker SPY --debug-data
```

### Complete Example

Full-featured backtest:
```bash
python finvolley.py --ticker TSLA --start-date 10-1-2025 --end-date 10-13-2025 --position-size 1000 --base-dir F:/inputs/stocks
```

**Note:** Make sure your date range matches the actual data in your CSV files. Use `--debug-data` to verify available dates.

## 📊 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ticker` | str | SPY | Stock ticker symbol |
| `--start-date` | str | None | Start date (MM-DD-YYYY) |
| `--end-date` | str | None | End date (MM-DD-YYYY) |
| `--position-size` | float | 100.0 | Position size in dollars |
| `--base-dir` | str | F:/inputs/stocks | Base directory for stock data |
| `--debug-data` | flag | False | Print all loaded data points for inspection |

## 📈 Report Output

When you close the game window, a detailed performance report is generated:

### Console Output
- Session details (ticker, duration, date range)
- Position details (size, entry price, shares)
- Performance metrics (P&L, max/min P&L, total hits, score)

### File Outputs
1. **Text Report**: `finvolley_report_{ticker}_{timestamp}.txt`
   - Comprehensive session summary
   - All performance metrics

2. **CSV Log**: `finvolley_log_{ticker}_{timestamp}.csv`
   - Tick-by-tick performance data
   - Columns: timestamp, spot_price, flip_price, pnl, hits, score

### Example Report
```
============================================================
GAMMA FLIP PING PONG - PERFORMANCE REPORT
============================================================

Session Details:
  Ticker: SPY
  Start Time: 2025-10-14 10:30:15
  End Time: 2025-10-14 10:35:42
  Duration: 327.00 seconds
  Backtesting Period: 10-1-2025 to 10-13-2025

Position Details:
  Position Size: $100.00
  Entry Price: $573.50
  Shares: 0.1744
  Final Spot Price: $578.25
  Final Flip Price: $577.89

Performance Metrics:
  Final P&L: $0.83
  P&L %: 0.83%
  Max P&L: $1.52
  Min P&L: -$0.45
  Total Hits: 47
  Flip Score: 23
  Price Updates: 327
  Average P&L: $0.61

============================================================
```

## 🎯 How to Play

1. **Launch the game** with your desired parameters
2. **Watch the simulation**: 
   - Green paddle (Spot) and Red paddle (Flip) automatically track the ball
   - Ball represents market volatility crossing the gamma flip line
3. **Monitor your position**:
   - P&L updates in real-time (green = profit, red = loss)
   - Flip Score increases on successful profitable hits
4. **Close the window** to generate your performance report
5. **Review results**: Check console output and saved files

## 🔍 Understanding the Metrics

### Flip Score
Represents successful market timing when your position is profitable. A higher score indicates better alignment with profitable price movements.

### Total Hits
All paddle-ball interactions. Tracks market volatility and activity level.

### P&L Metrics
- **Final P&L**: Your position's profit/loss at session end
- **Max P&L**: Best performance during the session
- **Min P&L**: Worst drawdown during the session
- **P&L %**: Percentage return on your position size

### Price Updates
Number of 1-second price updates (1 per second in simulation)

## 🧪 Testing Strategies

Use this tool to:
- **Compare tickers**: Run multiple sessions to see which stocks had better gamma dynamics
- **Test date ranges**: Find periods of high volatility or strong trends
- **Position sizing**: Experiment with different position sizes to understand scaling
- **Pattern recognition**: Observe how price movements interact with flip levels

## 📁 Output Files Location

All reports and logs are saved in the `finvolley/` directory:
```
finvolley/
├── finvolley.py
├── data_loader.py
├── README.md
├── finvolley_report_SPY_20241014_103542.txt
├── finvolley_log_SPY_20241014_103542.csv
└── ...
```

## 🐛 Troubleshooting

### Inspecting Your Data

**To see exactly what dates are in your data:**
```bash
# Basic inspection (shows first 3 and last 3 data points):
python finvolley.py --ticker SPY

# Full data inspection (shows ALL data points):
python finvolley.py --ticker SPY --debug-data

# Check specific date range:
python finvolley.py --ticker SPY --start-date 10-1-2025 --end-date 10-13-2025 --debug-data
```

The console output will show:
- Total number of data points loaded
- First 3 dates and prices
- Last 3 dates and prices (or ALL with `--debug-data`)

**Common findings:**
- **Weekend gaps**: Markets are closed Sat/Sun, so 10/11-10/12 might not exist
- **Holiday gaps**: Market holidays won't have data
- **CSV update lag**: Your CSV might not include the most recent trading days

### "IndexError: list index out of range" or "Date filter resulted in empty dataset"
**This is the most common error!** It means your date range doesn't match your data.

**Solution:**
1. Run with `--debug-data` to see what dates are actually loaded
2. Check the console output for: `"Available date range in data: ..."`
3. The data might be from a different year than you specified
4. Example: You specified 2024 but data is from 2025

**Fix:**
```bash
# First, inspect your data:
python finvolley.py --ticker SPY --debug-data

# Then use the correct year:
python finvolley.py --start-date 10-1-2025 --end-date 10-13-2025
```

### "Stock file not found" Error
- Verify your CSV file exists in the base directory
- Check ticker symbol matches filename (e.g., SPY.csv for --ticker SPY)
- Try specifying --base-dir explicitly

### "No price column found" Error
- Ensure CSV has either "Close/Last" or "Close" column
- Check CSV format matches expected structure
- The error message will show available columns to help debug

### "No date directories found" Error
- This occurs if using option chain data (not applicable for finvolley)
- Ensure you're pointing to stock data directory, not options directory

### Pygame Issues
- Install pygame: `pip install pygame`
- On some systems, you may need: `pip install pygame --upgrade`

## 🔧 Technical Details

### Price Update Logic

**Live Mode** (no dates specified):
- Uses latest price from CSV
- Simulates drift based on gamma flip position
- Spot price increases when ball is right of gamma line
- Flip price increases when ball is left of gamma line

**Backtest Mode** (dates specified):
- Iterates through historical prices (1 per second)
- Flip price tracks spot price with 95% convergence
- Real market data drives the simulation

### Position Calculation
```python
shares = position_size / entry_price
pnl = (current_spot_price - entry_price) * shares
pnl_percentage = (pnl / position_size) * 100
```

## 🎨 Customization

Want to modify the game? Key variables to adjust in `finvolley.py`:

- **PADDLE_SPEED**: Paddle tracking speed (search for `PADDLE_SPEED = 4`)
- **PRICE_STEP**: Price drift magnitude in live mode (search for `PRICE_STEP = max(`)
- **Update frequency**: Change from 1000ms to adjust tick rate (search for `if now - last_price_update_ms >= 1000`)
- **Colors**: Customize the visual theme (search for `# Colors` section)
- **POSITION_SIZE**: Default position size (use `--position-size` flag or modify default)

## 📚 Learn More

This tool visualizes concepts from:
- **Gamma exposure** in options markets
- **Dealer hedging** behavior
- **Price action** around flip levels
- **Position P&L** dynamics

For a deeper understanding, research:
- Options gamma and dealer flows
- Spot-gamma relationships
- Zero-gamma levels in markets

## 🤝 Contributing

Ideas for improvements? Consider:
- Additional technical indicators overlays
- Multiple position tracking
- Trade signal generation
- Enhanced visualizations
- Real-time data integration

## 📄 License

Free to use for educational and research purposes.

## 🙏 Acknowledgments

Built with:
- **Pygame**: Graphics and game engine
- **Pandas**: Data processing
- **data_loader.py**: Stock data utilities

---

**Happy Trading! 🚀📈**

*Remember: This is a visualization tool for educational purposes. Past performance does not guarantee future results.*

