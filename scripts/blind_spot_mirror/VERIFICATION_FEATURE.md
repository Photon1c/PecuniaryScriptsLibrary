# Data Source Verification Feature

## ✅ What Changed

Added **data source verification** to flight plans and console output to help you confirm the correct CSV prices are being used.

## 📊 New Flight Plan Format

Flight plans now include a **Data Source Verification** section at the top:

```markdown
# Flight Plan: SPY

## Data Source Verification
- **Current Spot Price**: $652.00 (from CSV last row)
- **Data Date**: 2024-01-12
- **Flight Mode**: Macro Cruise (Daily)
- **Generated**: 2024-01-12T16:09:21

## Trade Plan
- **Direction**: PUT
- **Confidence**: 42%
- **Horizon**: 1-2 days
- **Entry**: {'price': 639.96, 'reason': 'Flight-based PUT signal'}
- **Stop**: {'price': 666.08, 'reason': '2% risk limit'}
...
```

### What You Can Verify

1. **Current Spot Price** - The exact price from your CSV's last row (`Close/Last` column)
2. **Data Date** - The date from that last row (confirms proper sorting)
3. **Flight Mode** - Whether daily or intraday analysis was used
4. **Generated** - When this plan was created

## 🖥️ Console Output

### Plan Mode

When running `python cli.py plan --symbol SPY`, you'll see:

```
📊 Data Verification:
   Symbol: SPY
   Spot Price: $652.00 (from CSV last row)
   Data Date: 2024-01-12
   Mode: Macro Cruise (Daily)

✈️ Flight plan written to: reports/flight_plan_SPY_20241012_160921.md
```

### Watch Mode

When running `python cli.py watch --symbol SPY`, each update shows:

```
[BSM] Starting watch on SPY (mode: daily)
[BSM] SPY @ $652.00 (2024-01-12) | Altitude: -2.77% | Phase: Thrust | Fuel: 0.0% | Stalls: 2
```

The format is:
```
[BSM] SYMBOL @ $PRICE (DATE) | Altitude: X% | Phase: Y | Fuel: Z% | Stalls: N
```

## 🔍 How to Verify Your CSV

### Step 1: Check Your CSV Last Row

Open `F:/inputs/stocks/SPY.csv` and look at the **last row**:

```csv
Date,Open,High,Low,Close/Last,Volume
...
2024-01-10,650.00,652.00,649.00,651.50,85000000
2024-01-11,651.50,653.00,650.50,652.75,87000000
2024-01-12,652.75,655.00,652.00,652.00,92000000  ← This row
```

### Step 2: Run Your Analysis

```bash
python cli.py plan --symbol SPY
```

### Step 3: Compare Output

Console should show:
```
Spot Price: $652.00 (from CSV last row)
Data Date: 2024-01-12
```

Flight plan file should show:
```
- **Current Spot Price**: $652.00 (from CSV last row)
- **Data Date**: 2024-01-12
```

✅ **Match!** The correct row is being used.

## 🛠️ Troubleshooting

### Issue: Wrong price shown

**Problem**: Spot price doesn't match your CSV's last row

**Check:**
1. Is your CSV sorted by date? (should be: `stock_df.sort_values('Date')`)
2. Is the `Close/Last` column properly formatted? (should be numeric)
3. Did you update the CSV after running the analysis?

**Solution**: The system sorts by date automatically, so the last row after sorting is used.

### Issue: Old date shown

**Problem**: Data Date is older than expected

**Check:**
1. When was your CSV last updated?
2. Run your CSV update script
3. Verify the CSV file timestamp matches your expectation

**Solution**: Update your CSV files, then re-run the analysis.

## 💡 Benefits

1. **Confidence**: Know exactly what price is being used
2. **Debugging**: Quickly spot stale data issues
3. **Compliance**: Audit trail shows data source and timing
4. **Verification**: Cross-check against your broker's prices

## 📝 Example Workflow

### Morning Trading Routine

```bash
# 1. Update your CSV files
python update_spy_data.py

# 2. Generate plan (verify price in output)
python cli.py plan --symbol SPY

# Output shows:
# Spot Price: $652.00 (from CSV last row)
# Data Date: 2024-01-12  ← Today's date ✓

# 3. Check flight plan file for full verification
cat reports/flight_plan_SPY_20241012_090000.md

# 4. If price matches your broker → proceed with plan
# 5. If price is stale → re-update CSV and try again
```

## 🎯 Technical Details

### Where Data Comes From

```python
# In flight_sim_engine.py (line 73):
sampled = stock_df.tail(5).copy()  # Last 5 rows after sorting

# In flight_sim_engine.py (line 131):
"spot": float(sampled['Close/Last'].iloc[-1])  # Last row's close price
```

### What Gets Stored

All verification data is stored in the PlayCard's `audit` field:

```python
audit = {
    "spot_price": 652.00,
    "date": "2024-01-12",
    "flight_mode": "Macro Cruise (Daily)",
    "generated_at": "2024-01-12T16:09:21.123456",
    "signal_source": "flight_simulation",
    "signal_count": 1
}
```

This ensures full traceability of every trading plan.

---

**Status**: ✅ Feature Implemented

You can now verify that the correct CSV prices are being used in every analysis!

