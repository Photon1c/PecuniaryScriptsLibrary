import pygame
import random
import os
import pandas as pd
import argparse
from datetime import datetime
from data_loader import load_stock_data

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 400
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Gamma Flip Ping Pong: Backtesting Simulator")
parser.add_argument("--ticker", type=str, default="SPY", help="Stock ticker symbol (default: SPY)")
parser.add_argument("--start-date", type=str, help="Start date for backtesting (format: MM-DD-YYYY)")
parser.add_argument("--end-date", type=str, help="End date for backtesting (format: MM-DD-YYYY)")
parser.add_argument("--base-dir", type=str, default="F:/inputs/stocks", help="Base directory for stock data")
parser.add_argument("--position-size", type=float, default=100.0, help="Position size in dollars (default: $100)")
parser.add_argument("--debug-data", action="store_true", help="Print all loaded data points for inspection")
args = parser.parse_args()

# Load stock data using data_loader
def get_stock_prices(ticker, start_date=None, end_date=None, base_dir="F:/inputs/stocks"):
    try:
        df = load_stock_data(ticker, base_dir=base_dir)
        
        # Clean price column
        if "Close/Last" in df.columns:
            df["Price"] = df["Close/Last"].apply(
                lambda x: float(str(x).replace("$", "").replace(",", "").strip())
            )
        elif "Close" in df.columns:
            df["Price"] = df["Close"].apply(
                lambda x: float(str(x).replace("$", "").replace(",", "").strip()) if isinstance(x, str) else float(x)
            )
        else:
            raise ValueError(f"No price column found in CSV. Available columns: {df.columns.tolist()}")
        
        # Ensure Date column is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # Filter by date range if provided
        if start_date or end_date:
            # Store original date range before filtering
            orig_min = df["Date"].min()
            orig_max = df["Date"].max()
            
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df["Date"] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df["Date"] <= end]
            
            if len(df) == 0:
                print(f"\nWARNING: No data found for date range {start_date} to {end_date}")
                print(f"Available date range in data: {orig_min.strftime('%m-%d-%Y')} to {orig_max.strftime('%m-%d-%Y')}")
                print(f"Note: Check if you meant year {orig_max.year} instead of the year you specified")
                raise ValueError("Date filter resulted in empty dataset")
        
        df = df.sort_values("Date").reset_index(drop=True)
        result = df[["Date", "Price"]].to_dict("records")
        
        if len(result) == 0:
            raise ValueError("No data available after processing")
        
        print(f"\n{'='*60}")
        print(f"Loaded {len(result)} price points for {ticker}")
        print(f"{'='*60}")
        
        # Debug mode: print all data
        if args.debug_data:
            print(f"\nALL DATA POINTS:")
            for i, point in enumerate(result):
                date_str = point['Date'].strftime('%m/%d/%Y') if isinstance(point['Date'], pd.Timestamp) else str(point['Date'])
                print(f"  {i+1}. {date_str}: ${point['Price']:.2f}")
        else:
            # Show first few dates
            print(f"\nFirst 3 data points:")
            for i, point in enumerate(result[:3]):
                date_str = point['Date'].strftime('%m/%d/%Y') if isinstance(point['Date'], pd.Timestamp) else str(point['Date'])
                print(f"  {i+1}. {date_str}: ${point['Price']:.2f}")
            
            # Show last few dates
            if len(result) > 3:
                print(f"\nLast 3 data points:")
                for i, point in enumerate(result[-3:]):
                    date_str = point['Date'].strftime('%m/%d/%Y') if isinstance(point['Date'], pd.Timestamp) else str(point['Date'])
                    idx = len(result) - 3 + i + 1
                    print(f"  {idx}. {date_str}: ${point['Price']:.2f}")
            
            print(f"\n(Use --debug-data flag to see all data points)")
        
        print(f"\n{'='*60}\n")
        return result
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Using fallback: single price point at $100")
        return [{"Date": datetime.now(), "Price": 100.0}]

# Load price data
price_data = get_stock_prices(args.ticker, args.start_date, args.end_date, args.base_dir)
current_price_idx = 0
SPY_SPOT_PRICE = price_data[current_price_idx]["Price"]

caption = f"Gamma Flip Ping Pong: {args.ticker}"
if args.start_date and args.end_date:
    caption += f" ({args.start_date} to {args.end_date})"
pygame.display.set_caption(caption)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FLIP_COLOR = (255, 100, 100)
SPOT_COLOR = (100, 255, 100)
GAMMA_LINE_COLOR = (255, 255, 0)

# Game Variables
GAMMA_FLIP = WIDTH // 2
BALL_RADIUS = 10
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 60

# Position Tracking
POSITION_SIZE = args.position_size
entry_price = SPY_SPOT_PRICE
shares = POSITION_SIZE / entry_price
current_pnl = 0.0
max_pnl = 0.0
min_pnl = 0.0
total_hits = 0
price_updates = 0

# Positions / Scores / Prices
flip_score = 0
spot_price = SPY_SPOT_PRICE
# Initialize flip price equal to spot (ensures non-zero start)
flip_price = SPY_SPOT_PRICE
if flip_price <= 0:
    flip_price = SPY_SPOT_PRICE

# Performance tracking
performance_log = []
start_time = datetime.now()

ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_vel_x = random.choice([-4, 4])
ball_vel_y = random.choice([-2, 2])

flip_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
spot_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
PADDLE_SPEED = 4

# Fonts
font = pygame.font.SysFont("Arial", 20)
price_font = pygame.font.SysFont("Arial", 24)

# Price update timing (simulate ~1 tick per second)
last_price_update_ms = pygame.time.get_ticks()

def draw():
    WIN.fill(BLACK)
    pygame.draw.line(WIN, GAMMA_LINE_COLOR, (GAMMA_FLIP, 0), (GAMMA_FLIP, HEIGHT), 2)
    pygame.draw.rect(WIN, FLIP_COLOR, (10, flip_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(WIN, SPOT_COLOR, (WIDTH - 20, spot_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(WIN, WHITE, (ball_x, ball_y), BALL_RADIUS)
    
    price_text = price_font.render(f"Spot: ${spot_price:.2f}   Flip: ${flip_price:.2f}", True, WHITE)
    WIN.blit(price_text, (WIDTH//2 - price_text.get_width()//2, 10))
    
    # Display P&L and position info
    pnl_color = (100, 255, 100) if current_pnl >= 0 else (255, 100, 100)
    pnl_text = font.render(f"P&L: ${current_pnl:.2f} | Hits: {total_hits} | Flip Score: {flip_score}", True, pnl_color)
    WIN.blit(pnl_text, (WIDTH//2 - pnl_text.get_width()//2, 40))
    
    position_text = font.render(f"Position: {shares:.2f} shares @ ${entry_price:.2f}", True, WHITE)
    WIN.blit(position_text, (WIDTH//2 - position_text.get_width()//2, 65))
    
    # Show current date in backtesting mode
    if len(price_data) > 1 and current_price_idx < len(price_data):
        current_date = price_data[current_price_idx]["Date"]
        if isinstance(current_date, pd.Timestamp):
            date_str = current_date.strftime("%m-%d-%Y")
        else:
            date_str = str(current_date)
        date_text = font.render(f"Date: {date_str}", True, WHITE)
        WIN.blit(date_text, (10, HEIGHT - 30))
    
    pygame.display.update()

def move_ball():
    global ball_x, ball_y, ball_vel_x, ball_vel_y, flip_score, spot_price, flip_price, last_price_update_ms, current_price_idx
    global current_pnl, max_pnl, min_pnl, total_hits, price_updates, performance_log
    
    ball_x += ball_vel_x
    ball_y += ball_vel_y

    if ball_y <= 0 or ball_y >= HEIGHT:
        ball_vel_y *= -1

    # Price step sizes (gentler: ~0.005% per second, min 0.005)
    PRICE_STEP = max(0.005, SPY_SPOT_PRICE * 0.00005)
    HIT_STEP = PRICE_STEP * 0.25

    # Check paddles
    if ball_x <= 20 and flip_paddle_y < ball_y < flip_paddle_y + PADDLE_HEIGHT:
        ball_vel_x *= -1
        total_hits += 1
        # very small nudge on interaction
        flip_price += random.choice([-1, 1]) * HIT_STEP
        # Reward successful hits based on position direction
        if spot_price > entry_price:
            flip_score += 1  # Profitable position, good hit
    elif ball_x >= WIDTH - 30 and spot_paddle_y < ball_y < spot_paddle_y + PADDLE_HEIGHT:
        ball_vel_x *= -1
        total_hits += 1
        # very small nudge on interaction
        spot_price += random.choice([-1, 1]) * HIT_STEP

    # Price update logic (apply at ~1 Hz)
    now = pygame.time.get_ticks()
    if now - last_price_update_ms >= 1000:
        price_updates += 1
        # If in backtesting mode with multiple price points, advance through historical data
        if len(price_data) > 1:
            current_price_idx = (current_price_idx + 1) % len(price_data)
            spot_price = price_data[current_price_idx]["Price"]
            # Adjust flip price to stay close to spot
            flip_price = spot_price + (flip_price - spot_price) * 0.95
        else:
            # Original drift logic for live mode
            if ball_vel_x > 0 and ball_x > GAMMA_FLIP:
                spot_price += PRICE_STEP
            elif ball_vel_x < 0 and ball_x < GAMMA_FLIP:
                flip_price += PRICE_STEP
        
        # Calculate P&L based on current spot price
        current_pnl = (spot_price - entry_price) * shares
        max_pnl = max(max_pnl, current_pnl)
        min_pnl = min(min_pnl, current_pnl)
        
        # Log performance
        performance_log.append({
            "timestamp": datetime.now(),
            "spot_price": spot_price,
            "flip_price": flip_price,
            "pnl": current_pnl,
            "hits": total_hits,
            "score": flip_score
        })
        
        last_price_update_ms = now

    # keep price non-negative
    if spot_price < 0:
        spot_price = 0.0
    if flip_price < 0:
        flip_price = 0.0

    # Reset ball if out of bounds
    if ball_x < 0 or ball_x > WIDTH:
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_vel_x *= random.choice([-1, 1])
        ball_vel_y = random.choice([-2, 2])

def move_paddles():
    global flip_paddle_y, spot_paddle_y
    # Flip follows ball slowly
    if flip_paddle_y + PADDLE_HEIGHT//2 < ball_y:
        flip_paddle_y += PADDLE_SPEED
    else:
        flip_paddle_y -= PADDLE_SPEED

    # Spot follows ball more erratically
    if random.random() < 0.7:
        if spot_paddle_y + PADDLE_HEIGHT//2 < ball_y:
            spot_paddle_y += PADDLE_SPEED
        else:
            spot_paddle_y -= PADDLE_SPEED

def generate_report():
    """Generate a performance report when the game ends"""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    report = []
    report.append("=" * 60)
    report.append("GAMMA FLIP PING PONG - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"\nSession Details:")
    report.append(f"  Ticker: {args.ticker}")
    report.append(f"  Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Duration: {duration:.2f} seconds")
    
    if args.start_date and args.end_date:
        report.append(f"  Backtesting Period: {args.start_date} to {args.end_date}")
    
    report.append(f"\nPosition Details:")
    report.append(f"  Position Size: ${POSITION_SIZE:.2f}")
    report.append(f"  Entry Price: ${entry_price:.2f}")
    report.append(f"  Shares: {shares:.4f}")
    report.append(f"  Final Spot Price: ${spot_price:.2f}")
    report.append(f"  Final Flip Price: ${flip_price:.2f}")
    
    report.append(f"\nPerformance Metrics:")
    report.append(f"  Final P&L: ${current_pnl:.2f}")
    pnl_pct = (current_pnl / POSITION_SIZE * 100) if POSITION_SIZE > 0 else 0
    report.append(f"  P&L %: {pnl_pct:.2f}%")
    report.append(f"  Max P&L: ${max_pnl:.2f}")
    report.append(f"  Min P&L: ${min_pnl:.2f}")
    report.append(f"  Total Hits: {total_hits}")
    report.append(f"  Flip Score: {flip_score}")
    report.append(f"  Price Updates: {price_updates}")
    
    if len(performance_log) > 0:
        avg_pnl = sum(p["pnl"] for p in performance_log) / len(performance_log)
        report.append(f"  Average P&L: ${avg_pnl:.2f}")
    
    report.append("\n" + "=" * 60)
    
    # Print to console
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"finvolley_report_{args.ticker}_{timestamp}.txt"
    filepath = os.path.join("finvolley", filename)
    
    try:
        with open(filepath, "w") as f:
            f.write(report_text)
        print(f"\nReport saved to: {filepath}")
    except Exception as e:
        print(f"\nCould not save report to file: {e}")
    
    # Save performance log as CSV
    if len(performance_log) > 0:
        try:
            df = pd.DataFrame(performance_log)
            csv_filename = f"finvolley_log_{args.ticker}_{timestamp}.csv"
            csv_filepath = os.path.join("finvolley", csv_filename)
            df.to_csv(csv_filepath, index=False)
            print(f"Performance log saved to: {csv_filepath}")
        except Exception as e:
            print(f"Could not save performance log: {e}")

# Main loop
clock = pygame.time.Clock()
run = True
while run:
    clock.tick(60)
    draw()
    move_ball()
    move_paddles()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

# Generate report before closing
generate_report()
pygame.quit()
