# live_data.py — load SPY price context + strike OI from local CSV folders (--live)
"""
Expected layout (defaults):
  F:\\inputs\\stocks       — OHLCV CSV(s); newest *SPY* file by mtime (filename contains ticker)
  F:\\inputs\\options\\log — chain CSV(s) may be nested (e.g. log\\spy\\04_17_2026\\spy_quotedata.csv);
  we walk subfolders and pick the newest .csv whose path contains the ticker (case-insensitive).

Use apply_live_config(..., ticker="QQQ") or --ticker to match other symbols.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
import sys
from datetime import datetime


def _norm_header(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")


def _build_header_map(fieldnames: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not fieldnames:
        return out
    for raw in fieldnames:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        out[_norm_header(s)] = s
    return out


def _decode_csv_file(path: str) -> str:
    """Decode broker CSVs: UTF-8 (with BOM), UTF-16 LE/BE, or latin-1 fallback."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) >= 2 and raw[0:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    try:
        return raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _non_comment_lines(text: str) -> list[str]:
    out: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(ln)
    return out


def _pick_start_and_delimiter(lines: list[str]) -> tuple[int, str]:
    """
    Choose header row index + delimiter.

    Preamble lines (e.g. 'SPY quotedata') often have no tabs/commas; the real
    header is the first row with the most delimiter-separated columns. Wrong
    choice makes DictReader put extra cells under None → 'headers: [None]'.
    """
    best_idx, best_cols, best_d = 0, 0, ","
    delims = (",", "\t", ";", "|")
    for i, ln in enumerate(lines[:120]):
        for d in delims:
            c = ln.count(d)
            if c == 0:
                continue
            cols = c + 1
            if cols > best_cols:
                best_cols, best_d, best_idx = cols, d, i
    if best_cols == 0:
        return 0, ","
    return best_idx, best_d


def _clean_numeric_token(s: str) -> str:
    """Strip common decorations from broker number cells."""
    t = str(s).strip()
    for ch in "$%€£ ":
        t = t.replace(ch, "")
    t = t.replace(",", "")
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    return t.strip()


def _to_float(s: str) -> float:
    t = _clean_numeric_token(s)
    if not t or t in ("-", "—", "NA", "N/A", ".", "null", "None"):
        raise ValueError("empty")
    return float(t)


def _options_csv_preference(path: str) -> int:
    """
    Higher = prefer for full-chain OI. Used with mtime so quotedata wins over wall_history.
    """
    b = os.path.basename(path).lower()
    if "quotedata" in b:
        return 300
    if "chain" in b or "strikes" in b or "open_interest" in b:
        return 200
    if "wall" in b or "summary" in b:
        return 50
    return 100


def latest_csv(
    directory: str,
    ticker: str | None = "SPY",
    *,
    recursive: bool = False,
    prefer_options_chain: bool = False,
) -> str | None:
    """
    Return path to the best matching *.csv, or None.

    If `ticker` is set (default "SPY"):
      - non-recursive: basename must contain the ticker (flat stocks folder).
      - recursive: full path must contain the ticker (nested options, e.g. …/spy/…/spy_quotedata.csv).

    When `prefer_options_chain` is True (options dir), pick max by (chain-ish filename score, mtime)
    so spy_quotedata.csv is chosen over spy_wall_history.csv when both exist.

    Pass `ticker=None` for any *.csv under `directory` (still respects `recursive` for where to look).
    """
    if not directory or not os.path.isdir(directory):
        return None
    needle = (ticker or "").strip().lower() or None
    candidates: list[str] = []

    def consider(path: str) -> None:
        if not os.path.isfile(path):
            return
        name = os.path.basename(path)
        if not name.lower().endswith(".csv"):
            return
        if needle is not None:
            if recursive:
                if needle not in os.path.normpath(path).lower():
                    return
            elif needle not in name.lower():
                return
        candidates.append(path)

    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                consider(os.path.join(root, name))
    else:
        for name in os.listdir(directory):
            consider(os.path.join(directory, name))

    if not candidates:
        return None
    if prefer_options_chain:
        return max(
            candidates,
            key=lambda p: (_options_csv_preference(p), os.path.getmtime(p)),
        )
    return max(candidates, key=os.path.getmtime)


def _parse_rows(path: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    text = _decode_csv_file(path)
    lines = _non_comment_lines(text)
    if not lines:
        return [], {}

    start, delim = _pick_start_and_delimiter(lines)
    body = "\n".join(lines[start:])
    reader = csv.DictReader(io.StringIO(body), delimiter=delim)
    raw_names = reader.fieldnames
    hmap = _build_header_map(raw_names)

    rows: list[dict[str, str]] = []
    for r in reader:
        clean: dict[str, str] = {}
        for k, v in r.items():
            if k is None:
                continue
            if v is None:
                continue
            vs = str(v).strip()
            if vs:
                clean[str(k).strip()] = vs
        if clean:
            rows.append(clean)

    return rows, hmap


def _sort_rows_by_date(rows: list[dict[str, str]], hmap: dict[str, str]) -> list[dict[str, str]]:
    date_keys = ("date", "datetime", "timestamp", "time", "dt")
    col = None
    for k in date_keys:
        if k in hmap:
            col = hmap[k]
            break
    if not col:
        return rows

    def sort_key(r: dict[str, str]) -> tuple[int, str]:
        raw = r.get(col, "") or ""
        txt = str(raw).strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"):
            try:
                return (0, datetime.strptime(txt[:10], fmt).isoformat())
            except ValueError:
                continue
        return (1, txt)

    try:
        return sorted(rows, key=sort_key)
    except Exception:
        return rows


def load_stock_ranges(path: str) -> tuple[float, float, float]:
    """
    From a stock OHLCV CSV: (last_close, range_low, range_high) for chart bounds.
    Uses the last up-to-252 rows for high/low window when possible.
    """
    rows, hmap = _parse_rows(path)
    if not rows:
        raise ValueError(f"No data rows in {path!r}")

    rows = _sort_rows_by_date(rows, hmap)

    # Normalized keys: "Close/Last" → close_last, "Adj Close" → adj_close, etc.
    _close_candidates = (
        "adj_close",
        "adjclose",
        "adjusted_close",
        "close_last",
        "last_close",
        "close",
        "last",
        "price",
        "c",
        "close_price",
    )
    close_col = None
    for cand in _close_candidates:
        if cand in hmap:
            close_col = hmap[cand]
            break
    if not close_col:
        raise ValueError(
            f"No close/price column found in {path!r}; "
            f"headers were: {list(rows[0].keys())[:20]}"
        )

    last_row = rows[-1]
    current = _to_float(last_row[close_col])

    low_c = hmap.get("low") or hmap.get("l")
    high_c = hmap.get("high") or hmap.get("h")
    window = rows[-252:] if len(rows) > 252 else rows

    if low_c and high_c:
        lows = [_to_float(r[low_c]) for r in window]
        highs = [_to_float(r[high_c]) for r in window]
        rng_lo = min(lows)
        rng_hi = max(highs)
    else:
        closes = [_to_float(r[close_col]) for r in window]
        rng_lo = min(closes)
        rng_hi = max(closes)

    span = max(rng_hi - rng_lo, 1.0)
    pad = max(15.0, span * 0.06)
    rng_lo -= pad
    rng_hi += pad
    rng_lo = min(rng_lo, current - 5.0)
    rng_hi = max(rng_hi, current + 5.0)
    return current, rng_lo, rng_hi


def _scale_oi_to_thousands(call: float, put: float) -> tuple[float, float]:
    """Viz uses 'k contracts'; normalize if values look like raw contracts."""
    m = max(call, put, 1.0)
    if m > 800:
        return call / 1000.0, put / 1000.0
    return call, put


def _sort_option_rows_by_time(
    rows: list[dict[str, str]], hmap: dict[str, str]
) -> list[dict[str, str]]:
    for dk in ("date_folder", "date", "expiration", "datetime", "timestamp"):
        if dk not in hmap:
            continue
        col = hmap[dk]

        def keyfn(r: dict[str, str]) -> tuple[int, str]:
            return (0, str(r.get(col, "") or ""))

        try:
            return sorted(rows, key=keyfn)
        except Exception:
            break
    return rows


def _load_wall_summary_oi(
    rows: list[dict[str, str]], hmap: dict[str, str]
) -> dict[int, tuple[float, float]] | None:
    """
    One row per snapshot: call wall + put wall strikes and OI
    (e.g. spy_wall_history.csv).
    """
    req = ("call_wall_strike", "call_wall_oi", "put_wall_strike", "put_wall_oi")
    if not all(k in hmap for k in req):
        return None

    rows = _sort_option_rows_by_time(rows, hmap)
    r = rows[-1]
    try:
        cs = int(round(_to_float(r[hmap["call_wall_strike"]])))
        ps = int(round(_to_float(r[hmap["put_wall_strike"]])))
        coi = _to_float(r[hmap["call_wall_oi"]])
        poi = _to_float(r[hmap["put_wall_oi"]])
    except (KeyError, ValueError):
        return None

    coi, poi = _scale_oi_to_thousands(coi, poi)
    out: dict[int, tuple[float, float]] = {}
    oc, op = out.get(cs, (0.0, 0.0))
    out[cs] = (oc + coi, op)
    oc, op = out.get(ps, (0.0, 0.0))
    out[ps] = (oc, op + poi)
    return dict(sorted(out.items()))


def _resolve_chain_strike_column(hmap: dict[str, str]) -> str | None:
    for cand in (
        "strike",
        "strike_price",
        "strikeprice",
        "k",
        "st",
    ):
        if cand in hmap:
            return hmap[cand]
    for nk, orig in sorted(hmap.items()):
        if "strike" not in nk:
            continue
        if any(x in nk for x in ("wall", "distance", "pct", "moneyness", "spread")):
            continue
        return orig
    return None


def _resolve_chain_oi_columns(hmap: dict[str, str]) -> tuple[str | None, str | None]:
    def find_oi(side: str) -> str | None:
        if side == "call":
            cands = (
                "call_oi",
                "calloi",
                "c_oi",
                "call_open_interest",
                "coi",
                "calls_oi",
                "open_interest_call",
                "oi_call",
                "calls",
            )
        else:
            cands = (
                "put_oi",
                "putoi",
                "p_oi",
                "put_open_interest",
                "poi",
                "puts_oi",
                "open_interest_put",
                "oi_put",
                "puts",
            )
        for c in cands:
            if c in hmap:
                return hmap[c]
        for nk, orig in hmap.items():
            if side == "call" and "call" in nk and ("oi" in nk or "open" in nk or "interest" in nk):
                if "wall" in nk or "iv" in nk or "volume" in nk:
                    continue
                return orig
            if side == "put" and "put" in nk and ("oi" in nk or "open" in nk or "interest" in nk):
                if "wall" in nk or "iv" in nk or "volume" in nk:
                    continue
                return orig
        return None

    return find_oi("call"), find_oi("put")


def _load_split_calls_puts_chain(path: str) -> dict[int, tuple[float, float]] | None:
    """
    Chains exported as one row per strike with duplicated column names, e.g.::

        Expiration Date, Calls, Last Sale, …, Open Interest, Strike, Puts, Last Sale, …, Open Interest

    csv.DictReader only keeps the *last* duplicate header, so dict-based parsing
    fails. We use column indices: last Open Interest before Strike = calls OI,
    first Open Interest after Strike = puts OI.
    """
    text = _decode_csv_file(path)
    lines = _non_comment_lines(text)
    if not lines:
        return None
    start, delim = _pick_start_and_delimiter(lines)
    body = "\n".join(lines[start:])
    rdr = csv.reader(io.StringIO(body), delimiter=delim)
    header = next(rdr, None)
    if not header:
        return None
    norms = [_norm_header(str(h).strip()) for h in header]

    # DictReader cannot represent two "Open Interest" columns — handle here first.
    if norms.count("open_interest") < 2:
        return None

    strike_idx = None
    for i, nh in enumerate(norms):
        if nh == "strike":
            strike_idx = i
            break
    if strike_idx is None:
        return None

    oi_keys = frozenset({"open_interest", "openinterest", "oi", "open_int", "openint"})
    call_oi_idx = None
    for i in range(strike_idx - 1, -1, -1):
        if norms[i] in oi_keys:
            call_oi_idx = i
            break

    put_oi_idx = None
    for i in range(strike_idx + 1, len(norms)):
        if norms[i] in oi_keys:
            put_oi_idx = i
            break

    if call_oi_idx is None or put_oi_idx is None:
        return None

    ncols = len(header)
    out: dict[int, tuple[float, float]] = {}
    for row in rdr:
        while len(row) < ncols:
            row.append("")
        raw_sk = str(row[strike_idx]).strip() if strike_idx < len(row) else ""
        if not raw_sk or raw_sk in ("-", "—"):
            continue
        raw_sk = raw_sk.split()[0]
        try:
            strike = int(round(_to_float(raw_sk)))
        except ValueError:
            continue
        try:
            c = _to_float(row[call_oi_idx] if call_oi_idx < len(row) else "0")
        except ValueError:
            c = 0.0
        try:
            p = _to_float(row[put_oi_idx] if put_oi_idx < len(row) else "0")
        except ValueError:
            p = 0.0
        c, p = _scale_oi_to_thousands(c, p)
        if strike in out:
            oc, op = out[strike]
            out[strike] = (oc + c, op + p)
        else:
            out[strike] = (c, p)

    return dict(sorted(out.items())) if out else None


def load_option_strike_oi(path: str) -> dict[int, tuple[float, float]]:
    """
    Parse option data into { strike: (call_k, put_k) }.

    Supports:
      - Full chain CSV (one row per strike).
      - Wall summary CSV (call_wall_strike / put_wall_strike + OI columns).
    """
    rows, hmap = _parse_rows(path)
    if not rows:
        raise ValueError(f"No data rows in {path!r}")

    wall = _load_wall_summary_oi(rows, hmap)
    if wall is not None:
        return wall

    # Calls … Open Interest | Strike | Puts … Open Interest (duplicate header names)
    split_first = _load_split_calls_puts_chain(path)
    if split_first:
        return split_first

    strike_key = _resolve_chain_strike_column(hmap)
    call_col, put_col = _resolve_chain_oi_columns(hmap)

    out: dict[int, tuple[float, float]] = {}
    if strike_key and call_col and put_col:
        for r in rows:
            raw_sk = str(r.get(strike_key, "") or "").strip()
            if not raw_sk or raw_sk in ("-", "—"):
                continue
            raw_sk = raw_sk.split()[0]
            try:
                strike = int(round(_to_float(raw_sk)))
            except (KeyError, ValueError):
                continue
            try:
                c_raw = str(r.get(call_col) or "").strip() or "0"
                p_raw = str(r.get(put_col) or "").strip() or "0"
                c = _to_float(c_raw)
                p = _to_float(p_raw)
            except ValueError:
                continue
            c, p = _scale_oi_to_thousands(c, p)
            if strike in out:
                oc, op = out[strike]
                out[strike] = (oc + c, op + p)
            else:
                out[strike] = (c, p)

    if not strike_key:
        raise ValueError(
            f"No strike column in {path!r}; columns (normalized): {sorted(hmap.keys())!s}"
        )
    if not call_col or not put_col:
        raise ValueError(
            f"Need call and put OI columns in {path!r}; "
            f"strike={strike_key!r}; columns (normalized): {sorted(hmap.keys())!s}"
        )
    if not out:
        raise ValueError(
            f"Parsed zero strikes from {path!r}. "
            f"If headers repeat (Calls … Open Interest, Strike, Puts … Open Interest), "
            f"the file should still work — check for blank strike rows or odd delimiters."
        )

    return dict(sorted(out.items()))


def _activity_scale(total: float) -> float:
    """Map reported volume to crowd-strength multipliers (~0.82–1.45)."""
    if total <= 0:
        return 1.0
    ref = 8e6
    r = math.log10(1.0 + total) / math.log10(1.0 + ref)
    return float(max(0.82, min(1.45, 0.86 + 0.32 * min(1.6, r))))


def load_stock_last_volume(path: str) -> float:
    rows, hmap = _parse_rows(path)
    if not rows:
        return 0.0
    for cand in ("volume", "vol"):
        if cand in hmap:
            k = hmap[cand]
            try:
                return max(0.0, float(_clean_numeric_token(rows[-1][k])))
            except (KeyError, ValueError):
                return 0.0
    return 0.0


def load_option_split_volume_sums(path: str) -> tuple[float, float]:
    """For mirrored chain CSVs: sum call-side Volume + put-side Volume columns."""
    text = _decode_csv_file(path)
    lines = _non_comment_lines(text)
    if not lines:
        return 0.0, 0.0
    start, delim = _pick_start_and_delimiter(lines)
    body = "\n".join(lines[start:])
    rdr = csv.reader(io.StringIO(body), delimiter=delim)
    header = next(rdr, None)
    if not header:
        return 0.0, 0.0
    norms = [_norm_header(str(h).strip()) for h in header]
    if norms.count("volume") < 2:
        return 0.0, 0.0
    strike_idx = next((i for i, nh in enumerate(norms) if nh == "strike"), None)
    if strike_idx is None:
        return 0.0, 0.0
    call_v_idx = None
    for i in range(strike_idx - 1, -1, -1):
        if norms[i] == "volume":
            call_v_idx = i
            break
    put_v_idx = None
    for i in range(strike_idx + 1, len(norms)):
        if norms[i] == "volume":
            put_v_idx = i
            break
    if call_v_idx is None or put_v_idx is None:
        return 0.0, 0.0
    ncols = len(header)
    s_call = 0.0
    s_put = 0.0
    for row in rdr:
        while len(row) < ncols:
            row.append("")
        try:
            s_call += max(0.0, float(_clean_numeric_token(row[call_v_idx])))
        except (ValueError, IndexError):
            pass
        try:
            s_put += max(0.0, float(_clean_numeric_token(row[put_v_idx])))
        except (ValueError, IndexError):
            pass
    return s_call, s_put


def load_option_dict_volume_sum(path: str) -> float:
    """Single Volume column (dict parse); sums all rows."""
    rows, hmap = _parse_rows(path)
    if not rows or "volume" not in hmap:
        return 0.0
    k = hmap["volume"]
    s = 0.0
    for r in rows:
        try:
            s += max(0.0, float(_clean_numeric_token(r.get(k, "0") or "0")))
        except ValueError:
            pass
    return s


def apply_live_config(
    stocks_dir: str,
    options_dir: str,
    ticker: str = "SPY",
) -> None:
    """Patch `config` module in-place; must run before other app imports bind config values."""
    import config as cfg

    sym = (ticker or "SPY").strip().upper() or "SPY"
    sp = latest_csv(stocks_dir, sym, recursive=False)
    op = latest_csv(options_dir, sym, recursive=True, prefer_options_chain=True)
    if not sp:
        print(
            f"[live] ERROR: no .csv in {stocks_dir!r} with {sym!r} in the filename "
            f"(e.g. SPY_daily.csv). Use --ticker or rename the file.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not op:
        print(
            f"[live] ERROR: no .csv under {options_dir!r} (recursive) with {sym!r} in the path, "
            f"e.g. …\\spy\\…\\spy_quotedata.csv",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        current, _lo_hist, _hi_hist = load_stock_ranges(sp)
        oi = load_option_strike_oi(op)
    except Exception as e:
        print(f"[live] ERROR reading CSVs: {e}", file=sys.stderr)
        sys.exit(2)

    if not oi:
        print(f"[live] ERROR: parsed zero strikes from {op!r}", file=sys.stderr)
        sys.exit(2)

    cfg.SPY_CURRENT = float(current)
    cfg.STRIKE_OI.clear()
    cfg.STRIKE_OI.update(oi)

    # Fit Y-axis to strikes + spot (stock history span is often far too wide for chain OI).
    strikes = list(cfg.STRIKE_OI.keys())
    band_lo = min(min(strikes), cfg.SPY_CURRENT)
    band_hi = max(max(strikes), cfg.SPY_CURRENT)
    span = max(band_hi - band_lo, 3.0)
    pad = max(14.0, span * 0.14)
    cfg.SPY_RANGE_LOW = float(band_lo - pad)
    cfg.SPY_RANGE_HIGH = float(band_hi + pad)

    v_call, v_put = load_option_split_volume_sums(op)
    if v_call + v_put <= 0.0:
        v_sum = load_option_dict_volume_sum(op)
        v_call, v_put = v_sum * 0.5, v_sum * 0.5
    cfg.LIVE_OPTION_VOLUME_CALL_SUM = float(v_call)
    cfg.LIVE_OPTION_VOLUME_PUT_SUM = float(v_put)
    cfg.LIVE_LAST_STOCK_VOLUME = float(load_stock_last_volume(sp))
    opt_vol = v_call + v_put
    cfg.LIVE_CROWD_COHESION_SCALE = _activity_scale(opt_vol)
    cfg.LIVE_CROWD_COMPRESSION_SCALE = _activity_scale(opt_vol + cfg.LIVE_LAST_STOCK_VOLUME)

    try:
        op_rel = os.path.relpath(op, options_dir)
    except ValueError:
        op_rel = os.path.basename(op)
    cfg.TITLE = f"{cfg.TITLE}  [LIVE: {os.path.basename(sp)} + {op_rel}]"

    print(f"[live] stocks: {sp}")
    print(f"[live] options: {op}  (under log: {op_rel})")
    print(
        f"[live] {sym} last={cfg.SPY_CURRENT:.2f}  chart_range=[{cfg.SPY_RANGE_LOW:.2f}, {cfg.SPY_RANGE_HIGH:.2f}]  "
        f"strikes={len(cfg.STRIKE_OI)}  "
        f"crowd_scales=(coh={cfg.LIVE_CROWD_COHESION_SCALE:.2f}, comp={cfg.LIVE_CROWD_COMPRESSION_SCALE:.2f})"
    )
