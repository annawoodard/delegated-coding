#!/usr/bin/env python3
"""
quota_trajectory.py — project whether you'll run out of Claude quota.

What this can and can't do
--------------------------
The real quota % and reset time live server-side; /usage paints them in a
dialog and never writes them to disk. This tool can't read them directly.

What it CAN read: every assistant message in your session logs
(~/.claude/projects/*/*.jsonl), each with a timestamp and per-model token
counts. It turns those into a price-weighted "burn" curve, fits a line over
the current window, and — once you tell it the current % off /usage — projects
when the curve crosses 100% and whether that lands before the window resets.

Typical use
-----------
1. Run `/usage` in Claude Code, note the weekly % and when it resets.
2. Run:  python3 quota_trajectory.py --calibrate 42 --reset "2026-06-15T00:00"
   (--calibrate is the % shown; --reset is the next reset time, optional)
3. Read the verdict.

Without --calibrate it still reports burn rate and totals, it just can't say
"you'll run out at X" because it doesn't know your limit.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

# Price-per-Mtok proxies (USD). Weekly limits track value, so dollar-equivalent
# is a better quota proxy than raw token counts. Unknown models fall back to Opus.
PRICING = {
    "claude-opus-4-8":   dict(inp=15, out=75, cw5=18.75, cw1h=30, cr=1.5),
    "claude-opus-4-7":   dict(inp=15, out=75, cw5=18.75, cw1h=30, cr=1.5),
    "claude-sonnet-4-6": dict(inp=3,  out=15, cw5=3.75,  cw1h=6,  cr=0.30),
    "claude-haiku-4-5":  dict(inp=1,  out=5,  cw5=1.25,  cw1h=2,  cr=0.10),
    "claude-fable-5":    dict(inp=15, out=75, cw5=18.75, cw1h=30, cr=1.5),
}
DEFAULT_PRICE = PRICING["claude-opus-4-8"]


def cost_usd(model, u):
    """Dollar-equivalent burn for one message's usage block."""
    p = PRICING.get(model, DEFAULT_PRICE)
    cc = u.get("cache_creation", {}) or {}
    cw5 = cc.get("ephemeral_5m_input_tokens", 0)
    cw1h = cc.get("ephemeral_1h_input_tokens", 0)
    # If the breakdown is missing, treat all cache-creation as 5m writes.
    if not cc:
        cw5 = u.get("cache_creation_input_tokens", 0)
    return (
        u.get("input_tokens", 0) * p["inp"]
        + u.get("output_tokens", 0) * p["out"]
        + cw5 * p["cw5"]
        + cw1h * p["cw1h"]
        + u.get("cache_read_input_tokens", 0) * p["cr"]
    ) / 1_000_000


def load_events(logdir):
    """Yield (datetime_utc, model, cost_usd) for every usage-bearing message."""
    events = []
    for f in glob.glob(os.path.join(logdir, "*", "*.jsonl")):
        try:
            fh = open(f)
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = o.get("message", {})
                u = msg.get("usage")
                ts = o.get("timestamp")
                if not u or not ts:
                    continue
                model = msg.get("model", "")
                if model == "<synthetic>":
                    continue
                try:
                    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
                events.append((t.astimezone(dt.timezone.utc), model, cost_usd(model, u)))
    events.sort(key=lambda e: e[0])
    return events


def linfit(xs, ys):
    """Least-squares slope/intercept. xs in hours, ys cumulative cost."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def fmt_dur(hours):
    if hours < 0:
        return "already over"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def analyze(events, window_h, now, calibrate_pct, reset, label):
    cutoff = now - dt.timedelta(hours=window_h)
    win = [(t, c) for t, m, c in events if t >= cutoff]
    print(f"\n=== {label} (last {window_h:g}h window) ===")
    if not win:
        print("  no activity in window")
        return
    spent = sum(c for _, c in win)
    elapsed_h = (now - win[0][0]).total_seconds() / 3600 or 1e-9

    # Cumulative-cost line over the window, x = hours since window start.
    t0 = win[0][0]
    xs, ys, run = [], [], 0.0
    for t, c in win:
        run += c
        xs.append((t - t0).total_seconds() / 3600)
        ys.append(run)
    slope, _ = linfit(xs, ys)  # burn $/h from the fit
    avg_rate = spent / elapsed_h

    print(f"  burned so far : ${spent:.2f}-equiv   ({len(win)} msgs)")
    print(f"  burn rate     : ${slope:.2f}/h (fit), ${avg_rate:.2f}/h (avg)")

    if calibrate_pct is None:
        print("  limit unknown : pass --calibrate <pct from /usage> to project exhaustion")
        return

    if calibrate_pct <= 0:
        print("  calibrate pct must be > 0")
        return
    limit = spent / (calibrate_pct / 100.0)
    remaining = limit - spent
    rate = slope if slope > 0 else avg_rate
    print(f"  implied limit : ${limit:.2f}-equiv  (you said {calibrate_pct:g}% used)")
    print(f"  remaining     : ${remaining:.2f}-equiv  ({100 - calibrate_pct:g}%)")

    if rate <= 0:
        print("  projection    : burn rate ~0, won't run out at this pace")
        return
    hrs_to_empty = remaining / rate
    empty_at = now + dt.timedelta(hours=hrs_to_empty)
    print(f"  runs out in   : {fmt_dur(hrs_to_empty)}  (~{empty_at:%Y-%m-%d %H:%M} UTC)")

    if reset:
        hrs_to_reset = (reset - now).total_seconds() / 3600
        print(f"  resets in     : {fmt_dur(hrs_to_reset)}  ({reset:%Y-%m-%d %H:%M} UTC)")
        if hrs_to_empty < hrs_to_reset:
            short_h = hrs_to_reset - hrs_to_empty
            print(f"  VERDICT       : ⚠️  you run out ~{fmt_dur(short_h)} BEFORE reset")
            safe_rate = remaining / hrs_to_reset
            print(f"                  stay under ${safe_rate:.2f}/h to make it")
        else:
            print("  VERDICT       : ✅  you make it to reset with margin")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default=os.path.expanduser("~/.claude/projects"),
                    help="dir of session logs (default: ~/.claude/projects)")
    ap.add_argument("--calibrate", type=float, metavar="PCT",
                    help="current %% used, read from /usage, to anchor the limit")
    ap.add_argument("--reset", metavar="ISO",
                    help='next reset time, e.g. "2026-06-15T00:00" (UTC assumed)')
    ap.add_argument("--window", type=float, default=168,
                    help="primary window hours (default 168 = weekly)")
    args = ap.parse_args()

    events = load_events(args.logdir)
    if not events:
        print("No usage records found under", args.logdir, file=sys.stderr)
        sys.exit(1)

    now = dt.datetime.now(dt.timezone.utc)
    reset = None
    if args.reset:
        try:
            reset = dt.datetime.fromisoformat(args.reset)
            if reset.tzinfo is None:
                reset = reset.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            print(f"bad --reset {args.reset!r}; use ISO like 2026-06-15T00:00", file=sys.stderr)
            sys.exit(1)

    print(f"loaded {len(events)} messages, {events[0][0]:%Y-%m-%d} → {events[-1][0]:%Y-%m-%d}")
    analyze(events, 5, now, args.calibrate, reset, "5-hour session")
    analyze(events, args.window, now, args.calibrate, reset, "weekly")


if __name__ == "__main__":
    main()
