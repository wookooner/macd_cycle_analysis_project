import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root():
    here = Path(__file__).resolve()
    for current in [here.parent] + list(here.parents):
        if (current / 'data' / 'cycle_data' / 'structured').exists():
            return current
    for current in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
        if (current / 'data' / 'cycle_data' / 'structured').exists():
            return current
    raise FileNotFoundError("Could not locate project root with data/cycle_data/structured")


def resolve_cycle_data_dir(project_root):
    base_dir = project_root / 'data' / 'cycle_data' / 'structured'
    dataset_name = os.environ.get('BACKTEST_DATASET', '').strip().lower()

    candidates = []
    if dataset_name:
        candidates.append(base_dir / dataset_name)
    candidates.extend([base_dir / 'btc', base_dir])

    required = ['cycles_1h.parquet', 'cycles_4h.parquet', 'cycles_1d.parquet', 'cycles_1w.parquet', 'cycle_hierarchy_map.json']
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate

    raise FileNotFoundError(f"Could not find cycle data directory under {base_dir}")


PROJECT_ROOT = find_project_root()
DATA_DIR = resolve_cycle_data_dir(PROJECT_ROOT)

print(f"Using cycle data from: {DATA_DIR}")

df_1h = pd.read_parquet(DATA_DIR / 'cycles_1h.parquet')
df_4h = pd.read_parquet(DATA_DIR / 'cycles_4h.parquet')
df_1d = pd.read_parquet(DATA_DIR / 'cycles_1d.parquet')
df_1w = pd.read_parquet(DATA_DIR / 'cycles_1w.parquet')

with open(DATA_DIR / 'cycle_hierarchy_map.json', 'r', encoding='utf-8') as f:
    hierarchy = json.load(f)


def build_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        cid = row['cycle_id']
        candle_data = row['candle_data']
        features = row['cycle_features']
        lookup[cid] = {
            'type': row['cycle_type'],
            'dur': row['duration_candles'],
            'candle_data': candle_data,
            'start_price': candle_data[0]['close'],
            'end_price': candle_data[-1]['close'],
            'price_pct': features['change']['price_pct'],
            'start_date': str(row['start_date']),
        }
    return lookup


print("Building lookups...")
lk = {
    '1h': build_lookup(df_1h),
    '4h': build_lookup(df_4h),
    '1d': build_lookup(df_1d),
    '1w': build_lookup(df_1w),
}

h1h = hierarchy['1h']
h4h = hierarchy['4h']


def get_parents(cid):
    """Returns (4h_id, 1d_id, 1w_id) or None."""
    info = h1h.get(cid, {})
    pids = info.get('parent_cycle_ids', {})
    p4h_list = pids.get('4h', [])
    p1d_list = pids.get('1d', [])
    p1w_list = pids.get('1w', [])
    return (
        p4h_list[0] if p4h_list else None,
        p1d_list[0] if p1d_list else None,
        p1w_list[0] if p1w_list else None,
    )


def get_1h_children(p4h_id):
    return h4h.get(p4h_id, {}).get('child_cycle_ids', {}).get('1h', [])


all_1h_ids = sorted(lk['1h'].keys(), key=lambda x: int(x.split('_')[-1]))
print(f"Total 1h cycles: {len(all_1h_ids)}")


def calc_ret(info, entry_idx=1):
    candle_data = info['candle_data']
    if len(candle_data) <= entry_idx:
        return None

    entry_price = candle_data[entry_idx]['close']
    exit_price = info['end_price']
    if info['type'] == 'up':
        return (exit_price - entry_price) / entry_price * 100
    return (entry_price - exit_price) / entry_price * 100


def run(name, filt, entry_idx=1):
    trades = []
    prev_dur = None

    for cid in all_1h_ids:
        info = lk['1h'][cid]
        p4h, p1d, p1w = get_parents(cid)
        if not all([p4h, p1d, p1w]):
            prev_dur = info['dur']
            continue
        if p4h not in lk['4h'] or p1d not in lk['1d'] or p1w not in lk['1w']:
            prev_dur = info['dur']
            continue

        ctx = {
            'info': info,
            'cid': cid,
            '4h': lk['4h'][p4h],
            '1d': lk['1d'][p1d],
            '1w': lk['1w'][p1w],
            'p4h_id': p4h,
            'prev_dur': prev_dur,
            'entry_idx': entry_idx,
            'confirmed_candles': entry_idx + 1,
        }

        children = get_1h_children(p4h)
        ctx['idx_in_4h'] = children.index(cid) if cid in children else -1
        ctx['total_in_4h'] = len(children)

        timeframes = [lk['1w'][p1w]['type'], lk['1d'][p1d]['type'], lk['4h'][p4h]['type'], info['type']]
        n_up = sum(1 for t in timeframes if t == 'up')
        combo = ''.join('U' if t == 'up' else 'D' for t in timeframes)
        ctx['n_up'] = n_up
        ctx['combo'] = combo

        ok, direction = filt(ctx)
        prev_dur = info['dur']
        if not ok:
            continue

        ret = calc_ret(info, entry_idx)
        if ret is None:
            continue

        if (direction == 'long' and info['type'] == 'down') or (
            direction == 'short' and info['type'] == 'up'
        ):
            ret = -ret

        trades.append(
            {
                'ret': ret,
                'dur': info['dur'],
                'n_up': n_up,
                'combo': combo,
                'direction': direction,
                'correct': (direction == 'long' and info['type'] == 'up')
                or (direction == 'short' and info['type'] == 'down'),
                'year': int(info['start_date'][:4]),
            }
        )

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def s1_all(ctx):
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s2_dur5(ctx):
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s3_4h_align(ctx):
    if ctx['4h']['type'] != ctx['info']['type']:
        return False, None
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s4_4h_align_dur5(ctx):
    if ctx['4h']['type'] != ctx['info']['type']:
        return False, None
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s5_first_1h_in_4h(ctx):
    if ctx['idx_in_4h'] != 0:
        return False, None
    if ctx['4h']['type'] != ctx['info']['type']:
        return False, None
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s6_nup_rulebook(ctx):
    n = ctx['n_up']
    if n == 4:
        return True, 'long'
    if n == 0:
        return True, 'short'
    if n == 3:
        return True, 'long'
    if n == 1:
        return True, 'short'
    return False, None


def s7_nup_4h_aligned(ctx):
    n = ctx['n_up']
    if n == 4:
        return True, 'long'
    if n == 0:
        return True, 'short'
    if n == 3 and ctx['4h']['type'] == ctx['info']['type']:
        return True, 'long'
    if n == 1 and ctx['4h']['type'] == ctx['info']['type']:
        return True, 'short'
    return False, None


def s8_4h_transition(ctx):
    """Enter when 4h and 1h are aligned on the first 1h cycle in that 4h cycle."""
    if ctx['idx_in_4h'] != 0:
        return False, None
    if ctx['4h']['type'] != ctx['info']['type']:
        return False, None
    return True, 'long' if ctx['4h']['type'] == 'up' else 'short'


def s9_3candle_aligned(ctx):
    """4h aligned, but enter only from the sixth candle to make dur>5 observable."""
    if ctx['4h']['type'] != ctx['info']['type']:
        return False, None
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s10_skip_prev_short(ctx):
    if ctx['4h']['type'] != ctx['info']['type']:
        return False, None
    if ctx['prev_dur'] is not None and ctx['prev_dur'] <= 4:
        return False, None
    return True, 'long' if ctx['info']['type'] == 'up' else 'short'


def s11_nup_full_rules(ctx):
    if ctx['prev_dur'] is not None and ctx['prev_dur'] <= 4:
        return False, None

    n = ctx['n_up']
    combo = ctx['combo']
    if combo in ['UUUD', 'DDDU', 'UDDU', 'DUUD']:
        return False, None
    if n == 4:
        return True, 'long'
    if n == 0:
        return True, 'short'
    if n == 3 and ctx['4h']['type'] == ctx['info']['type']:
        return True, 'long'
    if n == 1 and ctx['4h']['type'] == ctx['info']['type']:
        return True, 'short'
    return False, None


strats = [
    ('S1_all', s1_all, 1),
    ('S2_dur5_entry5', s2_dur5, 4),
    ('S3_4h_align', s3_4h_align, 1),
    ('S4_4h_align_dur5_entry5', s4_4h_align_dur5, 4),
    ('S5_first_1h_in_4h', s5_first_1h_in_4h, 1),
    ('S6_n_up_rulebook_entry5', s6_nup_rulebook, 4),
    ('S7_n_up_4h_aligned_entry5', s7_nup_4h_aligned, 4),
    ('S8_4h_transition_entry5', s8_4h_transition, 4),
    ('S9_4h_aligned_entry6', s9_3candle_aligned, 5),
    ('S10_skip_prev_short_entry5', s10_skip_prev_short, 4),
    ('S11_full_rules_entry5', s11_nup_full_rules, 4),
]


print("\n" + "=" * 120)
print(
    f"{'Strategy':<30} {'Trades':>6} {'WR%':>6} {'Avg%':>8} {'Cum%':>9} "
    f"{'PF':>6} {'AvgW':>8} {'AvgL':>8} {'W/L':>6} {'Sharpe':>8} {'MDD%':>8} {'AvgDur':>8}"
)
print("=" * 120)

all_dfs = {}
for strategy_name, strategy_filter, strategy_entry_idx in strats:
    df = run(strategy_name, strategy_filter, strategy_entry_idx)
    all_dfs[strategy_name] = df
    if df.empty:
        print(f"{strategy_name:<30} {'N/A':>6}")
        continue

    winners = df[df['ret'] > 0]
    losers = df[df['ret'] <= 0]
    win_rate = len(winners) / len(df) * 100
    avg_w = winners['ret'].mean() if len(winners) else 0
    avg_l = abs(losers['ret'].mean()) if len(losers) else 0.001
    pf = winners['ret'].sum() / abs(losers['ret'].sum()) if len(losers) and losers['ret'].sum() != 0 else 999
    sharpe = df['ret'].mean() / df['ret'].std() if df['ret'].std() > 0 else 0
    cum = df['ret'].cumsum()
    mdd = (cum - cum.cummax()).min()

    print(
        f"{strategy_name:<30} {len(df):>6} {win_rate:>6.1f} {df['ret'].mean():>8.3f} "
        f"{df['ret'].sum():>9.1f} {pf:>6.2f} {avg_w:>8.3f} {avg_l:>8.3f} "
        f"{avg_w / avg_l:>6.2f} {sharpe:>8.3f} {mdd:>8.1f} {df['dur'].mean():>8.1f}"
    )


print("\n\n" + "=" * 80)
print("Yearly cumulative return comparison")
print("=" * 80)

key_strats = ['S1_all', 'S3_4h_align', 'S6_n_up_rulebook_entry5', 'S11_full_rules_entry5']
years = range(2017, 2027)

header = f"{'Year':>6}"
for name in key_strats:
    header += f" {name:>22}"
print(header)
print("-" * 120)

for year in years:
    line = f"{year:>6}"
    for name in key_strats:
        df = all_dfs[name]
        if df.empty:
            line += f" {'N/A':>22}"
            continue
        sub = df[df['year'] == year]
        if len(sub) == 0:
            line += f" {'---':>22}"
            continue
        winners = sub[sub['ret'] > 0]
        win_rate = len(winners) / len(sub) * 100
        line += f" {sub['ret'].sum():>10.1f}({win_rate:>4.0f}%)"
    print(line)


print("\n\n" + "=" * 80)
print("Duration bucket breakdown")
print("=" * 80)

dur_bins = [(3, 4, '3-4'), (5, 7, '5-7'), (8, 12, '8-12'), (13, 19, '13-19'), (20, 999, '20+')]
for name in key_strats:
    df = all_dfs[name]
    if df.empty:
        continue
    print(f"\n--- {name} ---")
    for lo, hi, label in dur_bins:
        sub = df[(df['dur'] >= lo) & (df['dur'] <= hi)]
        if len(sub) == 0:
            continue
        winners = sub[sub['ret'] > 0]
        print(
            f"  dur {label:>5}: n={len(sub):>5}, WR={len(winners) / len(sub) * 100:>5.1f}%, "
            f"mean={sub['ret'].mean():>7.3f}%, sum={sub['ret'].sum():>8.1f}%"
        )


print("\n\n" + "=" * 80)
print("S3 4h-align breakdown by n_up")
print("=" * 80)

df3 = all_dfs['S3_4h_align']
if not df3.empty:
    for nu in sorted(df3['n_up'].unique()):
        sub = df3[df3['n_up'] == nu]
        winners = sub[sub['ret'] > 0]
        losers = sub[sub['ret'] <= 0]
        pf = winners['ret'].sum() / abs(losers['ret'].sum()) if len(losers) and losers['ret'].sum() != 0 else 999
        print(
            f"  n_up={nu}: n={len(sub):>5}, WR={len(winners) / len(sub) * 100:>5.1f}%, "
            f"mean={sub['ret'].mean():>7.3f}%, PF={pf:>6.2f}"
        )


print("\n\n" + "=" * 80)
print("S11 combo breakdown")
print("=" * 80)

df11 = all_dfs['S11_full_rules_entry5']
if not df11.empty:
    combo_stats = []
    for combo in df11['combo'].unique():
        sub = df11[df11['combo'] == combo]
        winners = sub[sub['ret'] > 0]
        combo_stats.append(
            {
                'combo': combo,
                'n': len(sub),
                'wr': len(winners) / len(sub) * 100,
                'mean': sub['ret'].mean(),
                'sum': sub['ret'].sum(),
            }
        )
    combo_df = pd.DataFrame(combo_stats).sort_values('mean', ascending=False)
    print(f"\n{'combo':>6} {'n':>5} {'WR%':>6} {'Avg%':>8} {'Cum%':>8}")
    for _, row in combo_df.iterrows():
        print(f"{row['combo']:>6} {row['n']:>5} {row['wr']:>6.1f} {row['mean']:>8.3f} {row['sum']:>8.1f}")


print("\n\n" + "=" * 80)
print("S5 vs S8 transition detail")
print("=" * 80)

for name in ['S5_first_1h_in_4h', 'S8_4h_transition_entry5']:
    df = all_dfs[name]
    if df.empty:
        continue
    print(f"\n--- {name} ---")
    print(f"  trades: {len(df)}, WR: {len(df[df['ret'] > 0]) / len(df) * 100:.1f}%")
    print(f"  avg: {df['ret'].mean():.3f}%, cum: {df['ret'].sum():.1f}%")
    print("  by n_up")
    for nu in sorted(df['n_up'].unique()):
        sub = df[df['n_up'] == nu]
        winners = sub[sub['ret'] > 0]
        print(f"    n_up={nu}: n={len(sub):>4}, WR={len(winners) / len(sub) * 100:>5.1f}%, mean={sub['ret'].mean():>7.3f}%")


print("\n\nDone!")
