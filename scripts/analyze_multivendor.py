"""Paper-grade analysis of the three-vendor log-budget experiment.

Reports, using the SAME aggregation rule as the frozen DeepSeek analysis
(log_statistics.group_scores: records -> trajectories -> groups, equal weight per group):

  1. group-weighted accuracy per provider x log condition, with the DeepSeek reference cells
  2. full vs each bounded condition, paired by group, sign-flip test, Holm-corrected
  3. first-pass validity, separating records that were valid on the first attempt from those
     that only succeeded after an archive/re-run cycle (temperature=0 is not deterministic,
     so retrying a content failure enriches the surviving set)
  4. abstention and within-one-step rates, and cost

Makes no API calls.
"""
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from log_statistics import group_scores, paired, holm

COND_ORDER = ['full', 'prefix6000', 'prefix12000', 'suffix6000', 'suffix12000', 'spread6000', 'spread12000']


def cond_name(r):
    return r['policy'] if r.get('budget') is None else r['policy'] + str(r['budget'])


def average_groups(rows):
    v = group_scores(rows)
    return sum(v.values()) / len(v) if v else None


def rows_for(recs, **kw):
    out = []
    for r in recs:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', default='outputs/multivendor_formal_v3')
    ap.add_argument('--pricing-json', default='configs/pricing_multivendor.json')
    ap.add_argument('--deepseek-analysis', default='outputs/formal_log_study_v1/analysis.json')
    a = ap.parse_args()
    out = ROOT / a.directory

    recs, unreadable = [], 0
    for p in glob.glob(str(out / 'records' / '*.json')):
        try:
            r = json.load(open(p, encoding='utf-8'))
        except Exception:
            unreadable += 1
            continue
        if r.get('status') != 'completed':
            continue
        step = (r.get('prediction') or {}).get('step')
        r['abstained'] = step is None
        r['key'] = Path(p).stem
        recs.append(r)

    archived = {Path(p).stem for p in glob.glob(str(out / 'failed_archive' / '*' / '*.json'))}
    price = json.load(open(ROOT / a.pricing_json, encoding='utf-8'))

    print(f'directory: {a.directory}')
    print(f'completed records: {len(recs)}   unreadable: {unreadable}   '
          f'distinct archived attempts: {len(archived)}')
    print()

    # ---- 1. accuracy by provider x condition, DeepSeek method --------------------
    print('=== group-weighted exact accuracy by provider x condition (records->trajectories->groups) ===')
    print(f'{"model":17}{"condition":13}{"groups":>7}{"calls":>7}{"group_acc":>11}{"pooled":>9}{"with1":>8}{"abst":>7}')
    print('-' * 79)
    ref = {}
    try:
        ds = json.load(open(ROOT / a.deepseek_analysis, encoding='utf-8'))
        for t in ds.get('tables', []):
            ref[(t['model'], t['policy'] if t.get('budget') is None else t['policy'] + str(t['budget']))] = t
    except Exception:
        pass
    for (m, c), t in sorted(ref.items(), key=lambda kv: (kv[0][1] != 'full', kv[0][0], kv[0][1])):
        print(f'{m:17}{c:13}{t["groups"]:>7}{t["calls"]:>7}{t["group_accuracy"]:>11.4f}'
              f'{t["pooled_accuracy"]:>9.4f}{t["within_one_group_accuracy"]:>8.4f}'
              f'{t["abstentions"] / t["calls"]:>7.3f}')
    print('-' * 79)
    for prov in ('glm', 'qwen', 'kimi'):
        for c in COND_ORDER:
            sub = [r for r in recs if r['provider'] == prov and cond_name(r) == c]
            if not sub:
                continue
            ga = average_groups(sub)
            pooled = sum(1 for r in sub if r['correct']) / len(sub)
            w1 = average_groups([dict(r, correct=(r['correct'] or (r.get('prediction') or {}).get('step') is not None
                                                   and abs((r['prediction'] or {}).get('step', -99) - GOLDP[r['id']]) <= 1))
                                 for r in sub])
            abst = sum(1 for r in sub if r['abstained']) / len(sub)
            print(f'{prov:17}{c:13}{len(group_scores(sub)):>7}{len(sub):>7}{ga:>11.4f}{pooled:>9.4f}{w1:>8.4f}{abst:>7.3f}')

    # ---- 2. full vs bounded, paired by group ------------------------------------
    print()
    print('=== full vs each bounded condition, paired by group (sign-flip, Holm within provider) ===')
    for prov in ('glm', 'qwen', 'kimi'):
        full = [r for r in recs if r['provider'] == prov and cond_name(r) == 'full']
        tests, names = [], []
        for c in COND_ORDER[1:]:
            sub = [r for r in recs if r['provider'] == prov and cond_name(r) == c]
            if not sub:
                continue
            tests.append(paired(full, sub)); names.append(c)
        ps = holm([t['sign_flip_p'] for t in tests])
        for c, t, p in zip(names, tests, ps):
            mark = '*' if p < .05 else ' '
            print(f'  {prov:5} full-{c:13} diff={t["difference"]:+.4f}  '
                  f'boot95=[{t["bootstrap95"][0]:+.4f},{t["bootstrap95"][1]:+.4f}]  p={t["sign_flip_p"]:.4f} holm={p:.4f}{mark}')

    # ---- 3. first-pass vs post-retry validity ------------------------------------
    print()
    print('=== validity: first attempt vs after archive/re-run ===')
    print(f'{"provider":9}{"completed":>10}{"first_pass":>12}{"first_pass%":>13}{"recovered":>11}')
    print('-' * 55)
    for prov in ('glm', 'qwen', 'kimi'):
        sub = [r for r in recs if r['provider'] == prov]
        fp = [r for r in sub if r['key'] not in archived]
        print(f'{prov:9}{len(sub):>10}{len(fp):>12}{len(fp) / len(sub) * 100:>12.2f}%{len(sub) - len(fp):>11}')

    # ---- 4. cost -----------------------------------------------------------------
    print()
    print(f'{"provider":9}{"input":>12}{"cached":>11}{"output":>10}{"cost_CNY":>10}')
    print('-' * 52)
    total = 0.0
    for prov in ('glm', 'qwen', 'kimi'):
        sub = [r for r in recs if r['provider'] == prov]
        ti = sum((r.get('usage') or {}).get('prompt_tokens') or 0 for r in sub)
        tc = sum(min(((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens') or 0,
                     (r.get('usage') or {}).get('prompt_tokens') or 0) for r in sub)
        to = sum((r.get('usage') or {}).get('completion_tokens') or 0 for r in sub)
        pr = price[prov]
        c = ((ti - tc) / 1e6 * pr['in'] + tc / 1e6 * pr['cached'] + to / 1e6 * pr['out'])
        total += c
        print(f'{prov:9}{ti:>12,}{tc:>11,}{to:>10,}{c:>10.2f}')
    print(f'{"TOTAL":9}{"":>12}{"":>11}{"":>10}{total:>10.2f}')


GOLDP = {}
if __name__ == '__main__':
    GOLDP = {x['id']: x['step'] for x in json.load(
        open(ROOT / 'data/processed/log_study_v1/labels_private.json', encoding='utf-8'))}
    main()
