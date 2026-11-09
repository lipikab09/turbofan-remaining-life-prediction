"""
AeroGuard-RUL: calibrated, decision-aware Remaining Useful Life prediction for NASA C-MAPSS.

Stages
 1. Regime-aware normalisation (KMeans on operating settings -> per-regime z-score)
 2. Learned Health Index (HI): a 1-D "damage coordinate" per engine
 3. Window features (level, mean, slope, volatility) + HI dynamics (causal, no look-ahead)
 4. Quantile gradient boosting (7 quantiles) -> predictive distribution of RUL
 5. Conformal calibration: CQR 90% interval + one-sided *guaranteed safe horizon*
 6. Cost-aware maintenance policy simulation + fleet triage
 7. Permutation-based sensor attribution
Everything is exported to one JSON embedded in dashboard/index.html
"""
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

COLS = ['unit', 'cycle'] + [f'op{i}' for i in (1, 2, 3)] + [f's{i}' for i in range(1, 22)]
SENS = [f's{i}' for i in range(1, 22)]
OPS = ['op1', 'op2', 'op3']
CAP, WIN = 125, 30
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
COST = dict(failure=100.0, planned=10.0, waste_per_cycle=0.4)   # editable business assumptions
LEAD = 15                                                       # cycles needed to schedule a shop visit


# ------------------------------------------------------------------ data
def load_real(fd, data_dir):
    rd = lambda f: pd.read_csv(data_dir / f, sep=r'\s+', header=None, names=COLS)
    tr, te = rd(f'train_{fd}.txt'), rd(f'test_{fd}.txt')
    rul = pd.read_csv(data_dir / f'RUL_{fd}.txt', header=None)[0].values
    return tr, te, rul


def synth(fd, seed=7, n=100):
    """Statistically C-MAPSS-like fallback so the project runs with zero downloads."""
    rng = np.random.default_rng(seed)
    nreg = 1 if fd in ('FD001', 'FD003') else 6
    centres = rng.uniform(0, 1, (nreg, 3)) * np.array([42, .84, 100]) if nreg > 1 else np.zeros((1, 3))
    base = rng.uniform(100, 2000, 21)
    dead = rng.choice(21, 7, replace=False)
    sign = rng.choice([-1, 1], 21)
    mag = rng.uniform(.02, .06, 21) * base
    reg_off = rng.normal(0, .03, (nreg, 21)) * base if nreg > 1 else np.zeros((1, 21))
    noise = .006 * base

    def engine(u, L):
        t = np.arange(1, L + 1)
        r = rng.integers(0, nreg, L) if nreg > 1 else np.zeros(L, int)
        ops = centres[r] + rng.normal(0, .002, (L, 3))
        k = rng.uniform(2.2, 3.2)
        wear = (t / L) ** k * rng.uniform(.8, 1.2)
        X = base + reg_off[r] + sign * mag * wear[:, None] * rng.uniform(.5, 1.5, 21) + rng.normal(0, 1, (L, 21)) * noise
        X[:, dead] = base[dead] + reg_off[r][:, dead]
        d = pd.DataFrame(np.c_[np.full(L, u), t, ops, X], columns=COLS)
        d[['unit', 'cycle']] = d[['unit', 'cycle']].astype(int)
        return d

    tr = pd.concat([engine(u, int(rng.integers(150, 360))) for u in range(1, n + 1)], ignore_index=True)
    te, rul = [], []
    for u in range(1, n + 1):
        L = int(rng.integers(150, 360)); full = engine(u, L)
        cut = int(rng.integers(max(40, L // 3), L - 5))
        te.append(full.iloc[:cut]); rul.append(L - cut)
    return tr, pd.concat(te, ignore_index=True), np.array(rul)


# ------------------------------------------------------------------ preprocessing
class Normaliser:
    def fit(self, df, multi):
        self.km = KMeans(6, n_init=5, random_state=0).fit(df[OPS]) if multi else None
        r = self.reg(df)
        g = df.groupby(r)[SENS]
        self.mu, self.sd = g.mean(), g.std().replace(0, np.nan)
        self.keep = [s for s in SENS if self.sd[s].notna().all() and (self.sd[s] > 1e-6).all()]
        return self

    def reg(self, df):
        return self.km.predict(df[OPS]) if self.km is not None else np.zeros(len(df), int)

    def transform(self, df):
        r = self.reg(df)
        Z = (df[self.keep].values - self.mu.loc[r, self.keep].values) / self.sd.loc[r, self.keep].values
        return pd.DataFrame(Z, columns=self.keep, index=df.index)


def add_rul(df):
    df = df.copy(); df['rul'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']; return df


def feats(Zu, hi_u, t, keep):
    """Features for one engine using only rows <= t (no look-ahead)."""
    w = Zu[max(0, t - WIN + 1): t + 1]; n = len(w)
    x = np.arange(n) - (n - 1) / 2
    slope = (x[:, None] * (w - w.mean(0))).sum(0) / max((x ** 2).sum(), 1e-9) if n > 1 else np.zeros(w.shape[1])
    last = Zu[max(0, t - 4): t + 1].mean(0)
    hw = hi_u[max(0, t - WIN + 1): t + 1]
    hs = np.polyfit(np.arange(len(hw)), hw, 1)[0] if len(hw) > 2 else 0.0
    return np.r_[last, w.mean(0), slope, w.std(0) if n > 1 else np.zeros(w.shape[1]), t + 1, hi_u[t], hs]


def names(keep):
    return [f'{a}_{s}' for a in ('last', 'mean', 'slope', 'std') for s in keep] + ['cycle', 'hi_last', 'hi_slope']


def build(df, Z, hi, keep, step, start=10):
    X, y = [], []
    rul_all = df['rul'].values
    for u, idx in df.groupby('unit').indices.items():
        Zu, hu, rul = Z.values[idx], hi.values[idx], rul_all[idx]
        for t in range(start, len(idx), step):
            X.append(feats(Zu, hu, t, keep)); y.append(min(rul[t], CAP))
    return np.array(X), np.array(y, float)


# ------------------------------------------------------------------ metrics
def nasa_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def policy_cost(h, rul):
    """Replace after h more cycles. Engine dies first -> failure cost; else planned + wasted-life cost."""
    c = COST
    return np.where(rul < h, c['failure'], c['planned'] + c['waste_per_cycle'] * (rul - h))


def status(safe):
    return 'critical' if safe <= LEAD else 'schedule' if safe <= 3 * LEAD else 'healthy'


# ------------------------------------------------------------------ main
def run(fd='FD001', data_dir='data', out='results', synthetic=False, log=print):
    t0 = time.time(); data_dir = Path(data_dir); out = Path(out); out.mkdir(exist_ok=True)
    real = (data_dir / f'train_{fd}.txt').exists() and not synthetic
    tr, te, rul_true = load_real(fd, data_dir) if real else synth(fd)
    log(f'[data] {fd} source={"NASA C-MAPSS" if real else "SYNTHETIC demo"} train_engines={tr.unit.nunique()} test_engines={te.unit.nunique()}')
    multi = fd in ('FD002', 'FD004')
    tr = add_rul(tr)
    units = tr.unit.unique(); np.random.default_rng(0).shuffle(units)
    cut = int(len(units) * .75); fit_u = set(units[:cut])
    fit_mask = tr.unit.isin(fit_u).values

    nm = Normaliser().fit(tr[fit_mask], multi); keep = nm.keep
    Ztr, Zte = nm.transform(tr), nm.transform(te)
    hi_model = Ridge(1.0).fit(Ztr[fit_mask].values, (np.minimum(tr.rul[fit_mask], CAP) / CAP).values)

    def hindex(Z, d):   # HI in [0,1] (1 = healthy), causally smoothed so it works online
        h = pd.Series(np.clip(hi_model.predict(Z.values), 0, 1), index=Z.index)
        return h.groupby(d['unit'].values).transform(lambda v: v.rolling(8, min_periods=1).mean())
    hi_tr, hi_te = hindex(Ztr, tr), hindex(Zte, te)
    log(f'[prep] regimes={"6 (KMeans)" if multi else "1"} sensors_kept={len(keep)}/21')

    Xf, yf = build(tr[fit_mask], Ztr[fit_mask], hi_tr[fit_mask], keep, step=3)
    Xc, yc = build(tr[~fit_mask], Ztr[~fit_mask], hi_tr[~fit_mask], keep, step=5)
    log(f'[train] fit_samples={len(Xf)} calib_samples={len(Xc)}')
    models = {q: HGB(loss='quantile', quantile=q, max_iter=250, learning_rate=.06, max_depth=5,
                     min_samples_leaf=20, random_state=0).fit(Xf, yf) for q in QS}
    predq = lambda X: np.sort(np.column_stack([models[q].predict(X) for q in QS]), axis=1).clip(0, CAP)

    # conformal calibration on held-out ENGINES
    Pc = predq(Xc); n = len(yc); lvl = min(1, math.ceil((n + 1) * .9) / n)
    qhat = float(np.quantile(np.maximum(Pc[:, 0] - yc, yc - Pc[:, -1]), lvl))   # CQR, 90% interval
    qlow = float(np.quantile(Pc[:, 1] - yc, lvl))                               # one-sided safe horizon
    log(f'[conformal] CQR width correction={qhat:.1f} cycles | safe-horizon shift={qlow:.1f}')

    # test snapshots
    lastX, units_te = [], []
    for u, idx in te.groupby('unit').indices.items():
        lastX.append(feats(Zte.values[idx], hi_te.values[idx], len(idx) - 1, keep)); units_te.append(u)
    P = predq(np.array(lastX)); true = np.asarray(rul_true, float)
    med = P[:, 3]; lo, hi = (P[:, 0] - qhat).clip(0), (P[:, -1] + qhat).clip(0, CAP + 20)
    safe = (P[:, 1] - qlow).clip(0)
    rmse = float(np.sqrt(np.mean((med - true) ** 2))); mae = float(np.mean(np.abs(med - true)))
    score = nasa_score(med, true); tc = np.minimum(true, CAP)
    cov = float(np.mean((tc >= lo) & (tc <= hi))); safe_ok = float(np.mean(tc >= safe))
    log(f'[test] RMSE={rmse:.2f} MAE={mae:.2f} NASA-score={score:.0f} 90%-coverage={cov:.1%} safe-horizon-reliability={safe_ok:.1%}')

    pol = {'Run to failure': policy_cost(np.full(len(true), 1e9), true), 'Replace immediately': policy_cost(np.zeros(len(true)), true),
           'Point estimate (median)': policy_cost(med, true), 'Conformal safe horizon': policy_cost(safe, true)}
    pol = {k: float(v.sum()) for k, v in pol.items()}
    saving = 1 - pol['Conformal safe horizon'] / min(pol['Run to failure'], pol['Replace immediately'], pol['Point estimate (median)'])
    log('[cost] ' + ' | '.join(f'{k}: {v:,.0f}' for k, v in pol.items()))

    # attribution, grouped by sensor
    sel = np.random.default_rng(1).choice(len(Xc), min(500, len(Xc)), replace=False)
    pi = permutation_importance(models[.5], Xc[sel], yc[sel], n_repeats=3, random_state=0, scoring='neg_root_mean_squared_error')
    agg = {}
    for nme, v in zip(names(keep), pi.importances_mean):
        k = nme.split('_', 1)[1] if nme.split('_')[0] in ('last', 'mean', 'slope', 'std') else nme
        agg[k] = agg.get(k, 0) + max(v, 0)
    tot = sum(agg.values()) or 1
    imp = sorted(([k, round(100 * v / tot, 2)] for k, v in agg.items()), key=lambda r: -r[1])[:12]

    # per-engine online replay (prediction at many points in each engine's life)
    eng = {}
    for i, (u, idx) in enumerate(te.groupby('unit').indices.items()):
        L = len(idx); ts = list(range(15, L - 1, max(2, L // 45))) + [L - 1]
        Pt = predq(np.array([feats(Zte.values[idx], hi_te.values[idx], t, keep) for t in ts]))
        k = max(1, L // 80)
        eng[int(u)] = dict(t=[t + 1 for t in ts], p=Pt[:, 3].round(1).tolist(), lo=(Pt[:, 0] - qhat).clip(0).round(1).tolist(),
                           up=(Pt[:, -1] + qhat).clip(0, CAP + 20).round(1).tolist(), sf=(Pt[:, 1] - qlow).clip(0).round(1).tolist(),
                           truth=[float(min(true[i] + L - 1 - t, CAP)) for t in ts],
                           hx=list(range(1, L + 1))[::k], hi=hi_te.values[idx][::k].round(3).tolist())
    fleet = [dict(unit=int(u), cycles=int((te.unit == u).sum()), true=float(true[i]), pred=round(float(med[i]), 1),
                  lo=round(float(lo[i]), 1), hi=round(float(hi[i]), 1), safe=round(float(safe[i]), 1),
                  status=status(safe[i]), health=round(float(hi_te[(te.unit == u).values].iloc[-1]), 3))
             for i, u in enumerate(units_te)]
    data = dict(meta=dict(dataset=fd, source='NASA C-MAPSS' if real else 'Synthetic C-MAPSS-like demo data', cap=CAP, lead=LEAD,
                          cost=COST, regimes=6 if multi else 1, sensors=len(keep), seconds=round(time.time() - t0, 1)),
                metrics=dict(rmse=round(rmse, 2), mae=round(mae, 2), nasa=round(score), coverage=round(cov * 100, 1),
                             safe=round(safe_ok * 100, 1), saving=round(saving * 100, 1)),
                policies=pol, importance=imp, fleet=fleet, engines=eng)
    (out / 'metrics.json').write_text(json.dumps({**data['metrics'], 'policies': pol, 'source': data['meta']['source']}, indent=2))
    tpl = Path(__file__).with_name('dashboard_template.html').read_text(encoding='utf-8')
    dash = out.parent / 'dashboard' / 'index.html'; dash.parent.mkdir(exist_ok=True)
    dash.write_text(tpl.replace('/*__DATA__*/null', json.dumps(data)), encoding='utf-8')
    log(f'[done] dashboard -> {dash}  ({time.time() - t0:.0f}s)')
    return data
