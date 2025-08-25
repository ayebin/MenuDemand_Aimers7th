import os

import glob, time, random, re
from tqdm import tqdm

import numpy as np
import pandas as pd
import holidays

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


# ---------- Utils ----------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    

def month_to_season(m: int) -> int:
    if m in [12,1,2]: return 0
    if m in [3,4,5]: return 1
    if m in [6,7,8]: return 2
    return 3


# ---------- Data Preprocess ----------
def ensure_long_train(path: str) -> str:
    df = pd.read_csv(path)
    if "영업일자" in df.columns and ("영업장명_메뉴명" not in df.columns):
        id_col = "영업일자"
        value_cols = [c for c in df.columns if c != id_col]
        long_df = df.melt(id_vars=[id_col], value_vars=value_cols,
                          var_name="영업장명_메뉴명", value_name="매출수량")
    else:
        long_df = df.copy()

    long_df["영업일자"] = pd.to_datetime(long_df["영업일자"])
    all_dates = pd.date_range(long_df["영업일자"].min(), long_df["영업일자"].max(), freq="D")
    keys = long_df["영업장명_메뉴명"].astype(str).unique()
    full_index = pd.MultiIndex.from_product([keys, all_dates], names=["영업장명_메뉴명","영업일자"])

    long_df = (long_df.astype({"영업장명_메뉴명":"string"})
        .set_index(["영업장명_메뉴명","영업일자"])
        .reindex(full_index)
        .reset_index()
        .rename(columns={"level_0":"영업장명_메뉴명","level_1":"영업일자"}))

    long_df["매출수량"] = pd.to_numeric(long_df["매출수량"], errors="coerce").fillna(0).clip(lower=0)

    out = "./data/pivot_holiday_train_long.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    long_df.to_csv(out, index=False)
    return out


def smape_ignore_zero(a, p, eps=1e-8):
    a = np.asarray(a, float); p = np.asarray(p, float)
    mask = (a != 0)
    if not np.any(mask): return np.nan
    num = 2.0 * np.abs(a[mask] - p[mask])
    den = np.abs(a[mask]) + np.abs(p[mask]) + eps
    return float(np.mean(num / den))


# ---------- Availability helpers ----------
TEMP_PAUSE_THRESHOLD = 14 

def _runlen_ones(arr: np.ndarray) -> np.ndarray:
    """arr: 0/1 array -> 현재 위치까지의 '연속 1 길이'"""
    out = np.zeros_like(arr, dtype=int)
    run = 0
    for i, z in enumerate(arr):
        if z == 1:
            run += 1
        else:
            run = 0
        out[i] = run
    return out

def add_causal_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - is_available_causal: '이전날까지' 양 판매가 한 번이라도 있었으면 1
    - is_temp_paused_causal: 출시 후 lag 기준 연속 0 run >= TEMP_PAUSE_THRESHOLD
    - is_selling_7_causal: 직전 7일(오늘 제외) 중 양 판매 ANY
    """
    df = df.sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)

    pos_lag1 = df.groupby("영업장명_메뉴명")["매출수량"].shift(1).fillna(0) > 0
    pos_lag1 = pos_lag1.astype(int)

    cum_pos_before = pos_lag1.groupby(df["영업장명_메뉴명"]).cumsum()
    df["is_available_causal"] = (cum_pos_before > 0).astype(int)

    zero_lag1 = (df.groupby("영업장명_메뉴명")["매출수량"].shift(1).fillna(0) <= 0).astype(int)
    zero_run_lag = zero_lag1.groupby(df["영업장명_메뉴명"]).transform(
        lambda s: pd.Series(_runlen_ones(s.to_numpy()), index=s.index)
    )
    df["is_temp_paused_causal"] = ((df["is_available_causal"] == 1) & (zero_run_lag >= TEMP_PAUSE_THRESHOLD)).astype(int)

    rolling_any7 = pos_lag1.groupby(df["영업장명_메뉴명"]).rolling(7, min_periods=1).max()
    df["is_selling_7_causal"] = rolling_any7.reset_index(level=0, drop=True).fillna(0).astype(int)

    return df


# ---------- Feature Engineering ----------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "영업장명_메뉴명" in df.columns and (("업장명" not in df.columns) or ("메뉴명" not in df.columns)):
        parts = df["영업장명_메뉴명"].astype(str).str.split("_", n=1, expand=True)
        df["업장명"] = parts[0]
        df["메뉴명"] = parts[1].fillna("UNKNOWN")
    if "영업장명_메뉴명" not in df.columns:
        df["영업장명_메뉴명"] = df["업장명"].astype(str) + "_" + df["메뉴명"].astype(str)

    if "매출수량" not in df.columns:
        df["매출수량"] = 0

    df["영업일자"] = pd.to_datetime(df["영업일자"])
    df = df.sort_values(["영업장명_메뉴명","영업일자"]).reset_index(drop=True)
    df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0).clip(lower=0)

    df["월"] = df["영업일자"].dt.month
    df["요일"] = df["영업일자"].dt.weekday
    df["day_of_year"] = df["영업일자"].dt.dayofyear
    df["season_id"] = df["월"].apply(month_to_season).astype(int)

    df["doy_sin"] = np.sin(2*np.pi*(df["day_of_year"]-1)/365.0)
    df["doy_cos"] = np.cos(2*np.pi*(df["day_of_year"]-1)/365.0)
    for k in [2, 3]:
        df[f'doy_sin_{k}'] = np.sin(2*np.pi*k*(df["day_of_year"]-1)/365.0)
        df[f'doy_cos_{k}'] = np.cos(2*np.pi*k*(df["day_of_year"]-1)/365.0)

    df["is_weekend"] = df["요일"].isin([5,6]).astype(int)
    df["is_summer_peak"] = df["월"].isin([7,8]).astype(int)
    df["is_winter_peak"] = df["월"].isin([12,1,2]).astype(int)

    start_year = int(df["영업일자"].min().year)-1
    end_year   = int(df["영업일자"].max().year)+1
    kr_holidays = holidays.KR(years=range(start_year, end_year+1), observed=True)
    holi_idx = pd.DatetimeIndex(pd.to_datetime(list(kr_holidays.keys()))).tz_localize(None)
    d = df["영업일자"].dt.normalize()
    df["is_holiday"] = d.isin(holi_idx).astype(int)
    df["is_day_before_holiday"] = (d + pd.Timedelta(days=1)).isin(holi_idx).astype(int)
    df["is_day_after_holiday"]  = (d - pd.Timedelta(days=1)).isin(holi_idx).astype(int)

    grp = df.groupby("영업장명_메뉴명")["매출수량"]
    df["sales_lag_1"] = grp.shift(1).fillna(0)
    df["sales_rolling_mean_7"]  = grp.shift(1).rolling(7,  min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df["sales_rolling_mean_14"] = grp.shift(1).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df["sales_rolling_mean_28"] = grp.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df["rolling_std_7"]  = grp.shift(1).rolling(7,  min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
    df["rolling_std_14"] = grp.shift(1).rolling(14, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
    df["rolling_std_28"] = grp.shift(1).rolling(28, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)

    df["월시작"] = df["영업일자"].values.astype("datetime64[M]")
    vym = (df.groupby(["업장명","메뉴명","월시작"], as_index=False)["매출수량"].mean()
             .rename(columns={"매출수량":"월평균"}))
    vym["avg_sales_prev_month"] = (
        vym.sort_values(["업장명","메뉴명","월시작"])
           .groupby(["업장명","메뉴명"])["월평균"].shift(1)
    )
    df = df.merge(
        vym.loc[:, ["업장명","메뉴명","월시작","avg_sales_prev_month"]],
        on=["업장명","메뉴명","월시작"], how="left"
    )
    df["avg_sales_prev_month"] = df["avg_sales_prev_month"].fillna(0.0)
    
    r7  = grp.shift(1).rolling(7,  min_periods=1).mean().reset_index(level=0, drop=True)
    r28 = grp.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    df["recent_momentum"] = (r7 - r28).fillna(0).astype(float)

    special_winter = {"포레스트릿", "카페테리아"}
    special_spring_fall = {"화담숲주막", "화담숲카페"}
    always_zero = {"담하", "미라시아", "느티나무", "연회장", "라그로타"}
    def _seasonal_flag(store: str, month: int) -> int:
        if store in special_winter and month in (12, 1, 2): return 1
        if store in special_spring_fall and month in (4, 5, 6, 10, 11, 12): return 1
        if store in always_zero: return 0
        return 0
    df["seasonal_weight_flag"] = [
        _seasonal_flag(s, m) for s, m in zip(df["업장명"].astype(str), df["월"].astype(int))
    ]

    df = df.sort_values(["업장명","메뉴명","영업일자"]).reset_index(drop=True)
    df["_pos_date"] = df["영업일자"].where(df["매출수량"] > 0)
    df["last_pos_date"] = df.groupby(["업장명","메뉴명"])["_pos_date"].transform('ffill')
    df["days_since_last_pos_sale"] = (df["영업일자"] - df["last_pos_date"]).dt.days.fillna(9999).astype(int)
    df.drop(columns=["_pos_date","last_pos_date"], inplace=True)

    df["is_zero"] = (df["매출수량"] <= 0).astype(int)
    zgrp = df.groupby("영업장명_메뉴명")["is_zero"]
    df["zero_rate_7"]  = zgrp.shift(1).rolling(7,  min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df["zero_rate_14"] = zgrp.shift(1).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df["zero_rate_28"] = zgrp.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df.drop(columns=["is_zero"], inplace=True)

    df = add_causal_availability_features(df)

    for c in ['is_weekend','is_holiday','is_day_before_holiday','is_day_after_holiday',
              'is_summer_peak','is_winter_peak','seasonal_weight_flag',
              'is_available_causal','is_temp_paused_causal','is_selling_7_causal']:
        if c in df.columns: df[c] = df[c].astype(int)

    return df

def local_scale_window(X, feature_names, cols_to_scale):
    X = X.copy()
    idx = [feature_names.index(c) for c in cols_to_scale if c in feature_names]
    if not idx: return X
    for n in range(X.shape[0]):
        for j in idx:
            seq = X[n, :, j]
            med = np.median(seq)
            iqr = np.percentile(seq, 75) - np.percentile(seq, 25)
            if iqr <= 1e-6: iqr = 1.0
            X[n, :, j] = (seq - med) / iqr
    return X

def create_sequences_multi(data, L, H, feats, s2i, i2i):
    Xs, ys, dows, mons, seas = [], [], [], [], []
    fdw, fmn, fse, sids, iids = [], [], [], [], []
    favails, fpauses = [], []
    fut_end_dates = []

    for name, g in data.groupby("영업장명_메뉴명"):
        g = g.sort_values("영업일자")
        x   = g[feats].values
        y   = g["매출수량"].values
        dow = g["요일"].astype(int).values
        mon = (g["월"].astype(int)-1).values
        sea = g["season_id"].astype(int).values
        fav = g["is_available_causal"].astype(int).values
        fpa = g["is_temp_paused_causal"].astype(int).values

        sid = s2i.get(g["업장명"].iloc[0], s2i["<UNK_STORE>"])
        iid = i2i.get(g["메뉴명"].iloc[0], i2i["<UNK_ITEM>"])
        dates = pd.to_datetime(g["영업일자"].values)

        for i in range(len(g) - L - H + 1):
            Xs.append(x[i:i+L]); ys.append(y[i+L:i+L+H])
            dows.append(dow[i:i+L]); mons.append(mon[i:i+L]); seas.append(sea[i:i+L])
            sids.append(sid); iids.append(iid)

            td = dates[i+L:i+L+H]
            idx = pd.DatetimeIndex(td)
            fdw.append(idx.weekday.values.astype(int))
            fmn.append((idx.month.values.astype(int)-1))
            fse.append(np.array([month_to_season(m) for m in idx.month.values], dtype=int))
            fut_end_dates.append(np.datetime64(td[-1], 'ns'))

            favails.append(fav[i+L:i+L+H])
            fpauses.append(fpa[i+L:i+L+H])

    return (
        np.array(Xs), np.array(ys), np.array(dows), np.array(mons), np.array(seas),
        np.array(fdw), np.array(fmn), np.array(fse),
        np.array(favails), np.array(fpauses),
        np.array(sids), np.array(iids),
        np.array(fut_end_dates, dtype='datetime64[ns]')
    )
    
# ---------- Model ----------
class HurdleLSTMFC(nn.Module):
    def __init__(self, input_size_num, n_stores, n_items,
                 emb_store=8, emb_item=12, emb_dow=4, emb_mon=4, emb_season=3,
                 hidden=128, layers=1, dropout=0.0, horizon=7):
        super().__init__()
        self.horizon=horizon
        self.emb_store  = nn.Embedding(n_stores, emb_store)
        self.emb_item   = nn.Embedding(n_items,  emb_item)
        self.emb_dow    = nn.Embedding(7,  emb_dow)
        self.emb_mon    = nn.Embedding(12, emb_mon)
        self.emb_season = nn.Embedding(4,  emb_season)
        feat_in = input_size_num + emb_store + emb_item + emb_dow + emb_mon + emb_season
        
        self.lstm = nn.LSTM(feat_in, hidden, layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)

        self.proj_q = nn.Linear(hidden, hidden, bias=False)
        self.proj_k = nn.Linear(hidden, hidden, bias=False)
        self.proj_v = nn.Linear(hidden, hidden, bias=False)

        self.emb_dow_f    = self.emb_dow
        self.emb_mon_f    = self.emb_mon
        self.emb_season_f = self.emb_season

        cond_dim = hidden + emb_dow + emb_mon + emb_season
        self.step_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_zero = nn.Linear(hidden, 1)
        self.out_qty  = nn.Linear(hidden, 1)
        self.logit_temp = nn.Parameter(torch.tensor(1.3))

    def forward(self, x_num, dow_seq, mon_seq, season_seq,
                store_id, item_id,
                fut_dow, fut_mon, fut_season):
        B, L, _ = x_num.shape
        e = torch.cat([
            x_num,
            self.emb_dow(dow_seq),
            self.emb_mon(mon_seq),
            self.emb_season(season_seq),
            self.emb_store(store_id).unsqueeze(1).expand(B, L, -1),
            self.emb_item(item_id).unsqueeze(1).expand(B, L, -1),
        ], dim=2)

        out, _ = self.lstm(e)
        q = self.proj_q(out[:, -1, :]).unsqueeze(1)
        k = self.proj_k(out)
        v = self.proj_v(out)
        attn = torch.softmax((q @ k.transpose(1,2)) / (k.size(-1) ** 0.5), dim=-1)
        ctx = (attn @ v).squeeze(1)
        
        Hh = fut_dow.size(1)
        f_e = torch.cat([self.emb_dow_f(fut_dow),
                         self.emb_mon_f(fut_mon),
                         self.emb_season_f(fut_season)], dim=-1)
        ctx_rep = ctx.unsqueeze(1).expand(B, Hh, ctx.size(-1))
        z = torch.cat([ctx_rep, f_e], dim=-1).reshape(B*Hh, -1)
        
        h = self.step_mlp(z)
        logits = self.out_zero(h)
        p = torch.sigmoid(logits * self.logit_temp).view(B, Hh)
        qy = self.out_qty(h).view(B, Hh)  
        return p, qy


def focal_bce_elementwise(pred, target, gamma=1.0, alpha=0.75, eps=1e-7):
    """returns per-step loss (B,H) without reduction"""
    pred = pred.clamp(eps, 1-eps)
    bce  = -(alpha*target*torch.log(pred) + (1-alpha)*(1-target)*torch.log(1-pred))
    pt   = torch.where(target==1, pred, 1-pred)
    return bce * ((1-pt)**gamma)


# ---------- metric & postprocess tuning ----------
def _comp_smape(a, p, eps=1e-8):
    a = np.asarray(a, float); p = np.asarray(p, float)
    m = (a != 0)
    if not np.any(m): return np.nan
    return float(np.mean(2.0*np.abs(a[m]-p[m])/(np.abs(a[m])+np.abs(p[m])+eps)))

def competition_score(df, store_weights=None):
    if store_weights is None: store_weights = {}
    scores, wts = [], []
    for s, g in df.groupby('restaurant', sort=False):
        item_scores = []
        for _, gi in g.groupby('item', sort=False):
            gi = gi[gi['A']>0]
            if len(gi)==0: continue
            item_scores.append(_comp_smape(gi['A'], gi['P']))
        if not item_scores: continue
        scores.append(np.mean(item_scores))
        wts.append(float(store_weights.get(s, 1.0)))
    if not scores: return np.nan
    scores = np.array(scores); wts = np.array(wts)
    return float((scores*wts).sum()/wts.sum())

def apply_theta_eps(df, theta, eps, pcol='p_eff', qcol='q'):
    if theta is None or eps is None:
        out = df.copy()
        out['P'] = out.get('P', out[qcol].astype(float)).astype(float)
        return out
    out = df.copy()
    pred = (out[pcol].astype(float) > float(theta)).astype(float) * out[qcol].astype(float)
    out['P'] = np.maximum(float(eps), pred)
    return out


def tune_theta_eps_per_store(val_df, thetas=None, floors=None, store_weights=None):
    if thetas is None: thetas = np.linspace(0.05, 0.45, 9)
    if floors is None: floors = np.linspace(0.0, 0.25, 6)
    params = {}; outs = []
    for s, g in val_df.groupby('restaurant', sort=False):
        best = (None, None, np.inf)
        for th in thetas:
            for eps in floors:
                sc = competition_score(
                    apply_theta_eps(g, th, eps)[['restaurant','item','date','A','P']],
                    store_weights={s:1.0}
                )
                if not np.isfinite(sc):
                    continue 
                if sc < best[2]:
                    best = (float(th), float(eps), float(sc))
        theta, eps, best_sc = best
        if theta is None or eps is None:
            continue
        g2 = apply_theta_eps(g, theta, eps)
        outs.append(g2)
        params[s] = {'theta':theta, 'eps':eps}
    val_tuned = pd.concat(outs, ignore_index=True) if outs else val_df.copy()
    total_sc = competition_score(val_tuned[['restaurant','item','date','A','P']], store_weights)
    return val_tuned, params, float(total_sc) if np.isfinite(total_sc) else np.inf

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fit_temperature_per_store(val_df_post, stores, Ts=np.linspace(0.6, 2.5, 20)):
    """p_eff에 대한 Platt 온도(T)를 스토어 단위로 그리드서치."""
    out = {}
    for s in stores:
        g = val_df_post[val_df_post['restaurant'] == s].copy()
        if len(g) == 0:
            continue
        y = (g['A'] > 0).astype(float).values
        p = np.clip(g['p_eff'].astype(float).values, 1e-6, 1 - 1e-6)
        z = _logit(p)
        best = (1.0, 1e9)
        for T in Ts:
            ph = _sigmoid(z / T)
            bce = -(y * np.log(ph) + (1 - y) * np.log(1 - ph)).mean()
            if bce < best[1]:
                best = (float(T), float(bce))
        out[s] = best[0]
    return out

def tune_theta_eps_per_store_item(val_df, thetas=None, floors=None, store_weights=None, min_rows=60):
    """스토어×아이템 단위 θ/ε (표본 적으면 스킵), 부족분은 스토어 단위로 fallback"""
    if thetas is None: thetas = np.linspace(0.05, 0.45, 9)
    if floors is None: floors = np.linspace(0.0, 0.25, 6)
    params_si = {}; outs = []
    covered_idx = set()

    for (s, it), g in val_df.groupby(['restaurant', 'item'], sort=False):
        if len(g) < min_rows:
            continue
        best = (None, None, np.inf)
        for th in thetas:
            for eps in floors:
                sc = competition_score(
                    apply_theta_eps(g, th, eps)[['restaurant','item','date','A','P']],
                    store_weights={s:1.0}
                )
                if not np.isfinite(sc):
                    continue 
                if sc < best[2]:
                    best = (float(th), float(eps), float(sc))
        theta, eps, best_sc = best
        if theta is None or eps is None:
            continue
        g2 = apply_theta_eps(g, theta, eps)
        outs.append(g2)
        params_si[(s, it)] = {'theta': theta, 'eps': eps}
        covered_idx.update(g.index.tolist())

    rem_idx = sorted(set(val_df.index) - covered_idx)
    if rem_idx:
        rem = val_df.loc[rem_idx].copy()
        base, params_s, _ = tune_theta_eps_per_store(rem, thetas=thetas, floors=floors, store_weights=store_weights)
        if len(base):
            outs.append(base)
    else:
        params_s = {}

    val_tuned = pd.concat(outs, ignore_index=True) if outs else val_df.copy()
    return val_tuned, params_si, params_s


def affine_calib_per_store(dfP, store_weights=None, alphas=None, betas=None):
    if alphas is None: alphas = np.linspace(0.7, 1.5, 17)
    if betas  is None: betas  = np.linspace(-0.25, 0.25, 11)
    params = {}; outs = []
    for s, g in dfP.groupby('restaurant', sort=False):
        best = (1.0, 0.0, 1e9)
        for a in alphas:
            for b in betas:
                tmp = g.copy()
                tmp['P'] = np.maximum(0.0, a*tmp['P'].astype(float) + b)
                sc = competition_score(tmp[['restaurant','item','date','A','P']], store_weights={s:1.0})
                if sc < best[2]:
                    best = (float(a), float(b), float(sc))
        a, b, _ = best
        g2 = g.copy(); g2['P'] = np.maximum(0.0, a*g2['P'] + b)
        outs.append(g2)
        params[s] = {'alpha':a, 'beta':b}
    out = pd.concat([*outs], ignore_index=True) if outs else dfP.copy()
    score = competition_score(out[['restaurant','item','date','A','P']], store_weights)
    return out, params, float(score)

def compute_caps_from_train(train_long, q=0.95):
    df = train_long.copy()
    df['매출수량'] = pd.to_numeric(df['매출수량'], errors='coerce').fillna(0)
    pos = df[df['매출수량']>0]
    if len(pos)==0:
        return {}
    caps = (pos.groupby(['업장명','메뉴명'])['매출수량']
              .quantile(q).reset_index()
              .rename(columns={'매출수량':'cap'}))
    caps['cap'] = caps['cap'].astype(float).clip(lower=0)
    caps_map = {(r['업장명'], r['메뉴명']): float(r['cap']) for _, r in caps.iterrows()}
    return caps_map


# ---------- Train + Val (Opt) -> Save Artifacts ----------
def train_full_and_pack(train_path: str):
    print("Loading train:", train_path)
    df = pd.read_csv(train_path)
    df = build_features(df)

    last_dt  = pd.to_datetime(df["영업일자"]).max()
    val_start= last_dt - pd.Timedelta(days=VAL_DAYS-1)
    val_end  = last_dt
    if TRAIN_FULL:
        print("TRAIN_FULL=True: 전체 기간으로 학습 (검증/얼리스톱 없음)")
    else:
        print(f"Validation window: {val_start.date()} - {val_end.date()}")
        
    if TRAIN_FULL:
        caps_source = df[['업장명','메뉴명','영업일자','매출수량']].copy()
        caps_map = compute_caps_from_train(caps_source, q=CAP_Q)
    else:
        caps_source = df[['업장명','메뉴명','영업일자','매출수량']].copy()
        caps_train_only = caps_source[caps_source['영업일자'] < val_start]
        caps_map = compute_caps_from_train(caps_train_only, q=CAP_Q)

    features = [
        '매출수량',
        'sales_lag_1','sales_rolling_mean_7','sales_rolling_mean_14',
        'sales_rolling_mean_28','rolling_std_7','rolling_std_14','rolling_std_28',
        'avg_sales_prev_month',
        'is_weekend','is_holiday','is_day_before_holiday','is_day_after_holiday',
        'is_summer_peak','is_winter_peak',
        'doy_sin','doy_cos','doy_sin_2','doy_cos_2','doy_sin_3','doy_cos_3',  
        'seasonal_weight_flag',
        'days_since_last_pos_sale',
        'recent_momentum',
        'zero_rate_7','zero_rate_14','zero_rate_28',
        'is_available_causal','is_temp_paused_causal','is_selling_7_causal',
    ]
    for c in features:
        if c not in df.columns: df[c] = 0
    df[features] = df[features].replace([np.inf,-np.inf],0).fillna(0)

    log_cols = [
        '매출수량','sales_lag_1','sales_rolling_mean_7',
        'sales_rolling_mean_14','sales_rolling_mean_28','avg_sales_prev_month',
    ]
    df[log_cols] = np.log1p(df[log_cols])

    to_scale = [c for c in features if c != '매출수량']
    x_scaler = MinMaxScaler()

    if not TRAIN_FULL and VAL_DAYS > 0:
        train_df = df[df['영업일자'] < val_start].copy()
        val_df = df[df['영업일자'] >= val_start].copy()
        print(f"Fitting scaler ONLY on train data (before {val_start.date()})...")
        x_scaler.fit(train_df[to_scale])
        train_df[to_scale] = x_scaler.transform(train_df[to_scale])
        val_df[to_scale]   = x_scaler.transform(val_df[to_scale])
        data_for_seq = pd.concat([train_df, val_df])
    else:
        print("TRAIN_FULL is True. Fitting scaler on all data...")
        df[to_scale] = x_scaler.fit_transform(df[to_scale])
        data_for_seq = df

    stores = ["<UNK_STORE>"] + sorted(df["업장명"].unique().tolist())
    items  = ["<UNK_ITEM>"] + sorted(df["메뉴명"].unique().tolist())
    store2id = {s:i for i,s in enumerate(stores)}
    item2id  = {m:i for i,m in enumerate(items)}
    id2store = {i:s for s,i in store2id.items()}
    id2item  = {i:m for m,i in item2id.items()}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (Xall,yall,dwall,mnall,seall,fdwall,fnall,fseall,
     favall, fpall, sidall,iidall,fend) = create_sequences_multi(
        data_for_seq, SEQ_LEN, H, features, store2id, item2id
    )
    fend = np.asarray(fend).astype('datetime64[ns]')

    if TRAIN_FULL:
        train_mask = np.ones_like(fend, dtype=bool)
        val_mask   = np.zeros_like(fend, dtype=bool)
    else:
        train_mask = (fend <  np.datetime64(val_start))
        val_mask   = (fend >= np.datetime64(val_start)) & (fend <= np.datetime64(val_end))

    n_train = int(train_mask.sum())
    n_val   = int(val_mask.sum())
    print(f"• Split windows | train:{n_train}  val:{n_val}")

    local_cols = ['sales_lag_1','sales_rolling_mean_7','sales_rolling_mean_14',
                  'sales_rolling_mean_28','rolling_std_7','rolling_std_14',
                  'rolling_std_28','avg_sales_prev_month','recent_momentum',
                  'zero_rate_7','zero_rate_14','zero_rate_28',]

    def to_tensor_pack(Xn,dw,mn,se,fdw,fmn,fse,fav,fpause,sid,iid,y,mask):
        Xn, dw, mn, se = Xn[mask], dw[mask], mn[mask], se[mask]
        fdw, fmn, fse = fdw[mask], fmn[mask], fse[mask]
        fav, fpause   = fav[mask], fpause[mask]
        sid, iid, y   = sid[mask], iid[mask], y[mask]
        Xn = local_scale_window(Xn, features, local_cols)
        return (
            torch.from_numpy(Xn).float(), torch.from_numpy(dw).long(),
            torch.from_numpy(mn).long(), torch.from_numpy(se).long(),
            torch.from_numpy(fdw).long(), torch.from_numpy(fmn).long(),
            torch.from_numpy(fse).long(), torch.from_numpy(fav).float(),
            torch.from_numpy(fpause).float(), torch.from_numpy(sid).long(),
            torch.from_numpy(iid).long(), torch.from_numpy(y).float(),
        )

    (Xtr_t,dwtr_t,mntr_t,setr_t,fdwtr_t,fmntr_t,fsetr_t,favtr_t,fpautr_t,
     sidtr_t,iidtr_t,ytr_t) = to_tensor_pack(
        Xall,dwall,mnall,seall,fdwall,fnall,fseall,favall,fpall,sidall,iidall,yall, mask = train_mask
    )
    if n_val > 0:
        (Xv_t,dwv_t,mnv_t,sev_t,fdwv_t,fmnv_t,fsev_t,favv_t,fpauv_t,
         sidv_t,iidv_t,yv_t) = to_tensor_pack(
            Xall,dwall,mnall,seall,fdwall,fnall,fseall,favall,fpall,sidall,iidall,yall, mask = val_mask
        )

    class DS(torch.utils.data.Dataset):
        def __init__(self, Xn,dw,mn,se,fdw,fmn,fse,fav,fpause,sid,iid,y):
            self.Xn,self.dw,self.mn,self.se=Xn,dw,mn,se
            self.fdw,self.fmn,self.fse=fdw,fmn,fse
            self.fav,self.fpause=fav,fpause
            self.sid,self.iid,self.y=sid,iid,y
        def __len__(self): return self.Xn.size(0)
        def __getitem__(self,i):
            return (self.Xn[i],self.dw[i],self.mn[i],self.se[i],
                    self.fdw[i],self.fmn[i],self.fse[i],self.fav[i],self.fpause[i],
                    self.sid[i],self.iid[i],self.y[i])

    train_loader = DataLoader(
        DS(Xtr_t,dwtr_t,mntr_t,setr_t,fdwtr_t,fmntr_t,fsetr_t,favtr_t,fpautr_t,sidtr_t,iidtr_t,ytr_t),
        batch_size=BATCH, shuffle=True, num_workers=0
    )
    if n_val > 0:
        val_loader = DataLoader(
            DS(Xv_t,dwv_t,mnv_t,sev_t,fdwv_t,fmnv_t,fsev_t,favv_t,fpauv_t,sidv_t,iidv_t,yv_t),
            batch_size=BATCH, shuffle=False, num_workers=0
        )

    emb_store = max(4, int(np.ceil(np.log2(len(store2id) + 1))))
    emb_item  = max(8, int(np.ceil(np.log2(len(item2id) + 1))) + 2)
    model = HurdleLSTMFC(
        input_size_num=len(features), n_stores=len(store2id), n_items=len(item2id),
        emb_store=emb_store, emb_item=emb_item, emb_dow=4, emb_mon=4, emb_season=3,
        hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, horizon=H
    ).to(device)

    huber = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    print(f"Start training on {device}")
    t0 = time.time()

    best_s = float('inf')
    best_state = None
    bad = 0

    damha_id = store2id.get('담하', -1) 
    mirasia_id = store2id.get('미라시아', -1)
    HIGH_WEIGHT = HW

    state_at_snapshot = None

    # ---------------- Train Loop ----------------
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for Xn,dw,mn,se,fdw,fmn,fse,fav,fpause,sid,iid,yb in train_loader:
            Xn,dw,mn,se,fdw,fmn,fse,fav,fpause,sid,iid,yb = \
                Xn.to(device),dw.to(device),mn.to(device),se.to(device),fdw.to(device),fmn.to(device),fse.to(device),fav.to(device),fpause.to(device),sid.to(device),iid.to(device),yb.to(device)

            weights_batch = torch.ones(Xn.size(0), device=device)
            mask_hi = ((sid == damha_id) | (sid == mirasia_id))
            weights_batch = torch.where(mask_hi, torch.as_tensor(HIGH_WEIGHT, device=device, dtype=weights_batch.dtype), weights_batch)

            optimizer.zero_grad()
            p_buy, qty_log = model(Xn,dw,mn,se,sid,iid,fdw,fmn,fse)

            mask_weight = WEIGHT_PRELAUNCH + (1.0 - WEIGHT_PRELAUNCH) * fav
            mask_weight = mask_weight * torch.where(fpause>0.5, torch.as_tensor(WEIGHT_PAUSED, device=device), torch.as_tensor(1.0, device=device))

            tgt_bin = (yb > 0).float()
            loss_zero_raw = focal_bce_elementwise(p_buy, tgt_bin, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)  # (B,H)
            loss_zero_per = (loss_zero_raw * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1e-6)

            mask_pos = (yb > 0).float()
            qty_raw = huber(qty_log, yb) * mask_pos
            denom_qty = (mask_pos * mask_weight).sum(dim=1).clamp_min(1e-6)
            loss_qty_per = (qty_raw * mask_weight).sum(dim=1) / denom_qty

            total_per = LAMBDA_ZERO * loss_zero_per + LAMBDA_QTY * loss_qty_per
            weights_batch = weights_batch / weights_batch.mean().clamp_min(1e-6)
            loss = (total_per * weights_batch).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * Xn.size(0)

        tr_loss /= max(len(train_loader.dataset), 1)

        if TRAIN_FULL or n_val == 0:
            print(f"  - epoch {epoch:02d} | train {tr_loss:.4f}")
        else:
            # ---------- Validation ----------
            model.eval()
            va_loss = 0.0
            all_trues, all_preds_q, all_preds_qp, all_sids, all_iids, all_dates = [], [], [], [], [], []

            with torch.no_grad():
                for Xn,dw,mn,se,fdw,fmn,fse,fav,fpause,sid,iid,yb in val_loader:
                    Xn,dw,mn,se,fdw,fmn,fse,fav,fpause,sid,iid,yb = \
                        Xn.to(device),dw.to(device),mn.to(device),se.to(device),fdw.to(device),fmn.to(device),fse.to(device),fav.to(device),fpause.to(device),sid.to(device),iid.to(device),yb.to(device)

                    p_buy, qty_log = model(Xn,dw,mn,se,sid,iid,fdw,fmn,fse)

                    mask_weight = WEIGHT_PRELAUNCH + (1.0 - WEIGHT_PRELAUNCH) * fav
                    mask_weight = mask_weight * torch.where(fpause>0.5, torch.as_tensor(WEIGHT_PAUSED, device=device), torch.as_tensor(1.0, device=device))

                    tgt_bin = (yb > 0).float()
                    loss_zero_raw = focal_bce_elementwise(p_buy, tgt_bin, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)  # (B,H)
                    loss_zero_per = (loss_zero_raw * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1e-6)

                    mask_pos = (yb > 0).float()
                    qty_raw = huber(qty_log, yb) * mask_pos
                    denom_qty = (mask_pos * mask_weight).sum(dim=1).clamp_min(1e-6)
                    loss_qty_per = (qty_raw * mask_weight).sum(dim=1) / denom_qty

                    va_loss += (LAMBDA_ZERO*loss_zero_per + LAMBDA_QTY*loss_qty_per).mean().item() * Xn.size(0)

                    q = torch.expm1(qty_log).clamp(min=0)
                    p = p_buy
                    yhat_qp = (q * p).cpu().numpy()
                    yhat_q  = q.cpu().numpy()
                    ytrue   = torch.expm1(yb).cpu().numpy()

                    B = ytrue.shape[0]
                    repeated_sids = sid.cpu().numpy().repeat(H)
                    repeated_iids = iid.cpu().numpy().repeat(H)

                    all_trues.append(ytrue.ravel())
                    all_preds_q.append(yhat_q.ravel())
                    all_preds_qp.append(yhat_qp.ravel())
                    all_sids.append(repeated_sids)
                    all_iids.append(repeated_iids)
                    all_dates.append(np.arange(H*B))

            va_loss /= max(len(val_loader.dataset), 1)

            all_trues = np.concatenate(all_trues)
            all_preds_q = np.concatenate(all_preds_q)
            all_preds_qp = np.concatenate(all_preds_qp)
            all_sids = np.concatenate(all_sids)
            all_iids = np.concatenate(all_iids)
            _dates_dummy = np.concatenate(all_dates)

            val_results = pd.DataFrame({
                'sid': all_sids,
                'iid': all_iids,
                'true': all_trues,
                'pred_q': all_preds_q,
                'pred_qp': all_preds_qp
            })

            def _weighted_smape_from(val_res, use_q=True):
                pred_col = 'pred_q' if use_q else 'pred_qp'
                smape_by_item = val_res.groupby(['sid','iid']).apply(
                    lambda x: smape_ignore_zero(x['true'], x[pred_col]),
                    include_groups=False
                )
                smape_by_store = smape_by_item.groupby(level=0).mean()
                weights_by_store = smape_by_store.index.map(
                    lambda sid: HIGH_WEIGHT if sid in [damha_id, mirasia_id] else 1.0
                )
                return float(np.average(smape_by_store.fillna(0), weights=weights_by_store))

            va_smape_q_weighted  = _weighted_smape_from(val_results, use_q=True)
            va_smape_qp_weighted = _weighted_smape_from(val_results, use_q=False)

            if EARLYSTOP_ON == "q_only":
                va_smape = va_smape_q_weighted
            elif EARLYSTOP_ON == "qp":
                va_smape = va_smape_qp_weighted
            else:
                va_smape = min(va_smape_q_weighted, va_smape_qp_weighted)

            print(f"  - epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | weighted sMAPE(q*p) {va_smape_qp_weighted:.4f} | weighted sMAPE(q-only) {va_smape_q_weighted:.4f}")

            if va_smape < best_s - 1e-5:
                best_s = va_smape
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
                best_val_results = val_results.copy()  
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"Early stop at epoch {epoch} (best wSMAPE={best_s:.4f})")
                    break

        if SNAPSHOT_EPOCH is not None and epoch == SNAPSHOT_EPOCH:
            state_at_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if (not TRAIN_FULL) and (best_state is not None):
        model.load_state_dict(best_state)
        val_results = best_val_results  

    if SNAPSHOT_EPOCH is not None:
        if state_at_snapshot is not None:
            model.load_state_dict(state_at_snapshot)
            print(f"[SNAPSHOT] Loaded weights from epoch {SNAPSHOT_EPOCH} for final artifacts.")
        else:
            print(f"[SNAPSHOT] WARNING: epoch {SNAPSHOT_EPOCH} snapshot not found (EPOCHS<{SNAPSHOT_EPOCH}?). Using last epoch weights.")

    print(f"Training done in {time.time()-t0:.1f}s")

    # ---------------- Postprocess on Validation ----------------
    post_params = {'theta_eps':{}, 'affine':{}, 'caps':caps_map, 'scores':{}}
    if (not TRAIN_FULL) and (n_val > 0):
        val_df_post = val_results.copy()
        val_df_post['restaurant'] = val_df_post['sid'].map(id2store)
        val_df_post['item']       = val_df_post['iid'].map(id2item)
        val_df_post['date']       = np.arange(len(val_df_post)) 
        val_df_post['A']          = val_df_post['true']
        val_df_post['q']          = val_df_post['pred_q']

        val_df_post['p_eff']      = np.clip(val_df_post['pred_qp'] / (val_df_post['pred_q'] + 1e-8), 0.0, 1.0)
        val_df_post['P']          = val_df_post['q'] 

        store_weights = {id2store.get(damha_id, '담하'): HIGH_WEIGHT,
                         id2store.get(mirasia_id, '미라시아'): HIGH_WEIGHT}

        stores_all = sorted(val_df_post['restaurant'].unique())
        temp_map = fit_temperature_per_store(val_df_post, stores_all)
        def _calib_p_row(r):
            p = float(r['p_eff'])
            T = float(temp_map.get(r['restaurant'], 1.0))
            return _sigmoid(_logit(p)/T)
        val_df_post['p_eff'] = val_df_post.apply(_calib_p_row, axis=1)

        val_theta_si, theta_params_si, theta_params_s = tune_theta_eps_per_store_item(
            val_df_post, store_weights=store_weights, min_rows=60
        )

        val_aff, affine_params, sc_aff = affine_calib_per_store(val_theta_si, store_weights=store_weights)

        if caps_map:
            def _cap_row(r):
                cap = caps_map.get((r['restaurant'], r['item']), None)
                return min(r['P'], cap) if cap is not None else r['P']
            val_aff['P'] = val_aff.apply(_cap_row, axis=1)

        sc_final = competition_score(val_aff[['restaurant','item','date','A','P']], store_weights=store_weights)

        print(f"[POST] affine score    : {sc_aff:.6f}")
        print(f"[POST] final (cap)     : {sc_final:.6f}")

        post_params['theta_eps_si'] = {f"{k[0]}|||{k[1]}": v for k, v in theta_params_si.items()}
        post_params['theta_eps_s']  = theta_params_s
        post_params['p_temp']       = temp_map
        post_params['affine']       = affine_params
        post_params['scores']       = {'affine': sc_aff, 'final_cap': sc_final}
        
    model_kwargs = dict(
        input_size_num=len(features), n_stores=len(store2id), n_items=len(item2id),
        emb_store=emb_store, emb_item=emb_item, emb_dow=4, emb_mon=4, emb_season=3,
        hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, horizon=H
    )
    artifacts = dict(
        features=features,
        scaler=x_scaler,
        store2id=store2id, item2id=item2id,
        id2store=id2store, id2item=id2item,
        seq_len=SEQ_LEN, horizon=H, model_state=model.state_dict(),
        model_kwargs=model_kwargs, log_cols=log_cols,
        local_cols=local_cols,
        post_params=post_params
    )
    return artifacts


# ---------- Test Prediction + Submission in Wide Form ----------
def predict_on_test_and_fill_submission(artifacts, sub_path: str, save_path: str, test_glob: str):

    sub = pd.read_csv(sub_path)
    assert "영업일자" in sub.columns, "제출 파일에 '영업일자' 컬럼이 없습니다."
    wide_item_cols = [c for c in sub.columns if c != "영업일자"]
    assert len(wide_item_cols) > 0, "제출 파일에 예측 대상(품목) 컬럼이 없습니다."
    for col in wide_item_cols:
        sub[col] = sub[col].astype(float)
    
    feats      = artifacts["features"]
    scaler     = artifacts["scaler"]
    s2i        = artifacts["store2id"]
    i2i        = artifacts["item2id"]
    id2store   = artifacts["id2store"]
    id2item    = artifacts["id2item"]
    L          = artifacts["seq_len"]
    Hh         = artifacts["horizon"]
    log_cols   = artifacts["log_cols"]
    local_cols = artifacts["local_cols"]
    mk         = artifacts["model_kwargs"]
    pp         = artifacts.get("post_params", {})

    theta_eps_si  = pp.get('theta_eps_si', {}) 
    theta_eps_s   = pp.get('theta_eps_s', {})
    affine        = pp.get('affine', {})
    caps_map      = pp.get('caps', {})
    p_temp_map    = pp.get('p_temp', {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HurdleLSTMFC(**mk).to(device)
    model.load_state_dict(artifacts["model_state"])
    model.eval()

    label_to_idx = {str(v): i for i, v in enumerate(sub["영업일자"].astype(str).values)}
    item_col_set = set(wide_item_cols)

    test_paths = sorted(glob.glob(test_glob))
    print(f"Test files: {len(test_paths)}개")

    for path in test_paths:
        base = os.path.splitext(os.path.basename(path))[0]  
        dfp = pd.read_csv(path)

        if "영업장명_메뉴명" in dfp.columns and (("업장명" not in dfp.columns) or ("메뉴명" not in dfp.columns)):
            parts = dfp["영업장명_메뉴명"].astype(str).str.split("_", n=1, expand=True)
            dfp["업장명"] = parts[0]
            dfp["메뉴명"] = parts[1]

        feat = build_features(dfp)
        
        for c in feats:
            if c not in feat.columns:
                feat[c] = 0.0
        feat[feats] = feat[feats].replace([np.inf, -np.inf], 0).fillna(0)

        for (store, item), g in feat.groupby(["업장명","메뉴명"]):
            g = g.sort_values("영업일자")
            if len(g) < L:
                continue

            g = g.copy()
            g[log_cols] = np.log1p(g[log_cols])

            to_scale = [c for c in feats if c != '매출수량']
            x_block = g[feats].tail(L).copy()
            x_block[to_scale] = scaler.transform(x_block[to_scale])

            x_scaled = x_block.values
            x_scaled = local_scale_window(x_scaled.reshape(1, L, len(feats)), feats, local_cols).reshape(L, len(feats))

            dow = g["요일"].tail(L).astype(int).values
            mon = (g["월"].tail(L).astype(int)-1).values
            sea = g["season_id"].tail(L).astype(int).values

            last_date = pd.to_datetime(g["영업일자"].iloc[-1])
            fut_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=Hh, freq="D")
            fdw = fut_dates.weekday.values.astype(int)
            fmn = (fut_dates.month.values.astype(int)-1)
            fse = np.array([month_to_season(m) for m in fut_dates.month.values], dtype=int)

            Xn    = torch.from_numpy(x_scaled).unsqueeze(0).float().to(device)
            dw    = torch.from_numpy(dow).unsqueeze(0).long().to(device)
            mn    = torch.from_numpy(mon).unsqueeze(0).long().to(device)
            se_t  = torch.from_numpy(sea).unsqueeze(0).long().to(device)
            fdw_t = torch.from_numpy(fdw).unsqueeze(0).long().to(device)
            fmn_t = torch.from_numpy(fmn).unsqueeze(0).long().to(device)
            fse_t = torch.from_numpy(fse).unsqueeze(0).long().to(device)
            sid   = torch.tensor([s2i.get(store, s2i.get("<UNK_STORE>", 0))]).long().to(device)
            iid   = torch.tensor([i2i.get(item,  i2i.get("<UNK_ITEM>", 0))]).long().to(device)

            with torch.no_grad():
                p_buy, qlog = model(Xn,dw,mn,se_t,sid,iid,fdw_t,fmn_t,fse_t)
                q = np.expm1(qlog.detach().cpu().numpy())[0]; q[q<0]=0
                p = p_buy.detach().cpu().numpy()[0]

                if PRED_MODE == "q_only":
                    yhat = q
                elif PRED_MODE == "qp":
                    yhat = np.clip(q * p, 0, None)
                else: 
                    yhat = np.where(q >= Q_THRESH, q, np.clip(q * p, 0, None))

                if APPLY_POSTPROC:
                    p_eff = np.clip(p, 0.0, 1.0)
                    T = float(p_temp_map.get(store, 1.0))
                    if T != 1.0:
                        p_eff = 1.0 / (1.0 + np.exp(-_logit(p_eff) / T))

                    q_eff = q.copy()

                    key_si = f"{store}|||{item}"
                    te = theta_eps_si.get(key_si, None)
                    if te is None:
                        te = theta_eps_s.get(store, None)

                    if te is not None:
                        theta, eps = te['theta'], te['eps']
                        yhat = (p_eff > float(theta)).astype(float) * q_eff
                        yhat = np.maximum(float(eps), yhat)
                    else:
                        yhat = q_eff

                    af = affine.get(store, None)
                    if af is not None:
                        a, b = af['alpha'], af['beta']
                        yhat = np.maximum(0.0, a*yhat + b)

                    cap_val = caps_map.get((store, item), None)
                    if cap_val is not None:
                        yhat = np.minimum(yhat, cap_val)

            col_name = f"{store}_{item}"
            if col_name not in item_col_set:
                continue

            for k, y in enumerate(yhat, start=1):
                label = f"{base}+{k}일"
                idx = label_to_idx.get(label, None)
                if idx is not None:
                    sub.at[idx, col_name] = float(y)

    for c in wide_item_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0).clip(lower=0)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    sub.to_csv(save_path, index=False)
    print("Saved:", save_path)
    
    
# ---------- Setting ----------
TRAIN_PATH = './data/pivot_holiday_train.csv'
TEST_GLOB = './data/test/TEST_*.csv'
SAMPLE_SUB_PATH = './data/sample_submission.csv'
SAVE_PATH = './prediction/prediction.csv'

SEQ_LEN = 28 
H       = 7
HIDDEN  = 64
LAYERS  = 2
DROPOUT = 0.2
BATCH   = 512
EPOCHS  = 30
LR      = 1e-3
WD      = 1e-4
HW      = 1.5

FOCAL_GAMMA = 1.2
FOCAL_ALPHA = 0.75
LAMBDA_ZERO = 0.3
LAMBDA_QTY  = 1.0

CAP_Q = 0.95        
APPLY_POSTPROC = True

WEIGHT_PRELAUNCH = 0.0  
WEIGHT_PAUSED    = 0.5 

SEED = 42

TRAIN_FULL   = True  
VAL_DAYS     = 56
PATIENCE     = 8
SNAPSHOT_EPOCH = 23   

EARLYSTOP_ON = "q_only"     

PRED_MODE    = "q_only"   
Q_THRESH     = 0.3         

# ---------- Main ----------
if __name__ == "__main__":
    seed_everything(SEED)
    train_path = ensure_long_train(TRAIN_PATH)
    artifacts = train_full_and_pack(train_path)
    predict_on_test_and_fill_submission(artifacts, SAMPLE_SUB_PATH, SAVE_PATH, TEST_GLOB)
