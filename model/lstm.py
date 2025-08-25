import os

import glob, time, random
import re

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

def ensure_long_train(path: str) -> str:
    df = pd.read_csv(path)
    if "영업일자" in df.columns and ("영업장명_메뉴명" not in df.columns):
        id_col = "영업일자"
        value_cols = [c for c in df.columns if c != id_col]
        long_df = df.melt(id_vars=[id_col], value_vars=value_cols,
                          var_name="영업장명_메뉴명", value_name="매출수량")
    else:
        long_df = df.copy()

    # 연속 날짜 보장
    long_df["영업일자"] = pd.to_datetime(long_df["영업일자"])
    all_dates = pd.date_range(long_df["영업일자"].min(), long_df["영업일자"].max(), freq="D")
    keys = long_df["영업장명_메뉴명"].astype(str).unique()
    full_index = pd.MultiIndex.from_product([keys, all_dates], names=["영업장명_메뉴명","영업일자"])

    long_df = (long_df
        .astype({"영업장명_메뉴명":"string"})
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
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = (a != 0)
    if not np.any(mask):
        return np.nan
    num = 2.0 * np.abs(a[mask] - p[mask]) 
    den = np.abs(a[mask]) + np.abs(p[mask]) + eps
    return float(np.mean(num / den))


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
    df = df.sort_values(["영업장명_메뉴명", "영업일자"]).reset_index(drop=True)
    df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0).clip(lower=0)

    df["월"] = df["영업일자"].dt.month
    df["요일"] = df["영업일자"].dt.weekday
    df["day_of_year"] = df["영업일자"].dt.dayofyear
    df["season_id"] = df["월"].apply(month_to_season).astype(int)
    df["doy_sin"] = np.sin(2*np.pi*(df["day_of_year"]-1)/365.0)
    df["doy_cos"] = np.cos(2*np.pi*(df["day_of_year"]-1)/365.0)
    df["is_weekend"] = df["요일"].isin([5,6]).astype(int)
    df["is_summer_peak"] = df["월"].isin([7,8]).astype(int)
    df["is_winter_peak"] = df["월"].isin([12,1,2]).astype(int)

    start_year = int(df["영업일자"].min().year) - 1
    end_year   = int(df["영업일자"].max().year) + 1
    kr_holidays = holidays.KR(years=range(start_year, end_year + 1), observed=True)
    holi_idx = pd.DatetimeIndex(pd.to_datetime(list(kr_holidays.keys()))).tz_localize(None)
    d = df["영업일자"].dt.normalize()
    df["is_holiday"]            = d.isin(holi_idx).astype(int)
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
        vym[["업장명","메뉴명","월시작","avg_sales_prev_month"]],
        on=["업장명","메뉴명","월시작"], how="left"
    )
    df["avg_sales_prev_month"] = df["avg_sales_prev_month"].fillna(0.0)

    for c in ['is_weekend','is_holiday','is_day_before_holiday','is_day_after_holiday',
              'is_summer_peak','is_winter_peak']:
        if c in df.columns:
            df[c] = df[c].astype(int)
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
    fut_end_dates = []  

    for name, g in data.groupby("영업장명_메뉴명"):
        g = g.sort_values("영업일자")
        x   = g[feats].values
        y   = g["매출수량"].values         
        dow = g["요일"].astype(int).values
        mon = (g["월"].astype(int)-1).values
        sea = g["season_id"].astype(int).values
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

    return (np.array(Xs), np.array(ys), np.array(dows), np.array(mons), np.array(seas),
            np.array(fdw), np.array(fmn), np.array(fse),
            np.array(sids), np.array(iids),
            np.array(fut_end_dates, dtype='datetime64[ns]'))
    
# ---------- Model ----------
class HurdleLSTMFC(nn.Module):
    '''
    Hurdle구조
    그 날 그 품목을 살지 (>0)를 확률 p로 예측
    산다고 가정했을 때 수량의 log (log1p(quantity)) qy를 예측
    '''
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
  
        self.lstm = nn.LSTM(feat_in, 
                            hidden, 
                            layers, 
                            batch_first=True,
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


def focal_bce_per_sample(pred, target, gamma=1.0, alpha=0.75, eps=1e-7):
    pred = pred.clamp(eps, 1-eps) 
    bce  = -(alpha*target*torch.log(pred) + (1-alpha)*(1-target)*torch.log(1-pred)) 
    pt   = torch.where(target==1, pred, 1-pred) 
    loss = bce * ((1-pt)**gamma) 
    return loss.mean(dim=1)


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

    features = [
        '매출수량', 'sales_lag_1', 'sales_rolling_mean_7', 'sales_rolling_mean_14',
        'sales_rolling_mean_28', 'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
        'avg_sales_prev_month', 'is_weekend', 'is_holiday', 'is_day_before_holiday',
        'is_day_after_holiday', 'is_summer_peak', 'is_winter_peak', 'doy_sin', 'doy_cos'
    ]
    for c in features:
        if c not in df.columns: df[c] = 0
    df[features] = df[features].replace([np.inf,-np.inf],0).fillna(0)

    log_cols = ['매출수량','sales_lag_1','sales_rolling_mean_7','sales_rolling_mean_14',
                'sales_rolling_mean_28','avg_sales_prev_month']
    df[log_cols] = np.log1p(df[log_cols])

    to_scale = [c for c in features if c != '매출수량']
    x_scaler = MinMaxScaler()

    if not TRAIN_FULL and VAL_DAYS > 0:
        train_df = df[df['영업일자'] < val_start].copy()
        val_df = df[df['영업일자'] >= val_start].copy()

        print(f"Fitting scaler ONLY on train data (before {val_start.date()})...")
        x_scaler.fit(train_df[to_scale])

        train_df[to_scale] = x_scaler.transform(train_df[to_scale])
        val_df[to_scale] = x_scaler.transform(val_df[to_scale])

        data_for_seq = pd.concat([train_df, val_df])
    else:
        print("TRAIN_FULL is True. Fitting scaler on all data...")
        df[to_scale] = x_scaler.fit_transform(df[to_scale])
        data_for_seq = df

    stores = ["<UNK_STORE>"] + sorted(df["업장명"].unique().tolist())
    items  = ["<UNK_ITEM>"] + sorted(df["메뉴명"].unique().tolist())
    store2id = {s:i for i,s in enumerate(stores)}
    item2id  = {m:i for i,m in enumerate(items)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xall,yall,dwall,mnall,seall,fdwall,fnall,fseall,sidall,iidall,fend = create_sequences_multi(
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
                  'rolling_std_28','avg_sales_prev_month']

    def to_tensor_pack(Xn,dw,mn,se,fdw,fmn,fse,sid,iid,y,mask):
        Xn, dw, mn, se = Xn[mask], dw[mask], mn[mask], se[mask]
        fdw, fmn, fse = fdw[mask], fmn[mask], fse[mask]
        sid, iid, y = sid[mask], iid[mask], y[mask]

        Xn = local_scale_window(Xn, features, local_cols)

        return (
            torch.from_numpy(Xn).float(), torch.from_numpy(dw).long(),
            torch.from_numpy(mn).long(), torch.from_numpy(se).long(),
            torch.from_numpy(fdw).long(), torch.from_numpy(fmn).long(),
            torch.from_numpy(fse).long(), torch.from_numpy(sid).long(),
            torch.from_numpy(iid).long(), torch.from_numpy(y).float(),
        )

    Xtr_t,dwtr_t,mntr_t,setr_t,fdwtr_t,fmntr_t,fsetr_t,sidtr_t,iidtr_t,ytr_t = to_tensor_pack(
        Xall,dwall,mnall,seall,fdwall,fnall,fseall,sidall,iidall,yall, mask = train_mask
    )
    if n_val > 0:
        Xv_t,dwv_t,mnv_t,sev_t,fdwv_t,fmnv_t,fsev_t,sidv_t,iidv_t,yv_t = to_tensor_pack(
            Xall,dwall,mnall,seall,fdwall,fnall,fseall,sidall,iidall,yall, mask = val_mask
        )

    class DS(torch.utils.data.Dataset):
        def __init__(self, Xn,dw,mn,se,fdw,fmn,fse,sid,iid,y):
            self.Xn,self.dw,self.mn,self.se=Xn,dw,mn,se
            self.fdw,self.fmn,self.fse=fdw,fmn,fse
            self.sid,self.iid,self.y=sid,iid,y
        def __len__(self): return self.Xn.size(0)
        def __getitem__(self,i):
            return (self.Xn[i],self.dw[i],self.mn[i],self.se[i],
                    self.fdw[i],self.fmn[i],self.fse[i],self.sid[i],self.iid[i],self.y[i])

    train_loader = DataLoader(DS(Xtr_t,dwtr_t,mntr_t,setr_t,fdwtr_t,fmntr_t,fsetr_t,sidtr_t,iidtr_t,ytr_t),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    if n_val > 0:
        val_loader = DataLoader(DS(Xv_t,dwv_t,mnv_t,sev_t,fdwv_t,fmnv_t,fsev_t,sidv_t,iidv_t,yv_t),
                                batch_size=BATCH, shuffle=False, num_workers=0)

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
        
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for Xn,dw,mn,se,fdw,fmn,fse,sid,iid,yb in train_loader:
            Xn,dw,mn,se,fdw,fmn,fse,sid,iid,yb = \
                Xn.to(device),dw.to(device),mn.to(device),se.to(device),fdw.to(device),fmn.to(device),fse.to(device),sid.to(device),iid.to(device),yb.to(device)
                
            weights = torch.ones(Xn.size(0), device=device)
            mask_hi = ((sid == damha_id) | (sid == mirasia_id))
            weights = torch.where(mask_hi, torch.as_tensor(HIGH_WEIGHT, device=device, dtype=weights.dtype), weights)

            
            optimizer.zero_grad()
            p_buy, qty_log = model(Xn,dw,mn,se,sid,iid,fdw,fmn,fse)  

            tgt_bin = (yb > 0).float()
            loss_zero_per = focal_bce_per_sample(p_buy, tgt_bin, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)  

            mask = (yb > 0).float()
            qty_raw = huber(qty_log, yb) * mask            
            denom = mask.sum(dim=1).clamp_min(1e-6)        
            loss_qty_per = qty_raw.sum(dim=1) / denom      

            total_per = LAMBDA_ZERO * loss_zero_per + LAMBDA_QTY * loss_qty_per
            
            weights = weights / weights.mean().clamp_min(1e-6)
            loss = (total_per * weights).mean()
         
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * Xn.size(0)

        tr_loss /= max(len(train_loader.dataset), 1)

        if TRAIN_FULL or n_val == 0:
            print(f"  - epoch {epoch:02d} | train {tr_loss:.4f}")
            continue  

        # ---------- Validation ----------
        model.eval()
        va_loss = 0.0
        smapes_qp, smapes_q = [], []
        
        all_trues, all_preds_q, all_preds_qp, all_sids, all_iids = [], [], [], [], []

        with torch.no_grad():
            for Xn,dw,mn,se,fdw,fmn,fse,sid,iid,yb in val_loader:
                Xn,dw,mn,se,fdw,fmn,fse,sid,iid,yb = \
                    Xn.to(device),dw.to(device),mn.to(device),se.to(device),fdw.to(device),fmn.to(device),fse.to(device),sid.to(device),iid.to(device),yb.to(device)

                p_buy, qty_log = model(Xn,dw,mn,se,sid,iid,fdw,fmn,fse)

                tgt_bin = (yb > 0).float()
                loss_zero = focal_bce_per_sample(p_buy, tgt_bin, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA).mean()
                mask = (yb > 0).float()
                denom = mask.sum(dim=1).clamp_min(1e-6)
                loss_qty = ((huber(qty_log, yb) * mask).sum(dim=1) / denom).mean()
                loss = LAMBDA_ZERO*loss_zero + LAMBDA_QTY*loss_qty
                va_loss += loss.item() * Xn.size(0)

                q = torch.expm1(qty_log).clamp(min=0)
                p = p_buy
                yhat_qp = (q * p).cpu().numpy().ravel()
                yhat_q  = q.cpu().numpy().ravel()
                ytrue   = torch.expm1(yb).cpu().numpy().ravel()
        
                repeated_sids = sid.cpu().numpy().repeat(H)
                repeated_iids = iid.cpu().numpy().repeat(H)
        
                all_trues.append(ytrue.ravel())
                all_preds_q.append(yhat_q.ravel())
                all_preds_qp.append(yhat_qp.ravel())
                all_sids.append(repeated_sids)
                all_iids.append(repeated_iids)

        va_loss /= max(len(val_loader.dataset), 1)
 
        all_trues = np.concatenate(all_trues)
        all_preds_q = np.concatenate(all_preds_q)
        all_preds_qp = np.concatenate(all_preds_qp)
        all_sids = np.concatenate(all_sids)
        all_iids = np.concatenate(all_iids)

        val_results = pd.DataFrame({
            'sid': all_sids,
            'iid': all_iids,
            'true': all_trues,
            'pred_q': all_preds_q,
            'pred_qp': all_preds_qp
        })

        smape_by_item_q = val_results.groupby(['sid', 'iid']).apply(
            lambda x: smape_ignore_zero(x['true'], x['pred_q']),
            include_groups=False
        )
        smape_by_item_qp = val_results.groupby(['sid', 'iid']).apply(
            lambda x: smape_ignore_zero(x['true'], x['pred_qp']),
            include_groups=False
        )
  
        smape_by_store_q = smape_by_item_q.groupby(level=0).mean()
        smape_by_store_qp = smape_by_item_qp.groupby(level=0).mean()

        weights_by_store = smape_by_store_q.index.map(
            lambda sid: HIGH_WEIGHT if sid in [damha_id, mirasia_id] else 1.0
        )

        va_smape_q_weighted = np.average(smape_by_store_q.fillna(0), weights=weights_by_store)
        va_smape_qp_weighted = np.average(smape_by_store_qp.fillna(0), weights=weights_by_store)

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
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {epoch} (best sMAPE={best_s:.4f})")
                break

    if (not TRAIN_FULL) and (best_state is not None):
        model.load_state_dict(best_state)

    print(f"Training done in {time.time()-t0:.1f}s")

    model_kwargs = dict(
        input_size_num=len(features), n_stores=len(store2id), n_items=len(item2id),
        emb_store=emb_store, emb_item=emb_item, emb_dow=4, emb_mon=4, emb_season=3,
        hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, horizon=H
    )
    artifacts = dict(
        features=features,
        scaler=x_scaler,              
        store2id=store2id, item2id=item2id,
        seq_len=SEQ_LEN, horizon=H, model_state=model.state_dict(),
        model_kwargs=model_kwargs, log_cols=log_cols,
        local_cols=local_cols
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
    L          = artifacts["seq_len"]
    Hh         = artifacts["horizon"]
    log_cols   = artifacts["log_cols"]
    local_cols = artifacts["local_cols"]
    mk         = artifacts["model_kwargs"]

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
SAVE_PATH = './prediction/prediction1.csv'

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

SEED = 42

# ---------- 학습/검증 토글 ----------

TRAIN_FULL   = True
VAL_DAYS     = 56
PATIENCE     = 8 

EARLYSTOP_ON = "q_only"     

PRED_MODE    = "q_only"   
Q_THRESH     = 0.3          

# ---------- Main ----------
seed_everything(SEED)
train_path = ensure_long_train(TRAIN_PATH)
artifacts = train_full_and_pack(train_path)
predict_on_test_and_fill_submission(artifacts, SAMPLE_SUB_PATH, SAVE_PATH, TEST_GLOB)