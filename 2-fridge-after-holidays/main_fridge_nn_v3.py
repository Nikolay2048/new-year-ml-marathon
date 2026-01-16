import os
import math
import random
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def split_pipe(s: Any) -> List[str]:
    if s is None:
        return []
    if isinstance(s, float) and np.isnan(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [t.strip() for t in s.split("|") if t.strip()]


def safe_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


MEAL_VOCAB = {"breakfast": 0, "lunch": 1, "dinner": 2, "late_snack": 3}


def ndcg5_hit_rank(target_idx: int, topk_idx: List[int]) -> float:
    for r, idx in enumerate(topk_idx, start=1):
        if idx == target_idx:
            return 1.0 / math.log2(r + 1)
    return 0.0


def hash_tokens(tokens: List[str], dim: int, seed: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h = (hash((t, seed)) % dim)
        sign = -1.0 if (hash((t, seed, "s")) & 1) else 1.0
        v[h] += sign
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


# -------------------------
# Load dishes/users & build base tables
# -------------------------
def load_dishes(dishes_csv: str):
    dishes = pd.read_csv(dishes_csv)
    dishes["category"] = dishes["category"].fillna("unknown").astype(str)

    dish_ids = dishes["dish_id"].astype(int).values
    did_to_idx = {did: i for i, did in enumerate(dish_ids)}

    cats = sorted(dishes["category"].unique().tolist())
    cat_to_idx = {c: i for i, c in enumerate(cats)}

    # calories scaling
    cal = dishes["calories"].fillna(0.0).astype(float).values
    cal_mean = float(np.mean(cal))
    cal_std = float(np.std(cal) + 1e-6)

    # store per dish:
    # base vec: [cal_z, spicy] + cat_onehot + tags_hash(tag_dim) + allerg_hash(tag_dim)
    def build_dish_base_table(tag_dim: int) -> np.ndarray:
        cat_dim = len(cat_to_idx)
        out = np.zeros((len(dishes), 2 + cat_dim + 2 * tag_dim), dtype=np.float32)
        for i, row in dishes.iterrows():
            cat = str(row["category"])
            cat_i = cat_to_idx.get(cat, 0)
            calories = safe_float(row.get("calories", 0.0))
            spicy = safe_int(row.get("spicy", 0))

            tags = split_pipe(row.get("tags", ""))
            allerg = split_pipe(row.get("allergen_tags", ""))

            out[i, 0] = (calories - cal_mean) / cal_std
            out[i, 1] = float(spicy)
            out[i, 2 + cat_i] = 1.0

            base = 2 + cat_dim
            out[i, base:base + tag_dim] = hash_tokens(tags, tag_dim, seed=404)
            out[i, base + tag_dim:base + 2 * tag_dim] = hash_tokens(allerg, tag_dim, seed=505)
        return out

    return dishes, did_to_idx, cat_to_idx, build_dish_base_table


def load_users(users_csv: str):
    users = pd.read_csv(users_csv)
    user_ids = users["user_id"].astype(int).values
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # base vec: [4 flags] + liked_hash(tag_dim) + disliked_hash(tag_dim) + allergy_hash(tag_dim)
    def build_user_base_table(tag_dim: int) -> np.ndarray:
        out = np.zeros((len(users), 4 + 3 * tag_dim), dtype=np.float32)
        for i, row in users.iterrows():
            prefers_light = safe_int(row.get("prefers_light_food", 0))
            sweet = safe_int(row.get("sweet_tooth", 0))
            coffee = safe_int(row.get("coffee_addict", 0))
            micro = safe_int(row.get("microwave_trust", 0))

            liked = split_pipe(row.get("liked_tags", ""))
            disliked = split_pipe(row.get("disliked_tags", ""))
            allergies = split_pipe(row.get("allergies", ""))

            out[i, 0:4] = np.array([prefers_light, sweet, coffee, micro], dtype=np.float32)
            out[i, 4:4 + tag_dim] = hash_tokens(liked, tag_dim, seed=101)
            out[i, 4 + tag_dim:4 + 2 * tag_dim] = hash_tokens(disliked, tag_dim, seed=202)
            out[i, 4 + 2 * tag_dim:4 + 3 * tag_dim] = hash_tokens(allergies, tag_dim, seed=303)
        return out

    return users, uid_to_idx, build_user_base_table


# -------------------------
# Train-derived statistics (history + popularity)
# -------------------------
def build_train_stats(
    train_csv: str,
    dishes_df: pd.DataFrame,
    did_to_idx: Dict[int, int],
    uid_to_idx: Dict[int, int],
    cat_to_idx: Dict[str, int],
    tag_dim: int,
    chunksize: int = 50000,
):
    """
    Build:
    - dish popularity overall and by meal_slot
    - user history: category distribution + tag-hash from chosen dishes
    """
    n_users = len(uid_to_idx)
    n_dishes = len(did_to_idx)
    cat_dim = len(cat_to_idx)

    dish_cnt = np.zeros(n_dishes, dtype=np.float32)
    dish_cnt_slot = np.zeros((4, n_dishes), dtype=np.float32)

    user_cat = np.zeros((n_users, cat_dim), dtype=np.float32)
    user_tag = np.zeros((n_users, tag_dim), dtype=np.float32)

    # Precompute per-dish category index and tag hash (from dishes_df) for speed
    dish_cat_idx = np.zeros(n_dishes, dtype=np.int32)
    dish_tag_hash = np.zeros((n_dishes, tag_dim), dtype=np.float32)

    for row in dishes_df.itertuples(index=False):
        did = int(getattr(row, "dish_id"))
        di = did_to_idx[did]
        cat = str(getattr(row, "category"))
        dish_cat_idx[di] = int(cat_to_idx.get(cat, 0))
        tags = split_pipe(getattr(row, "tags", ""))
        dish_tag_hash[di] = hash_tokens(tags, tag_dim, seed=777)

    reader = pd.read_csv(train_csv, chunksize=chunksize)
    for chunk in reader:
        # needed columns
        if "user_id" not in chunk.columns or "target_dish_id" not in chunk.columns or "meal_slot" not in chunk.columns:
            raise ValueError("train.csv must contain user_id, meal_slot, target_dish_id")

        uids = chunk["user_id"].astype(int).values
        t_dids = chunk["target_dish_id"].astype(int).values
        slots = chunk["meal_slot"].astype(str).values

        for uid, did, sl in zip(uids, t_dids, slots):
            uidx = uid_to_idx.get(int(uid), -1)
            di = did_to_idx.get(int(did), -1)
            if uidx < 0 or di < 0:
                continue

            s = MEAL_VOCAB.get(str(sl), 0)

            dish_cnt[di] += 1.0
            dish_cnt_slot[s, di] += 1.0

            cidx = dish_cat_idx[di]
            user_cat[uidx, cidx] += 1.0

            user_tag[uidx] += dish_tag_hash[di]

    # Normalize user history
    user_cat_sum = user_cat.sum(axis=1, keepdims=True) + 1e-6
    user_cat = user_cat / user_cat_sum

    # L2 norm for tag history
    norms = np.linalg.norm(user_tag, axis=1, keepdims=True) + 1e-6
    user_tag = user_tag / norms

    # Popularity features (log-scaled and normalized)
    dish_pop = np.log1p(dish_cnt)
    dish_pop = dish_pop / (dish_pop.max() + 1e-6)

    dish_pop_slot = np.log1p(dish_cnt_slot)
    dish_pop_slot = dish_pop_slot / (dish_pop_slot.max(axis=1, keepdims=True) + 1e-6)

    return user_cat.astype(np.float32), user_tag.astype(np.float32), dish_pop.astype(np.float32), dish_pop_slot.astype(np.float32)


# -------------------------
# Dataset (listwise)
# -------------------------
class FridgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, uid_to_idx: Dict[int, int], did_to_idx: Dict[int, int], is_train: bool):
        self.df = df.reset_index(drop=True)
        self.uid_to_idx = uid_to_idx
        self.did_to_idx = did_to_idx
        self.is_train = is_train
        self.cand_cols = [f"cand_{i:02d}" for i in range(1, 21)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]

        uid = int(r["user_id"])
        uidx = self.uid_to_idx.get(uid, -1)

        day = safe_int(r.get("day", 1))
        meal_slot = MEAL_VOCAB.get(str(r.get("meal_slot", "breakfast")), 0)
        hang = safe_int(r.get("hangover_level", 0))
        guests = safe_int(r.get("guests_count", 0))
        diet = safe_int(r.get("diet_mode", 0))
        load = safe_float(r.get("fridge_load_pct", 0.0))

        ctx = np.array([
            (day - 9.5) / 6.0,
            float(meal_slot),
            (hang - 1.5) / 1.5,
            (guests - 3.0) / 3.0,
            float(diet),
            (load - 55.0) / 30.0,
        ], dtype=np.float32)

        cands = []
        for c in self.cand_cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cands.append(-1)
            else:
                did = int(v)
                cands.append(self.did_to_idx.get(did, -1))
        cands = np.array(cands, dtype=np.int64)

        if self.is_train:
            target_did = int(r["target_dish_id"])
            target_didx = self.did_to_idx.get(target_did, -1)
            pos = int(np.where(cands == target_didx)[0][0])
            return uidx, ctx, cands, pos
        else:
            qid = int(r["query_id"])
            return qid, uidx, ctx, cands


# -------------------------
# Model
# -------------------------
class RankNetV3(nn.Module):
    def __init__(
        self,
        num_users: int,
        user_vec_dim: int,
        dish_vec_dim: int,
        ctx_dim: int,
        embed_u: int = 64,
        embed_d: int = 64,
        hidden: int = 320,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.u_emb = nn.Embedding(num_users + 1, embed_u)
        self.d_emb = nn.Embedding(4000, embed_d)

        self.user_proj = nn.Linear(user_vec_dim, 160)
        self.dish_proj = nn.Linear(dish_vec_dim, 160)

        self.meal_emb = nn.Embedding(4, 8)
        self.ctx_proj = nn.Linear(ctx_dim - 1 + 8, 80)

        in_dim = embed_u + embed_d + 160 + 160 + 80

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, user_ids, user_vec, dish_ids, dish_vec, ctx):
        B = user_ids.size(0)
        K = dish_ids.size(1)

        user_ids = torch.where(user_ids >= 0, user_ids, torch.zeros_like(user_ids))
        u_e = self.u_emb(user_ids)
        u_v = self.user_proj(user_vec)

        meal_slot = torch.clamp(ctx[:, 1].long(), 0, 3)
        meal_e = self.meal_emb(meal_slot)
        ctx_wo = torch.cat([ctx[:, :1], ctx[:, 2:]], dim=1)
        ctx_f = self.ctx_proj(torch.cat([ctx_wo, meal_e], dim=1))

        dish_ids = torch.where(dish_ids >= 0, dish_ids, torch.zeros_like(dish_ids))
        d_e = self.d_emb(dish_ids)
        d_v = self.dish_proj(dish_vec)

        u_e2 = u_e.unsqueeze(1).expand(B, K, u_e.size(-1))
        u_v2 = u_v.unsqueeze(1).expand(B, K, u_v.size(-1))
        ctx2 = ctx_f.unsqueeze(1).expand(B, K, ctx_f.size(-1))

        x = torch.cat([u_e2, u_v2, d_e, d_v, ctx2], dim=2)
        s = self.mlp(x).squeeze(-1)
        return s


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    n = 0
    for uidx, ctx, cands, pos in loader:
        uidx = uidx.to(device)
        ctx = ctx.to(device)
        cands = cands.to(device)
        pos = pos.to(device)

        user_vec = USER_TABLE_T[uidx.clamp(min=0)]
        dish_vec = DISH_TABLE_T[cands.clamp(min=0)]

        scores = model(uidx, user_vec, cands, dish_vec, ctx)
        loss = ce(scores, pos)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = uidx.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def eval_ndcg5(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for uidx, ctx, cands, pos in loader:
        uidx = uidx.to(device)
        ctx = ctx.to(device)
        cands = cands.to(device)
        pos = pos.to(device)

        user_vec = USER_TABLE_T[uidx.clamp(min=0)]
        dish_vec = DISH_TABLE_T[cands.clamp(min=0)]

        scores = model(uidx, user_vec, cands, dish_vec, ctx)
        top5 = torch.topk(scores, k=5, dim=1).indices.cpu().numpy()
        pos_np = pos.cpu().numpy()
        for i in range(len(pos_np)):
            total += ndcg5_hit_rank(int(pos_np[i]), list(top5[i]))
        n += len(pos_np)
    return total / max(1, n)


@torch.no_grad()
def predict_ensemble(models: List[nn.Module], test_df: pd.DataFrame, uid_to_idx, did_to_idx, device, sample_sub, out_csv):
    ds = FridgeDataset(test_df, uid_to_idx, did_to_idx, is_train=False)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    # invert dish index -> dish_id
    idx_to_did = np.zeros(max(did_to_idx.values()) + 1, dtype=np.int32)
    for did, di in did_to_idx.items():
        idx_to_did[di] = int(did)

    rows = []
    for qid, uidx, ctx, cands in dl:
        qid = qid.numpy().tolist()
        uidx = uidx.to(device)
        ctx = ctx.to(device)
        cands = cands.to(device)

        user_vec = USER_TABLE_T[uidx.clamp(min=0)]
        dish_vec = DISH_TABLE_T[cands.clamp(min=0)]

        # average scores
        avg_scores = None
        for m in models:
            m.eval()
            s = m(uidx, user_vec, cands, dish_vec, ctx)
            avg_scores = s if avg_scores is None else (avg_scores + s)
        avg_scores = avg_scores / float(len(models))

        top5_idx = torch.topk(avg_scores, k=5, dim=1).indices.cpu().numpy()
        cands_cpu = cands.cpu().numpy()

        for bi in range(len(qid)):
            idxs = top5_idx[bi].tolist()
            dish_indices = [int(cands_cpu[bi, j]) for j in idxs]
            dish_ids = [int(idx_to_did[di]) if 0 <= di < len(idx_to_did) else 1 for di in dish_indices]
            rows.append([qid[bi]] + dish_ids)

    sub = pd.DataFrame(rows, columns=["query_id", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"])
    sample = pd.read_csv(sample_sub)
    sub = sample[["query_id"]].merge(sub, on="query_id", how="left")
    for i, col in enumerate(["rec_1", "rec_2", "rec_3", "rec_4", "rec_5"], start=1):
        sub[col] = sub[col].fillna(i).astype(int)
    sub.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(sub.head(10))


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, default="data/train_holodilnik.csv")
    parser.add_argument("--test_csv", type=str, default="data/test_holodilnik.csv")
    parser.add_argument("--users_csv", type=str, default="data/users.csv")
    parser.add_argument("--dishes_csv", type=str, default="data/dishes.csv")
    parser.add_argument("--sample_sub", type=str, default="data/sample_submission_holodilnik.csv")
    parser.add_argument("--out_csv", type=str, default="submission.csv")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag_dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    parser.add_argument("--ensemble", type=int, default=3, help="кол-во сидов в ансамбле (2-5)")
    parser.add_argument("--chunksize_stats", type=int, default=50000)

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load
    dishes_df, did_to_idx, cat_to_idx, build_dish_base = load_dishes(args.dishes_csv)
    users_df, uid_to_idx, build_user_base = load_users(args.users_csv)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # train-derived stats
    print("Building train stats (history + popularity)...")
    user_cat_hist, user_tag_hist, dish_pop, dish_pop_slot = build_train_stats(
        train_csv=args.train_csv,
        dishes_df=dishes_df,
        did_to_idx=did_to_idx,
        uid_to_idx=uid_to_idx,
        cat_to_idx=cat_to_idx,
        tag_dim=args.tag_dim,
        chunksize=args.chunksize_stats,
    )

    # build tables (base + history/pop)
    user_base = build_user_base(args.tag_dim)  # [U, 4+3T]
    # append: cat_hist + tag_hist
    USER_TABLE = np.concatenate([user_base, user_cat_hist, user_tag_hist], axis=1).astype(np.float32)

    dish_base = build_dish_base(args.tag_dim)  # [D, 2+cat+2T]
    # append: pop_global (1) + pop_by_slot (4)
    dish_extra = np.concatenate(
        [dish_pop.reshape(-1, 1), dish_pop_slot.T], axis=1
    ).astype(np.float32)  # [D, 5]
    DISH_TABLE = np.concatenate([dish_base, dish_extra], axis=1).astype(np.float32)

    # move to device globals
    global USER_TABLE_T, DISH_TABLE_T
    USER_TABLE_T = torch.tensor(USER_TABLE, dtype=torch.float32, device=device)
    DISH_TABLE_T = torch.tensor(DISH_TABLE, dtype=torch.float32, device=device)

    # split train/val
    n = len(train_df)
    val_n = max(1, int(n * args.val_ratio))
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = train_df.iloc[:val_n].reset_index(drop=True)
    tr_df = train_df.iloc[val_n:].reset_index(drop=True)

    ds_tr = FridgeDataset(tr_df, uid_to_idx, did_to_idx, is_train=True)
    ds_va = FridgeDataset(val_df, uid_to_idx, did_to_idx, is_train=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    user_vec_dim = USER_TABLE_T.size(1)
    dish_vec_dim = DISH_TABLE_T.size(1)
    ctx_dim = 6

    models = []
    base_seed = args.seed

    for k in range(args.ensemble):
        cur_seed = base_seed + 100 * k
        seed_everything(cur_seed)

        model = RankNetV3(
            num_users=len(uid_to_idx),
            user_vec_dim=user_vec_dim,
            dish_vec_dim=dish_vec_dim,
            ctx_dim=ctx_dim,
            embed_u=64,
            embed_d=64,
            hidden=320,
            dropout=0.15,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best = -1.0
        best_state = None

        print(f"\n=== Train model {k+1}/{args.ensemble} (seed={cur_seed}) ===")
        for e in range(1, args.epochs + 1):
            loss = train_one_epoch(model, dl_tr, opt, device)
            score = eval_ndcg5(model, dl_va, device)
            print(f"  Epoch {e}/{args.epochs} | loss={loss:.4f} | val_ndcg5≈{score:.4f}")
            if score > best:
                best = score
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

        # load best
        model.load_state_dict(best_state)
        models.append(model)
        print(f"  -> best val_ndcg5≈{best:.4f}")

    # predict ensemble
    print("\nPredicting ensemble submission...")
    predict_ensemble(
        models=models,
        test_df=test_df,
        uid_to_idx=uid_to_idx,
        did_to_idx=did_to_idx,
        device=device,
        sample_sub=args.sample_sub,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
