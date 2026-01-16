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


def ndcg5_hit_rank(target_idx: int, topk_idx: List[int]) -> float:
    # target_idx in [0..19] position of true item among candidates
    # topk_idx list of indices ranked best->worst
    for r, idx in enumerate(topk_idx, start=1):
        if idx == target_idx:
            return 1.0 / math.log2(r + 1)
    return 0.0


# -------------------------
# Precompute user/dish vectors (hashed tags)
# -------------------------
def hash_tokens(tokens: List[str], dim: int, seed: int = 13) -> np.ndarray:
    """Simple signed hashing into fixed dim (like a tiny feature hasher)."""
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h = (hash((t, seed)) % dim)
        # alternate sign by another hash bit
        sign = -1.0 if (hash((t, seed, "s")) & 1) else 1.0
        v[h] += sign
    # L2 normalize (optional)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def build_user_table(users_csv: str, tag_dim: int) -> Tuple[np.ndarray, Dict[int, int]]:
    users = pd.read_csv(users_csv)
    user_ids = users["user_id"].astype(int).values
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # vector: [4 flags] + liked_hash(tag_dim) + disliked_hash(tag_dim) + allergy_hash(tag_dim)
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

    return out, uid_to_idx


def build_dish_table(dishes_csv: str, tag_dim: int) -> Tuple[np.ndarray, Dict[int, int], Dict[str, int]]:
    dishes = pd.read_csv(dishes_csv)
    dish_ids = dishes["dish_id"].astype(int).values
    did_to_idx = {did: i for i, did in enumerate(dish_ids)}

    # category vocab
    cats = dishes["category"].fillna("unknown").astype(str).unique().tolist()
    cat_to_idx = {c: i for i, c in enumerate(sorted(cats))}

    # vector: [calories_scaled, spicy] + category_onehot(len(cat_to_idx)) + tags_hash(tag_dim) + allergen_hash(tag_dim)
    cat_dim = len(cat_to_idx)
    out = np.zeros((len(dishes), 2 + cat_dim + 2 * tag_dim), dtype=np.float32)

    # scale calories roughly
    cal = dishes["calories"].fillna(0.0).astype(float).values
    cal_mean = float(np.mean(cal))
    cal_std = float(np.std(cal) + 1e-6)

    for i, row in dishes.iterrows():
        category = str(row.get("category", "unknown"))
        cat_i = cat_to_idx.get(category, 0)
        calories = safe_float(row.get("calories", 0.0))
        spicy = safe_int(row.get("spicy", 0))

        tags = split_pipe(row.get("tags", ""))
        allerg = split_pipe(row.get("allergen_tags", ""))

        # numeric
        out[i, 0] = (calories - cal_mean) / cal_std
        out[i, 1] = float(spicy)

        # category onehot
        out[i, 2 + cat_i] = 1.0

        base = 2 + cat_dim
        out[i, base:base + tag_dim] = hash_tokens(tags, tag_dim, seed=404)
        out[i, base + tag_dim:base + 2 * tag_dim] = hash_tokens(allerg, tag_dim, seed=505)

    return out, did_to_idx, cat_to_idx


# -------------------------
# Dataset (listwise: 20 candidates per row)
# -------------------------
MEAL_VOCAB = {"breakfast": 0, "lunch": 1, "dinner": 2, "late_snack": 3}


class FridgeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        uid_to_idx: Dict[int, int],
        did_to_idx: Dict[int, int],
        is_train: bool,
    ):
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

        # context numeric (normalize lightly)
        ctx = np.array([
            (day - 9.5) / 6.0,          # roughly center 1..18
            float(meal_slot),           # categorical id (embed later or onehot-like)
            (hang - 1.5) / 1.5,
            (guests - 3.0) / 3.0,
            float(diet),
            (load - 55.0) / 30.0,
        ], dtype=np.float32)

        # candidates -> dish indices
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
            # target position among 20 cands
            # по условию target в cand_*
            pos = int(np.where(cands == target_didx)[0][0])
            return uidx, ctx, cands, pos
        else:
            qid = int(r["query_id"])
            return qid, uidx, ctx, cands


# -------------------------
# Model
# -------------------------
class RankNet(nn.Module):
    def __init__(
        self,
        num_users: int,
        user_vec_dim: int,
        dish_vec_dim: int,
        ctx_dim: int,
        embed_dim_u: int = 64,
        embed_dim_d: int = 64,
        hidden: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.u_emb = nn.Embedding(num_users + 1, embed_dim_u)  # +1 for unknown
        self.d_emb = nn.Embedding(2000, embed_dim_d)  # we will map dish idx into [0..n_dishes-1], but keep large safe

        self.user_proj = nn.Linear(user_vec_dim, 128)
        self.dish_proj = nn.Linear(dish_vec_dim, 128)

        self.meal_emb = nn.Embedding(4, 8)  # meal_slot id in ctx[1]
        self.ctx_proj = nn.Linear(ctx_dim - 1 + 8, 64)  # ctx without meal_slot + meal_emb

        in_dim = embed_dim_u + embed_dim_d + 128 + 128 + 64

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        user_ids: torch.LongTensor,      # [B]
        user_vec: torch.FloatTensor,     # [B, Uv]
        dish_ids: torch.LongTensor,      # [B, 20]
        dish_vec: torch.FloatTensor,     # [B, 20, Dv]
        ctx: torch.FloatTensor,          # [B, C]
    ) -> torch.FloatTensor:
        """
        returns scores [B, 20]
        """
        B = user_ids.size(0)
        K = dish_ids.size(1)

        # unknown handling
        user_ids = torch.clamp(user_ids, min=0)
        user_ids = torch.where(user_ids >= 0, user_ids, torch.zeros_like(user_ids))
        u_e = self.u_emb(user_ids)  # [B, eu]
        u_v = self.user_proj(user_vec)  # [B, 128]

        # ctx: ctx[1] is meal_slot id (float), others numeric
        meal_slot = torch.clamp(ctx[:, 1].long(), 0, 3)
        meal_e = self.meal_emb(meal_slot)  # [B, 8]
        ctx_wo = torch.cat([ctx[:, :1], ctx[:, 2:]], dim=1)  # remove ctx[:,1]
        ctx_f = self.ctx_proj(torch.cat([ctx_wo, meal_e], dim=1))  # [B, 64]

        # dishes
        dish_ids = torch.clamp(dish_ids, min=0)
        dish_ids = torch.where(dish_ids >= 0, dish_ids, torch.zeros_like(dish_ids))
        d_e = self.d_emb(dish_ids)  # [B, 20, ed]
        d_v = self.dish_proj(dish_vec)  # [B, 20, 128]

        # expand user/context to [B,20,*]
        u_e2 = u_e.unsqueeze(1).expand(B, K, u_e.size(-1))
        u_v2 = u_v.unsqueeze(1).expand(B, K, u_v.size(-1))
        ctx2 = ctx_f.unsqueeze(1).expand(B, K, ctx_f.size(-1))

        x = torch.cat([u_e2, u_v2, d_e, d_v, ctx2], dim=2)  # [B, 20, in_dim]
        s = self.mlp(x).squeeze(-1)  # [B, 20]
        return s


# -------------------------
# Training / Inference
# -------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()

    for uidx, ctx, cands, pos in loader:
        uidx = uidx.to(device)
        ctx = ctx.to(device)
        cands = cands.to(device)
        pos = pos.to(device)

        # lookup dense vectors from global tables stored on device in closure
        user_vec = USER_TABLE_T[uidx.clamp(min=0)]  # [B, Uv]
        dish_vec = DISH_TABLE_T[cands.clamp(min=0)]  # [B, 20, Dv]

        scores = model(uidx, user_vec, cands, dish_vec, ctx)  # [B, 20]
        loss = ce(scores, pos)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = uidx.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(1, n)


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

        scores = model(uidx, user_vec, cands, dish_vec, ctx)  # [B, 20]
        top5 = torch.topk(scores, k=5, dim=1).indices.cpu().numpy()  # indices in 0..19
        pos_np = pos.cpu().numpy()

        for i in range(len(pos_np)):
            total += ndcg5_hit_rank(int(pos_np[i]), list(top5[i]))
        n += len(pos_np)

    return total / max(1, n)


@torch.no_grad()
def predict_submission(model, test_df, uid_to_idx, did_to_idx, device, out_csv, sample_sub_csv):
    ds = FridgeDataset(test_df, uid_to_idx, did_to_idx, is_train=False)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    rows = []
    for qid, uidx, ctx, cands in dl:
        qid = qid.numpy().tolist()
        uidx = uidx.to(device)
        ctx = ctx.to(device)
        cands = cands.to(device)

        user_vec = USER_TABLE_T[uidx.clamp(min=0)]
        dish_vec = DISH_TABLE_T[cands.clamp(min=0)]

        scores = model(uidx, user_vec, cands, dish_vec, ctx)  # [B,20]
        top5_idx = torch.topk(scores, k=5, dim=1).indices  # [B,5] (indices 0..19)
        top5_idx = top5_idx.cpu().numpy()

        cands_cpu = cands.cpu().numpy()  # [B,20] dish_index
        for bi in range(len(qid)):
            idxs = top5_idx[bi].tolist()
            dish_indices = [int(cands_cpu[bi, j]) for j in idxs]
            # map dish_index -> dish_id
            # build reverse map once
            rows.append([qid[bi]] + dish_indices)

    sub = pd.DataFrame(rows, columns=["query_id", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"])

    # convert dish_index -> dish_id
    # did_to_idx maps dish_id->dish_index, so invert:
    idx_to_did = np.zeros(max(did_to_idx.values()) + 1, dtype=np.int32)
    for did, di in did_to_idx.items():
        idx_to_did[di] = int(did)

    for col in ["rec_1", "rec_2", "rec_3", "rec_4", "rec_5"]:
        sub[col] = sub[col].clip(lower=0)
        sub[col] = sub[col].apply(lambda di: int(idx_to_did[int(di)]) if int(di) < len(idx_to_did) else 1)

    sample = pd.read_csv(sample_sub_csv)
    sub = sample[["query_id"]].merge(sub, on="query_id", how="left")
    # если где-то NaN — забьем 1..5 (редко)
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

    # paths defaults (files in data/)
    parser.add_argument("--train_csv", type=str, default="data/train_holodilnik.csv")
    parser.add_argument("--test_csv", type=str, default="data/test_holodilnik.csv")
    parser.add_argument("--users_csv", type=str, default="data/users.csv")
    parser.add_argument("--dishes_csv", type=str, default="data/dishes.csv")
    parser.add_argument("--sample_sub", type=str, default="data/sample_submission_holodilnik.csv")
    parser.add_argument("--out_csv", type=str, default="submission.csv")

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tag_dim", type=int, default=384)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # build tables
    user_table, uid_to_idx = build_user_table(args.users_csv, tag_dim=args.tag_dim)
    dish_table, did_to_idx, _ = build_dish_table(args.dishes_csv, tag_dim=args.tag_dim)

    # move tables to torch tensors on device (globals for speed)
    global USER_TABLE_T, DISH_TABLE_T
    USER_TABLE_T = torch.tensor(user_table, dtype=torch.float32, device=device)
    DISH_TABLE_T = torch.tensor(dish_table, dtype=torch.float32, device=device)

    # remap dish embedding size safely
    # (we used d_emb with size 2000; dish indices are < len(dish_table) so ok)

    # split train/val
    n = len(train_df)
    val_n = int(n * args.val_ratio)
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = train_df.iloc[:val_n].reset_index(drop=True)
    tr_df = train_df.iloc[val_n:].reset_index(drop=True)

    ds_tr = FridgeDataset(tr_df, uid_to_idx, did_to_idx, is_train=True)
    ds_va = FridgeDataset(val_df, uid_to_idx, did_to_idx, is_train=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # model
    user_vec_dim = USER_TABLE_T.size(1)
    dish_vec_dim = DISH_TABLE_T.size(1)
    ctx_dim = 6

    model = RankNet(
        num_users=len(uid_to_idx),
        user_vec_dim=user_vec_dim,
        dish_vec_dim=dish_vec_dim,
        ctx_dim=ctx_dim,
        embed_dim_u=64,
        embed_dim_d=64,
        hidden=256,
        dropout=0.15,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    best_path = "best_ranknet.pt"

    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dl_tr, opt, device)
        score = eval_ndcg5(model, dl_va, device)
        print(f"Epoch {e}/{args.epochs} | loss={loss:.4f} | val_ndcg5≈{score:.4f}")

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  -> saved best: {best_path}")

    # load best and train on full data briefly (optional light finetune)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # predict submission
    predict_submission(
        model=model,
        test_df=test_df,
        uid_to_idx=uid_to_idx,
        did_to_idx=did_to_idx,
        device=device,
        out_csv=args.out_csv,
        sample_sub_csv=args.sample_sub,
    )


if __name__ == "__main__":
    main()
