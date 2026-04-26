#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Item-CF推荐模型训练（内存优化版）
基于飞猪旅行数据集
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import os
import time

print("=" * 60)
print("飞猪旅行推荐系统 - Item-CF模型训练（内存优化版）")
print("=" * 60)

start_time = time.time()

# 数据路径
DATA_DIR = "/home/asd/论文资料/4/旅游推荐数据集"
BEHAVIOR_FILE = f"{DATA_DIR}/训练代码/data/user_item_behavior_history.csv"
MODEL_FILE = f"{DATA_DIR}/model_item_cf.pkl"

# 1. 加载数据
print("\n[1/3] 加载数据...")
start = time.time()

# 用户行为数据
behavior = pd.read_csv(BEHAVIOR_FILE, header=None,
                       names=['user_id', 'item_id', 'action', 'timestamp'])
print(f"  用户行为数据: {len(behavior):,} 条")

# 2. 数据预处理（限制规模）
print("\n[2/3] 数据预处理...")
start = time.time()

# 限制数据规模
MAX_USERS = 20000
MAX_ITEMS = 15000

all_users = behavior['user_id'].unique()[:MAX_USERS]
all_items = behavior['item_id'].unique()[:MAX_ITEMS]

print(f"  用户数: {len(all_users)}")
print(f"  景点数: {len(all_items)}")

# 创建ID映射
user2idx = {u: i for i, u in enumerate(all_users)}
item2idx = {it: i for i, it in enumerate(all_items)}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"  映射后用户数: {n_users}, 景点数: {n_items}")

# 3. 构建交互矩阵（使用稀疏矩阵）
print("\n[3/3] 构建交互矩阵...")
start = time.time()

from scipy.sparse import csr_matrix

# 只保留在映射范围内的数据
behavior_filtered = behavior[
    behavior['user_id'].isin(all_users) &
    behavior['item_id'].isin(all_items)
]

print(f"  过滤后行为数据: {len(behavior_filtered):,} 条")

# 构建稀疏交互矩阵
row_indices = []
col_indices = []
data = []

for _, row in behavior_filtered.iterrows():
    u_idx = user2idx[row['user_id']]
    i_idx = item2idx[row['item_id']]
    row_indices.append(u_idx)
    col_indices.append(i_idx)
    data.append(1.0)

user_item_csr = csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_items))

稀疏度 = 1.0 - (user_item_csr.nnz / (n_users * n_items))
print(f"  交互矩阵形状: {user_item_csr.shape}")
print(f"  非零元素: {user_item_csr.nnz:,}")
print(f"  稀疏度: {稀疏度:.4f}")
print(f"  构建耗时: {time.time() - start:.1f}秒")

# 4. 计算物品相似度
print("\n[4/4] 计算物品相似度...")
start = time.time()

# 使用矩阵乘法计算物品相似度
item_sim = user_item_csr.T.dot(user_item_csr)

# 归一化
norms = np.sqrt(item_sim.diagonal())
item_sim = item_sim / norms[:, np.newaxis]

# 只保留Top-K相似物品
K = 30

# 获取每个物品的Top-K相似物品
item_sim_topk = np.zeros_like(item_sim, dtype=np.float32)
for i in range(n_items):
    # 获取相似度最高的K个物品（不包括自己）
    sim_scores = item_sim[i].copy()
    sim_scores[i] = -1  # 排除自己
    top_k_indices = np.argpartition(sim_scores, -K)[-K:]
    top_k_indices = top_k_indices[np.argsort(-sim_scores[top_k_indices])]
    item_sim_topk[i, top_k_indices] = sim_scores[top_k_indices]

print(f"  物品相似度计算完成")
print(f"  耗时: {time.time() - start:.1f}秒")

# 5. 保存模型
print("\n[5/5] 保存模型...")
start = time.time()

model = {
    "user_item_matrix": user_item_csr,
    "item_sim": item_sim_topk,
    "user2idx": user2idx,
    "item2idx": item2idx,
    "idx2user": {i: u for u, i in user2idx.items()},
    "idx2item": {i: it for it, i in item2idx.items()},
    "n_users": n_users,
    "n_items": n_items,
    "K": K,
    "metrics": {
        "n_users": n_users,
        "n_items": n_items,
        "sparsity": float(稀疏度),
        "interactions": int(user_item_csr.nnz)
    }
}

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"  模型已保存: {MODEL_FILE}")
print(f"  文件大小: {os.path.getsize(MODEL_FILE) / 1024:.1f} KB")
print(f"  保存耗时: {time.time() - start:.1f}秒")

# 6. 评估
print("\n[6/6] 模型评估...")
start = time.time()

# 在测试集上评估（使用最后10%的行为作为测试集）
behavior_sorted = behavior_filtered.sort_values('timestamp')
split_idx = int(len(behavior_sorted) * 0.9)
train_data = behavior_sorted.iloc[:split_idx]
test_data = behavior_sorted.iloc[split_idx:]

print(f"  训练集: {len(train_data):,} 条")
print(f"  测试集: {len(test_data):,} 条")

# 计算Hit Rate@10
hits = 0
total = 0
test_users = test_data['user_id'].unique()

for user_id in test_users[:500]:  # 限制评估用户数
    if user_id not in user2idx:
        continue

    # 获取该用户历史交互的物品
    user_idx = user2idx[user_id]
    history_items = user_item_csr[user_idx].nonzero()[1]

    if len(history_items) == 0:
        continue

    # 获取推荐物品（基于相似度）
    rec_items = []
    for item_idx in history_items:
        # 获取最相似的K个物品
        similar_items = np.where(item_sim_topk[item_idx] > 0)[0]
        rec_items.extend(similar_items)

    # 去重并排序
    rec_items = list(set(rec_items))
    rec_items = [idx2item[i] for i in rec_items if i in idx2item]

    # 获取真实交互的物品
    gt_items = set(test_data[test_data['user_id'] == user_id]['item_id'])

    # 计算Hit Rate@10
    if len(rec_items) >= 10:
        rec_set = set(rec_items[:10])
        if len(gt_items & rec_set) > 0:
            hits += 1
    total += 1

hr_at_10 = hits / total if total > 0 else 0
print(f"  Hit Rate@10: {hr_at_10:.4f} ({hits}/{total})")
print(f"  评估耗时: {time.time() - start:.1f}秒")

# 总结
elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"训练完成!")
print(f"总耗时: {elapsed:.1f}秒")
print(f"模型文件: {MODEL_FILE}")
print(f"{'=' * 60}")
