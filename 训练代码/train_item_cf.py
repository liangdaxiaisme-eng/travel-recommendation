#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞猪旅行推荐系统 - 超轻量版
基于物品的协同过滤
"""

import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
import pickle
import time

print("=" * 60)
print("飞猪旅行推荐系统 - 基于物品的协同过滤")
print("=" * 60)

start_time = time.time()

# 1. 加载数据（限制大小）
print("\n[1] 加载数据...")
train = pd.read_csv("/root/飞猪数据集/交互数据/train.csv")
test = pd.read_csv("/root/飞猪数据集/交互数据/test.csv")

# 限制数据规模以加速
N_USERS = 30000
N_ITEMS = 10000

train = train[train['user_id'].isin(train['user_id'].unique()[:N_USERS])]
train = train[train['item_id'].isin(train['item_id'].unique()[:N_ITEMS])]

print(f"  训练集: {len(train)}, 测试集: {len(test)}")

# 2. 创建ID映射
print("\n[2] 创建ID映射...")
users = train['user_id'].unique()
items = train['item_id'].unique()
user2idx = {u: i for i, u in enumerate(users)}
item2idx = {it: i for i, it in enumerate(items)}
idx2item = {i: it for it, i in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"  用户数: {n_users}, 景点数: {n_items}")

# 3. 构建用户-物品交互（稀疏矩阵）
print("\n[3] 构建交互矩阵...")
rows = train['user_id'].map(user2idx).values
cols = train['item_id'].map(item2idx).values
data = np.ones(len(train))

user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
print(f"  矩阵: {user_item_matrix.shape}, 非零: {user_item_matrix.nnz}")

# 4. 计算物品相似度 (余弦相似度)
print("\n[4] 计算物品相似度...")
item_item_matrix = user_item_matrix.T @ user_item_matrix  # 物品-物品共现矩阵
item_norms = np.sqrt(item_item_matrix.diagonal())
item_norms[item_norms == 0] = 1  # 避免除零

# 归一化得到余弦相似度
item_similarity = item_item_matrix / item_norms[:, None] / item_norms[None, :]
item_similarity = item_similarity.tocsr()
print(f"  相似度矩阵: {item_similarity.shape}")

# 5. 生成推荐
print("\n[5] 评估模型...")

def recommend(user_id, k=10):
    if user_id not in user2idx:
        return []
    u_idx = user2idx[user_id]
    
    # 获取用户已交互的物品
    user_items = user_item_matrix[u_idx].indices
    if len(user_items) == 0:
        return []
    
    # 计算推荐分数（基于相似物品）
    scores = np.zeros(n_items)
    for item_idx in user_items:
        similarity = item_similarity[item_idx].toarray().flatten()
        scores += similarity
    
    # 排除已交互
    scores[user_items] = -np.inf
    
    # 取top-k
    top_k = np.argsort(scores)[-k:][::-1]
    return [idx2item[i] for i in top_k if scores[i] > -np.inf]

# 评估
hits = 0
total = 0
test_sample = test[test['user_id'].isin(user2idx.keys())]['user_id'].unique()[:1000]

for u in test_sample:
    gt = set(test[test['user_id'] == u]['item_id'])
    if not gt:
        continue
    recs = set(recommend(u, 10))
    if len(gt & recs) > 0:
        hits += 1
    total += 1

hit_rate = hits / total if total > 0 else 0
print(f"  Hit Rate@10: {hit_rate:.4f}")

# 6. 保存模型
print("\n[6] 保存模型...")
model = {
    "item_similarity": item_similarity,
    "user_item_matrix": user_item_matrix,
    "user2idx": user2idx,
    "item2idx": item2idx,
    "idx2item": idx2item,
    "metrics": {"hit_rate@10": hit_rate}
}

model_path = "/root/飞猪数据集/模型训练/model_item_cf.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

elapsed = time.time() - start_time
print(f"\n用时: {elapsed:.1f}秒")
print(f"模型: {model_path}")
print("=" * 60)
print("完成!")
print("=" * 60)