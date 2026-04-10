#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞猪旅行推荐系统 - 简化版矩阵分解
"""

import pandas as pd
import numpy as np
from scipy import sparse
import pickle
import time

print("=" * 60)
print("飞猪旅行推荐系统 - 模型训练")
print("=" * 60)

start_time = time.time()

# 1. 加载数据
print("\n[1] 加载数据...")
interactions = pd.read_csv("/root/飞猪数据集/交互数据/interactions_processed.csv")
train = pd.read_csv("/root/飞猪数据集/交互数据/train.csv")
test = pd.read_csv("/root/飞猪数据集/交互数据/test.csv")

print(f"  交互总数: {len(interactions)}")
print(f"  训练集: {len(train)}, 测试集: {len(test)}")

# 2. 创建ID映射
print("\n[2] 创建ID映射...")
all_users = list(set(train['user_id'].unique()) | set(test['user_id'].unique()))
all_items = list(set(train['item_id'].unique()) | set(test['item_id'].unique()))

user2idx = {u: i for i, u in enumerate(all_users)}
item2idx = {it: i for i, it in enumerate(all_items)}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"  用户数: {n_users}, 景点数: {n_items}")

# 3. 构建稀疏矩阵
print("\n[3] 构建交互矩阵...")
row, col, data = [], [], []
for _, r in train.iterrows():
    if r['user_id'] in user2idx and r['item_id'] in item2idx:
        row.append(user2idx[r['user_id']])
        col.append(item2idx[r['item_id']])
        data.append(1)

train_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_users, n_items), dtype=np.float32)
print(f"  矩阵形状: {train_matrix.shape}")
print(f"  非零元素: {train_matrix.nnz}")

# 4. SVD矩阵分解 (使用scipy)
print("\n[4] SVD矩阵分解...")
from scipy.sparse.linalg import svds

# 转换为密集矩阵的稀疏表示
train_dense = train_matrix.toarray()
# 中心化 (减去均值)
user_means = np.true_divide(train_dense.sum(axis=1), (train_dense != 0).sum(axis=1))
user_means = np.nan_to_num(user_means)

# SVD分解
k = 50  # 隐因子数
print(f"  隐因子数: {k}")

# 对于稀疏矩阵，使用arpack
try:
    U, sigma, Vt = svds(train_matrix.astype(float), k=k)
    sigma = np.diag(sigma)
    
    # 重构预测矩阵
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    print("  SVD分解完成!")
except Exception as e:
    print(f"  SVD出错: {e}")
    # 使用随机方法
    np.random.seed(42)
    U = np.random.randn(n_users, k) * 0.1
    sigma = np.random.randn(k) * 0.1
    Vt = np.random.randn(k, n_items) * 0.1
    predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)

# 5. 评估
print("\n[5] 模型评估...")

def get_top_k(user_id, k=10):
    if user_id not in user2idx:
        return []
    u_idx = user2idx[user_id]
    scores = predicted_ratings[u_idx]
    # 排除已交互
    user_items = set(train[train['user_id'] == user_id]['item_id'])
    for it in user_items:
        if it in item2idx:
            scores[item2idx[it]] = -np.inf
    top_k_idx = np.argsort(scores)[-k:][::-1]
    return [list(item2idx.keys())[list(item2idx.values()).index(i)] for i in top_k_idx]

# 计算Hit Rate
hits = 0
total = 0
test_users = test['user_id'].unique()[:2000]

for u in test_users:
    if u not in user2idx:
        continue
    ground_truth = set(test[test['user_id'] == u]['item_id'])
    if len(ground_truth) == 0:
        continue
    recommendations = set(get_top_k(u, 10))
    hit = len(ground_truth & recommendations)
    if hit > 0:
        hits += 1
    total += 1

hit_rate = hits / total if total > 0 else 0
print(f"  Hit Rate@10: {hit_rate:.4f}")

# 6. 保存模型
print("\n[6] 保存模型...")
model = {
    "U": U,
    "sigma": sigma,
    "Vt": Vt,
    "predicted_ratings": predicted_ratings,
    "user2idx": user2idx,
    "item2idx": item2idx,
    "metrics": {"hit_rate@10": hit_rate}
}

model_path = "/root/飞猪数据集/模型训练/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

elapsed = time.time() - start_time
print(f"\n总用时: {elapsed:.1f}秒")
print(f"模型已保存: {model_path}")
print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)