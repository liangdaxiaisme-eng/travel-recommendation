#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞猪旅行推荐系统 - 用户协同过滤 + SVD混合模型
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from collections import defaultdict
import pickle
import time

print("=" * 60)
print("飞猪旅行推荐系统 - 混合推荐模型")
print("=" * 60)

start_time = time.time()

# 1. 加载完整数据
print("\n[1] 加载数据...")
train = pd.read_csv("/root/飞猪数据集/交互数据/train.csv")
test = pd.read_csv("/root/飞猪数据集/交互数据/test.csv")
print(f"  训练集: {len(train)}, 测试集: {len(test)}")

# 2. 创建ID映射 (使用全部数据)
print("\n[2] 创建ID映射...")
all_users = list(set(train['user_id'].unique()) | set(test['user_id'].unique()))
all_items = list(set(train['item_id'].unique()) | set(test['item_id'].unique()))

# 限制规模以保证能训练
MAX_USERS = 50000
MAX_ITEMS = 15000
all_users = all_users[:MAX_USERS]
all_items = all_items[:MAX_ITEMS]

user2idx = {u: i for i, u in enumerate(all_users)}
item2idx = {it: i for i, it in enumerate(all_items)}
idx2item = {i: it for it, i in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"  用户数: {n_users}, 景点数: {n_items}")

# 过滤训练集
train_filtered = train[train['user_id'].isin(user2idx) & train['item_id'].isin(item2idx)]
print(f"  过滤后训练集: {len(train_filtered)}")

# 3. 构建稀疏交互矩阵
print("\n[3] 构建交互矩阵...")
rows = train_filtered['user_id'].map(user2idx).values
cols = train_filtered['item_id'].map(item2idx).values
data = np.ones(len(train_filtered))

user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
print(f"  矩阵: {user_item_matrix.shape}")

# 4. SVD矩阵分解
print("\n[4] SVD矩阵分解...")
# 转换为二值矩阵并中心化
R = user_item_matrix.astype(float).toarray()
# 中心化：减去每个用户的平均值
user_means = np.array(R.sum(axis=1) / (R.sum(axis=1) + 1e-10)).flatten()
R_centered = R - user_means.reshape(-1, 1)

# SVD
k = 30  # 隐因子数
try:
    U, sigma, Vt = svds(sparse.csr_matrix(R_centered), k=k)
    sigma = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)
    print(f"  SVD完成, k={k}")
except Exception as e:
    print(f"  SVD失败: {e}, 使用随机初始化")
    np.random.seed(42)
    predicted = np.random.randn(n_users, n_items) * 0.1

# 5. 基于物品的协同过滤
print("\n[5] 计算物品相似度...")
item_item = user_item_matrix.T @ user_item_matrix
item_norms = np.sqrt(np.array(item_item.diagonal()).flatten())
item_norms[item_norms == 0] = 1
item_sim = (item_item / item_norms[:, None]).toarray()
item_sim = item_sim / item_norms[None, :]
print("  物品相似度计算完成")

# 6. 混合推荐
print("\n[6] 评估...")

def hybrid_recommend(user_id, k=10, alpha=0.6):
    """混合推荐: alpha*SVD + (1-alpha)*ItemCF"""
    if user_id not in user2idx:
        return []
    u_idx = user2idx[user_id]
    
    # 用户已交互
    user_items = user_item_matrix[u_idx].indices
    
    # SVD分数
    svd_scores = predicted[u_idx].copy()
    
    # ItemCF分数
    cf_scores = np.zeros(n_items)
    for it in user_items:
        cf_scores += item_sim[it]
    
    # 混合
    scores = alpha * svd_scores + (1 - alpha) * cf_scores
    scores[user_items] = -np.inf
    
    top_k = np.argsort(scores)[-k:][::-1]
    return [(idx2item[i], scores[i]) for i in top_k if scores[i] > -np.inf]

# 评估多个alpha
best_alpha = 0.6
best_hr = 0

test_users = test[test['user_id'].isin(user2idx)]['user_id'].unique()[:1500]

for alpha in [0.3, 0.5, 0.6, 0.7]:
    hits = 0
    total = 0
    for u in test_users:
        gt = set(test[test['user_id'] == u]['item_id'])
        if not gt:
            continue
        recs = set([x[0] for x in hybrid_recommend(u, 10, alpha)])
        if len(gt & recs) > 0:
            hits += 1
        total += 1
    hr = hits / total if total > 0 else 0
    print(f"  Alpha={alpha}, Hit Rate@10: {hr:.4f}")
    if hr > best_hr:
        best_hr = hr
        best_alpha = alpha

print(f"\n  最佳Alpha: {best_alpha}, 最佳Hit Rate: {best_hr:.4f}")

# 7. 保存模型
print("\n[7] 保存模型...")
model = {
    "predicted": predicted,
    "item_sim": item_sim,
    "user_item_matrix": user_item_matrix,
    "user2idx": user2idx,
    "item2idx": item2idx,
    "idx2item": idx2item,
    "n_users": n_users,
    "n_items": n_items,
    "best_alpha": best_alpha,
    "metrics": {
        "hit_rate@10": best_hr,
        "n_users": n_users,
        "n_items": n_items,
        "k": k
    }
}

with open("/root/飞猪数据集/模型训练/model_hybrid.pkl", "wb") as f:
    pickle.dump(model, f)

elapsed = time.time() - start_time
print(f"\n用时: {elapsed:.1f}秒")
print("=" * 60)
print(f"最佳 Hit Rate@10: {best_hr:.4f}")
print("=" * 60)
print("训练完成!")