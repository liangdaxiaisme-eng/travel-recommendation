#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞猪旅行推荐系统模型训练
使用协同过滤 + 深度学习混合推荐
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle

print("=" * 60)
print("飞猪旅行推荐系统模型训练")
print("=" * 60)

# 1. 加载数据
print("\n[1/6] 加载数据...")
users = pd.read_csv("/root/飞猪数据集/用户数据/users_processed.csv")
items = pd.read_csv("/root/飞猪数据集/景点数据/items_processed.csv")
interactions = pd.read_csv("/root/飞猪数据集/交互数据/interactions_processed.csv")

print(f"  用户: {len(users)}, 景点: {len(items)}, 交互: {len(interactions)}")

# 2. 创建ID映射
print("\n[2/6] 创建ID映射...")
user2idx = {u: i for i, u in enumerate(interactions['user_id'].unique())}
item2idx = {it: i for i, it in enumerate(interactions['item_id'].unique())}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: it for it, i in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"  用户数: {n_users}, 景点数: {n_items}")

# 3. 构建用户-景点交互矩阵
print("\n[3/6] 构建交互矩阵...")
交互矩阵 = np.zeros((n_users, n_items), dtype=np.float32)
for _, row in interactions.iterrows():
    if row['user_id'] in user2idx and row['item_id'] in item2idx:
        u_idx = user2idx[row['user_id']]
        i_idx = item2idx[row['item_id']]
        交互矩阵[u_idx, i_idx] = 1.0

稀疏度 = 1 - (交互矩阵.sum() / (n_users * n_items))
print(f"  矩阵形状: {交互矩阵.shape}")
print(f"  稀疏度: {稀疏度:.4f}")

# 4. 训练集/测试集划分
print("\n[4/6] 划分训练集/测试集...")
train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)

# 创建训练集交互矩阵
train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
for _, row in train_data.iterrows():
    if row['user_id'] in user2idx and row['item_id'] in item2idx:
        u_idx = user2idx[row['user_id']]
        i_idx = item2idx[row['item_id']]
        train_matrix[u_idx, i_idx] = 1.0

test_matrix = np.zeros((n_users, n_items), dtype=np.float32)
for _, row in test_data.iterrows():
    if row['user_id'] in user2idx and row['item_id'] in item2idx:
        u_idx = user2idx[row['user_id']]
        i_idx = item2idx[row['item_id']]
        test_matrix[u_idx, i_idx] = 1.0

print(f"  训练集: {train_matrix.sum():.0f} 交互")
print(f"  测试集: {test_matrix.mean()*10000:.2f} 交互")

# 5. 矩阵分解模型 (SVD)
print("\n[5/6] 训练SVD推荐模型...")
n_factors = 64  # 隐向量维度

# 初始化
np.random.seed(42)
user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

# SGD训练
learning_rate = 0.005
reg = 0.02
n_epochs = 20

print(f"  隐向量维度: {n_factors}, 学习率: {learning_rate}, 正则化: {reg}")

positive_samples = list(zip(*np.where(train_matrix > 0)))
n_samples = len(positive_samples)

for epoch in range(n_epochs):
    np.random.shuffle(positive_samples)
    total_loss = 0
    
    for i, (u, j) in enumerate(positive_samples):
        # 预测
        pred = np.dot(user_factors[u], item_factors[j]) + user_bias[u] + item_bias[j]
        error = train_matrix[u, j] - pred
        total_loss += error ** 2
        
        # 更新
        user_factors[u] += learning_rate * (error * item_factors[j] - reg * user_factors[u])
        item_factors[j] += learning_rate * (error * user_factors[u] - reg * item_factors[j])
        user_bias[u] += learning_rate * (error - reg * user_bias[u])
        item_bias[j] += learning_rate * (error - reg * item_bias[j])
        
        if (i + 1) % 100000 == 0:
            print(f"    Epoch {epoch+1}: {i+1}/{n_samples}")
    
    rmse = np.sqrt(total_loss / n_samples)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, RMSE: {rmse:.4f}")

# 6. 模型评估
print("\n[6/6] 模型评估...")

# 计算Recall@K
def evaluate_recommendations(user_factors, item_factors, user_bias, item_bias, test_matrix, train_matrix, k=10):
    recalls = []
    ndcgs = []
    
    test_users = np.where(test_matrix.sum(axis=1) > 0)[0]
    
    for u in test_users[:1000]:  # 评估前1000个用户
        # 生成推荐列表
        scores = np.dot(user_factors[u], item_factors.T) + user_bias[u] + item_bias
        scores[train_matrix[u] > 0] = -np.inf  # 排除已交互
        
        top_k = np.argsort(scores)[-k:][::-1]
        
        # 计算Recall
        relevant = set(np.where(test_matrix[u] > 0)[0])
        if len(relevant) > 0:
            recommended = set(top_k)
            recall = len(relevant & recommended) / len(relevant)
            recalls.append(recall)
            
            # NDCG
            dcg = 0
            for i, item in enumerate(top_k):
                if item in relevant:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(recalls), np.mean(ndcgs)

recall@10, ndcg@10 = evaluate_recommendations(user_factors, item_factors, user_bias, item_bias, test_matrix, train_matrix, k=10)
print(f"  Recall@10: {recall@10:.4f}")
print(f"  NDCG@10: {ndcg@10:.4f}")

# 保存模型
print("\n保存模型...")
model_path = "/root/飞猪数据集/模型训练/model_svd.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "n_factors": n_factors,
        "metrics": {"recall@10": recall@10, "ndcg@10": ndcg@10}
    }, f)

print(f"  模型已保存: {model_path}")

print("\n" + "=" * 60)
print("模型训练完成！")
print("=" * 60)