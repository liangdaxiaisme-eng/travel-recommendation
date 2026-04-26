#!/usr/bin/env python3
"""
飞猪旅行推荐系统 - 训练推荐模型
使用Item-CF算法，基于用户行为数据训练
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle
import os
import time

print("=" * 50)
print("飞猪旅行推荐系统 - 模型训练")
print("=" * 50)

# 数据路径
DATA_DIR = "/home/asd/论文资料/4/旅游推荐数据集"
BEHAVIOR_FILE = f"{DATA_DIR}/user_item_behavior_history.csv"
ITEM_FILE = f"{DATA_DIR}/item_profile.csv"
USER_FILE = f"{DATA_DIR}/user_profile.csv"
MODEL_FILE = f"{DATA_DIR}/model.pkl"

# 1. 加载数据
print("\n[1/4] 加载数据...")
start = time.time()

# 用户行为数据
behavior = pd.read_csv(BEHAVIOR_FILE, header=None, 
                       names=['user_id', 'item_id', 'action', 'timestamp'])
print(f"  加载用户行为数据: {len(behavior):,} 条")

# 景点数据
try:
    items = pd.read_csv(ITEM_FILE)
    print(f"  加载景点数据: {len(items):,} 条")
except:
    items = None
    print("  景点数据加载失败")

# 用户数据
try:
    users = pd.read_csv(USER_FILE)
    print(f"  加载用户数据: {len(users):,} 条")
except:
    users = None
    print("  用户数据加载失败")

print(f"  加载耗时: {time.time()-start:.2f}秒")

# 2. 数据预处理
print("\n[2/4] 数据预处理...")
start = time.time()

# 只保留点击行为
behavior = behavior[behavior['action'] == 'clk'].copy()
print(f"  点击行为: {len(behavior):,} 条")

# 获取所有用户和景点
all_users = behavior['user_id'].unique()
all_items = behavior['item_id'].unique()
print(f"  用户数: {len(all_users):,}")
print(f"  景点数: {len(all_items):,}")

# 创建ID映射
user2idx = {u: i for i, u in enumerate(all_users)}
idx2user = {i: u for u, i in user2idx.items()}
item2idx = {it: i for i, it in enumerate(all_items)}
idx2item = {i: it for it, i in item2idx.items()}

n_users = len(all_users)
n_items = len(all_items)

# 创建用户-景点交互矩阵
print("  构建用户-景点矩阵...")
user_item_matrix = defaultdict(set)
for _, row in behavior.iterrows():
    user_item_matrix[row['user_id']].add(row['item_id'])

# 计算景点热度
item_counts = behavior['item_id'].value_counts()
print(f"  预处理耗时: {time.time()-start:.2f}秒")

# 3. 训练Item-CF模型
print("\n[3/4] 训练Item-CF模型...")
start = time.time()

# 计算景点相似度（基于共同用户）
print("  计算景点相似度...")
item_similarity = defaultdict(dict)

# 只计算热门景点的相似度（top 5000）
top_items = item_counts.head(5000).index.tolist()

for i, item1 in enumerate(top_items):
    if i % 500 == 0:
        print(f"    进度: {i}/{len(top_items)}")
    
    users1 = user_item_matrix.get(item1, set())
    for item2 in top_items:
        if item1 >= item2:
            continue
        users2 = user_item_matrix.get(item2, set())
        common = len(users1 & users2)
        if common > 0:
            # Jaccard相似度
            sim = common / len(users1 | users2)
            if sim > 0.01:  # 只保留相似度>0.01的
                item_similarity[item1][item2] = sim
                item_similarity[item2][item1] = sim

print(f"  训练耗时: {time.time()-start:.2f}秒")
print(f"  相似景点对: {sum(len(v) for v in item_similarity.values())//2:,}")

# 4. 保存模型
print("\n[4/4] 保存模型...")
model_data = {
    'user2idx': user2idx,
    'idx2user': idx2user,
    'item2idx': item2idx,
    'idx2item': idx2item,
    'user_item_matrix': dict(user_item_matrix),
    'item_similarity': dict(item_similarity),
    'item_counts': item_counts.to_dict(),
    'n_users': n_users,
    'n_items': n_items,
    'items': items,
    'users': users
}

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_data, f)

print(f"  模型已保存: {MODEL_FILE}")
print(f"  模型大小: {os.path.getsize(MODEL_FILE)/1024/1024:.2f} MB")

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)