#!/usr/bin/env python3
"""
飞猪旅行数据集预处理脚本
用于旅游景点推荐系统
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("飞猪旅行数据集预处理")
print("=" * 60)

# 数据路径
DATA_DIR = "/root/飞猪数据集/原始数据"
OUTPUT_DIR = "/root/飞猪数据集"

# 1. 加载用户数据
print("\n[1/5] 加载用户数据...")
user_df = pd.read_csv(f"{DATA_DIR}/user_profile.csv", header=None, 
                       names=['user_id', 'age', 'gender', 'occupation', 'city_id', 'user_tags'])
print(f"  用户总数: {len(user_df):,}")
print(f"  数据样例:\n{user_df.head(3)}")

# 2. 加载景点数据
print("\n[2/5] 加载景点数据...")
item_df = pd.read_csv(f"{DATA_DIR}/item_profile.csv", header=None,
                       names=['item_id', 'category_id', 'city_id', 'item_tags'])
print(f"  景点总数: {len(item_df):,}")
print(f"  数据样例:\n{item_df.head(3)}")

# 3. 加载交互数据
print("\n[3/5] 加载交互数据...")
# 由于数据量大，先查看行数
import subprocess
result = subprocess.run(f"wc -l {DATA_DIR}/user_item_behavior_history.csv", 
                        shell=True, capture_output=True, text=True)
total_lines = int(result.stdout.strip().split()[0])
print(f"  交互记录总数: {total_lines:,}")

# 分块读取，筛选购买行为
chunk_size = 5000000
pay_data = []

print("  正在筛选购买行为数据 (pay)...")
for i, chunk in enumerate(pd.read_csv(f"{DATA_DIR}/user_item_behavior_history.csv", 
                                        header=None, chunksize=chunk_size,
                                        names=['user_id', 'item_id', 'behavior_type', 'timestamp'])):
    pay_chunk = chunk[chunk['behavior_type'] == 'pay']
    if len(pay_chunk) > 0:
        pay_data.append(pay_chunk)
    print(f"    处理块 {i+1}: {len(chunk):,} 条, 购买行为: {len(pay_chunk):,}")
    if i >= 9:  # 先处理前10块看看规模
        break

# 合并购买数据
if pay_data:
    interaction_df = pd.concat(pay_data, ignore_index=True)
    print(f"  购买交互总数: {len(interaction_df):,}")
else:
    interaction_df = pd.DataFrame(columns=['user_id', 'item_id', 'behavior_type', 'timestamp'])
    print("  警告: 未找到购买行为数据")

# 4. 数据筛选
print("\n[4/5] 数据筛选与清洗...")

# 筛选有购买记录的用户和景点
valid_users = interaction_df['user_id'].unique()
valid_items = interaction_df['item_id'].unique()

# 保留有购买记录的用户画像
user_filtered = user_df[user_df['user_id'].isin(valid_users)]
print(f"  有效用户数: {len(user_filtered):,}")

# 保留有购买记录的景点
item_filtered = item_df[item_df['item_id'].isin(valid_items)]
print(f"  有效景点数: {len(item_filtered):,}")

# 5. 保存处理后的数据
print("\n[5/5] 保存处理后的数据...")

# 保存用户数据
user_filtered.to_csv(f"{OUTPUT_DIR}/用户数据/users_processed.csv", index=False)
print(f"  ✅ 用户数据已保存: {OUTPUT_DIR}/用户数据/users_processed.csv")

# 保存景点数据
item_filtered.to_csv(f"{OUTPUT_DIR}/景点数据/items_processed.csv", index=False)
print(f"  ✅ 景点数据已保存: {OUTPUT_DIR}/景点数据/items_processed.csv")

# 保存交互数据
interaction_df.to_csv(f"{OUTPUT_DIR}/交互数据/interactions_processed.csv", index=False)
print(f"  ✅ 交互数据已保存: {OUTPUT_DIR}/交互数据/interactions_processed.csv")

# 6. 划分训练集/测试集
print("\n[6/6] 划分训练集/测试集...")

# 按时间排序
interaction_df = interaction_df.sort_values('timestamp')

# 80% 训练，20% 测试
train_size = int(len(interaction_df) * 0.8)
train_df = interaction_df.iloc[:train_size]
test_df = interaction_df.iloc[train_size:]

train_df.to_csv(f"{OUTPUT_DIR}/交互数据/train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/交互数据/test.csv", index=False)

print(f"  训练集: {len(train_df):,} 条")
print(f"  测试集: {len(test_df):,} 条")

# 7. 生成数据统计报告
print("\n" + "=" * 60)
print("数据预处理完成！统计报告：")
print("=" * 60)
print(f"  有效用户数: {len(user_filtered):,}")
print(f"  有效景点数: {len(item_filtered):,}")
print(f"  购买交互数: {len(interaction_df):,}")
print(f"  训练集大小: {len(train_df):,}")
print(f"  测试集大小: {len(test_df):,}")
print("=" * 60)

# 保存统计信息
stats = {
    "处理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "有效用户数": len(user_filtered),
    "有效景点数": len(item_filtered),
    "购买交互数": len(interaction_df),
    "训练集大小": len(train_df),
    "测试集大小": len(test_df)
}

with open(f"{OUTPUT_DIR}/数据统计.txt", "w") as f:
    for k, v in stats.items():
        f.write(f"{k}: {v}\n")

print("\n✅ 预处理完成！")
