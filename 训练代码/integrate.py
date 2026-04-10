#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据整合脚本"""

import pandas as pd
import json
import os

print("=" * 60)
print("数据整合：评论与景点关联")
print("=" * 60)

# 加载评论数据
print("\n[1] 加载评论数据...")
with open("/root/飞猪数据集/原始数据/comments.json", "r", encoding="utf-8") as f:
    comments = json.load(f)
df = pd.DataFrame(comments)

city_col = 'city'
score_col = 'score'

评论数 = len(df)
城市数 = df[city_col].nunique()
平均分 = df[score_col].astype(float).mean()

print(f"  评论总数: {评论数}")
print(f"  涉及城市: {城市数}")
print(f"  平均评分: {平均分:.2f}")

# 保存处理后的评论
print("\n[2] 保存评论数据...")
output_path = "/root/飞猪数据集/原始数据/comments_processed.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
文件大小 = os.path.getsize(output_path) / 1024 / 1024
print(f"  已保存: comments_processed.csv ({文件大小:.1f} MB)")

# 统计总大小
print("\n[3] 数据集总大小...")
total = 0
datasets = {
    "用户数据": "/root/飞猪数据集/用户数据/users_processed.csv",
    "景点数据": "/root/飞猪数据集/景点数据/items_processed.csv", 
    "交互数据": "/root/飞猪数据集/交互数据/interactions_processed.csv",
    "评论数据": "/root/飞猪数据集/原始数据/comments_processed.csv"
}

for name, path in datasets.items():
    size = os.path.getsize(path) / 1024 / 1024
    total += size
    print(f"  {name}: {size:.1f} MB")

print(f"\n  总计: {total:.1f} MB")
print("=" * 60)
print("数据整合完成！")