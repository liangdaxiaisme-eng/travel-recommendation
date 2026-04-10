import pandas as pd
import json
import os

print("=" * 60)
print("数据整合：评论与景点关联")
print("=" * 60)

# 加载景点数据
print("\n[1] 加载景点数据...")
items = pd.read_csv("/root/飞猪数据集/景点数据/items_processed.csv")
print(f"  景点数: {len(items)}")

# 加载评论数据
print("\n[2] 加载评论数据...")
with open("/root/飞猪数据集/原始数据/comments.json", "r", encoding="utf-8") as f:
    comments = json.load(f)
comments_df = pd.DataFrame(comments)
print(f"  评论数: {len(comments_df)}")

# 数据统计
print("\n[3] 数据统计...")
print(f"  评论涉及城市: {comments_df["city"].nunique()}")
print(f"  平均评分: {comments_df["score"].astype(float).mean():.2f}")
print(f"  平均帮助数: {comments_df["helpful_count"].mean():.1f}")

# 保存
print("\n[4] 保存整合数据...")
comments_df.to_csv("/root/飞猪数据集/原始数据/comments_processed.csv", index=False, encoding="utf-8")
print("  保存完成: comments_processed.csv")

# 数据集清单
print("\n" + "=" * 60)
print("数据集完整清单")
print("=" * 60)

datasets = {
    "用户数据": "/root/飞猪数据集/用户数据/users_processed.csv",
    "景点数据": "/root/飞猪数据集/景点数据/items_processed.csv", 
    "交互数据": "/root/飞猪数据集/交互数据/interactions_processed.csv",
    "评论数据": "/root/飞猪数据集/原始数据/comments_processed.csv"
}

total = 0
for name, path in datasets.items():
    size = os.path.getsize(path)
    total += size
    print(f"  {name}: {size/1024/1024:.1f} MB")

print(f"\n  总大小: {total/1024/1024:.1f} MB")
print("=" * 60)
print("数据整合完成！")
