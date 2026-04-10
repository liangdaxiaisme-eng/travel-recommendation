#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞猪旅行推荐系统 - 深度学习模型
NeuMF (神经协同过滤) + 注意力机制
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
import pickle
import time

print("=" * 60)
print("飞猪旅行推荐系统 - 深度学习模型 (NeuMF + Attention)")
print("=" * 60)

device = torch.device('cpu')  # 服务器无GPU，用CPU
print(f"设备: {device}")

start_time = time.time()

# 1. 加载数据
print("\n[1] 加载数据...")
train = pd.read_csv("/root/飞猪数据集/交互数据/train.csv")
test = pd.read_csv("/root/飞猪数据集/交互数据/test.csv")

# 限制规模（深度学习需要更多内存）
MAX_USERS = 30000
MAX_ITEMS = 10000

train = train[train['user_id'].isin(train['user_id'].unique()[:MAX_USERS])]
train = train[train['item_id'].isin(train['item_id'].unique()[:MAX_ITEMS])]

# 创建ID映射
users = list(set(train['user_id'].unique()) | set(test['user_id'].unique()))
items = list(set(train['item_id'].unique()) | set(test['item_id'].unique()))
users = [u for u in users if u in train['user_id'].unique()][:MAX_USERS]
items = [i for i in items if i in train['item_id'].unique()][:MAX_ITEMS]

user2idx = {u: i for i, u in enumerate(users)}
item2idx = {it: i for i, it in enumerate(items)}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"  用户数: {n_users}, 景点数: {n_items}")
print(f"  训练集: {len(train)}, 测试集: {len(test)}")

# 2. 数据集类
class ReviewDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = []
        self.items = []
        self.labels = []
        
        for _, row in df.iterrows():
            if row['user_id'] in user2idx and row['item_id'] in item2idx:
                self.users.append(user2idx[row['user_id']])
                self.items.append(item2idx[row['item_id']])
                self.labels.append(1)
        
        self.users = torch.LongTensor(self.users)
        self.items = torch.LongTensor(self.items)
        self.labels = torch.FloatTensor(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# 3. NeuMF模型 + 注意力机制
class AttentionLayer(nn.Module):
    """注意力机制层"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: [batch, embed_dim]
        attn_weights = self.attention(x)  # [batch, 1]
        return torch.softmax(attn_weights, dim=0) * x

class NeuMF(nn.Module):
    """神经协同过滤 + 注意力"""
    def __init__(self, n_users, n_items, embed_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        # GMF部分
        self.gmf_embed_user = nn.Embedding(n_users, embed_dim)
        self.gmf_embed_item = nn.Embedding(n_items, embed_dim)
        
        # MLP部分
        self.mlp_embed_user = nn.Embedding(n_users, embed_dim)
        self.mlp_embed_item = nn.Embedding(n_items, embed_dim)
        
        mlp_layers = []
        input_dim = embed_dim * 2
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 注意力层
        self.attention = AttentionLayer(hidden_dims[-1])
        
        # 最终预测层
        self.fc = nn.Linear(hidden_dims[-1] + embed_dim, 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user_ids, item_ids):
        # GMF
        user_gmf = self.gmf_embed_user(user_ids)
        item_gmf = self.gmf_embed_item(item_ids)
        gmf_out = user_gmf * item_gmf
        
        # MLP
        user_mlp = self.mlp_embed_user(user_ids)
        item_mlp = self.mlp_embed_item(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_out = self.mlp(mlp_input)
        
        # 注意力
        attn_out = self.attention(mlp_out)
        
        # 融合
        concat = torch.cat([gmf_out, attn_out], dim=1)
        output = torch.sigmoid(self.fc(concat))
        
        return output.squeeze()

# 注意力可视化（可解释性）
class AttentionVisualizer:
    """注意力权重可视化模块"""
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
    
    def get_attention(self, user_ids, item_ids):
        """获取注意力权重"""
        self.model.eval()
        with torch.no_grad():
            user_mlp = self.model.mlp_embed_user(user_ids)
            item_mlp = self.model.mlp_embed_item(item_ids)
            mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
            
            # 手动计算注意力
            for i, layer in enumerate(self.model.mlp):
                if isinstance(layer, nn.Linear):
                    mlp_input = layer(mlp_input)
                elif isinstance(layer, nn.ReLU):
                    mlp_input = torch.relu(mlp_input)
            
            # 注意力权重
            attn = self.model.attention.attention(mlp_input)
            return attn.numpy()

# 4. 训练
print("\n[2] 训练NeuMF模型...")
train_dataset = ReviewDataset(train, user2idx, item2idx)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

model = NeuMF(n_users, n_items, embed_dim=32, hidden_dims=[64, 32, 16]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 5
print(f"  隐向量维度: 32")
print(f"  隐藏层: [64, 32, 16]")
print(f"  批次大小: 512")
print(f"  训练轮数: {n_epochs}")

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (users, items, labels) in enumerate(train_loader):
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/{n_epochs}, Avg Loss: {avg_loss:.4f}")

# 5. 评估
print("\n[3] 评估模型...")
model.eval()

def evaluate(model, test, user2idx, item2idx, k=10):
    hits = 0
    total = 0
    
    # 按用户分组
    test_users = test[test['user_id'].isin(user2idx)]['user_id'].unique()[:500]
    
    for user_id in test_users:
        if user_id not in user2idx:
            continue
        user_idx = torch.LongTensor([user2idx[user_id]]).to(device)
        
        # 获取用户所有可能的物品预测
        all_items = torch.LongTensor(list(range(n_items))).to(device)
        with torch.no_grad():
            scores = model(user_idx.repeat(n_items), all_items).numpy()
        
        # 排除训练集中已交互的
        train_items = set(train[train['user_id'] == user_id]['item_id'])
        for it in train_items:
            if it in item2idx:
                scores[item2idx[it]] = -1e9
        
        # Top-K
        top_k_idx = np.argsort(scores)[-k:][::-1]
        top_k_items = set([list(item2idx.keys())[list(item2idx.values()).index(i)] for i in top_k_idx])
        
        # 测试集真实交互
        gt = set(test[test['user_id'] == user_id]['item_id'])
        
        if len(gt & top_k_items) > 0:
            hits += 1
        total += 1
    
    return hits / total if total > 0 else 0

hit_rate = evaluate(model, test, user2idx, item2idx, k=10)
print(f"  Hit Rate@10: {hit_rate:.4f}")

# 6. 注意力分析
print("\n[4] 注意力权重分析...")
attn_viz = AttentionVisualizer(model)
sample_users = list(user2idx.keys())[:10]
sample_user_idx = torch.LongTensor([user2idx[u] for u in sample_users[:3]]).to(device)
sample_items = torch.LongTensor([0, 1, 2]).to(device)
attn_weights = attn_viz.get_attention(sample_user_idx, sample_items)
print(f"  样本注意力权重: {attn_weights[:3].flatten()}")

# 7. 保存模型
print("\n[5] 保存模型...")
model_data = {
    "model": model.state_dict(),
    "user2idx": user2idx,
    "item2idx": item2idx,
    "n_users": n_users,
    "n_items": n_items,
    "embed_dim": 32,
    "metrics": {"hit_rate@10": hit_rate},
    "attention_weights": attn_weights.tolist() if attn_weights is not None else None
}

torch.save(model_data, "/root/飞猪数据集/模型训练/model_neumf.pt")
print("  已保存: model_neumf.pt")

elapsed = time.time() - start_time
print(f"\n总用时: {elapsed:.1f}秒")
print("=" * 60)
print(f"NeuMF + Attention Hit Rate@10: {hit_rate:.4f}")
print("=" * 60)
print("训练完成!")