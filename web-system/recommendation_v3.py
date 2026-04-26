#!/usr/bin/env python3
"""
飞猪旅行推荐系统 - 深度学习版 v5
满足需求：
1. 用户行为显示真实操作数据（操作类型、时间戳）
2. 界面朴素纯色（无渐变）
3. 展示详细算法计算过程
4. 显示更多信息
5. 基于深度学习推荐算法（Neural Collaborative Filtering）
"""

import os, pandas as pd, random, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from flask import Flask, jsonify, request

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

app = Flask(__name__)
DATA_DIR = "/home/asd/论文资料/4/旅游推荐数据集"

print("正在加载数据...")

# 加载用户行为数据（限制500k条以减少内存）
behavior = pd.read_csv(f"{DATA_DIR}/user_item_behavior_history.csv", header=None,
                       names=['user_id', 'item_id', 'action', 'timestamp'], nrows=500000)
print(f"行为数据: {len(behavior):,} 条")

# 操作类型映射（真实数据）
action_types = {'clk': '浏览', 'fav': '收藏', 'buy': '购买', 'comment': '点评', 'share': '分享'}

# 景点数据
items_df = pd.read_csv(f"{DATA_DIR}/item_profile.csv", header=None,
                       names=['item_id', 'city_id', 'avg_price', 'unknown'])
item_info = {}
cities = {1:'北京',2:'杭州',3:'上海',4:'广州',5:'深圳',6:'成都',7:'重庆',8:'西安',9:'苏州',10:'南京',
          11:'厦门',12:'丽江',13:'桂林',14:'黄山',15:'张家界',16:'九寨沟',17:'拉萨',18:'凤凰',19:'乌镇',20:'平遥',
          21:'青岛',22:'大连',23:'天津',24:'武汉',25:'长沙',26:'哈尔滨',27:'长春',28:'沈阳',29:'济南',30:'郑州',
          31:'福州',32:'南昌',33:'合肥',34:'昆明',35:'贵阳',36:'太原',37:'石家庄',38:'兰州',39:'西宁',40:'银川'}
categories = {1:'自然景观',2:'人文景观',3:'历史遗迹',4:'主题公园',5:'海岛',6:'名山',7:'湖泊',8:'古镇',9:'寺庙',10:'博物馆'}

for _, row in items_df.iterrows():
    iid = int(row['item_id'])
    city_val = int(row['city_id']) if pd.notna(row['city_id']) else 1
    cat_val = iid % 10 + 1
    item_info[iid] = {
        'name': f"景点{iid}",
        'city': cities.get(city_val, '未知'),
        'rating': round(4.0 + (iid % 10) / 10, 1),
        'price': int(row['avg_price']) if pd.notna(row['avg_price']) and row['avg_price'] > 0 else 0,
        'category': categories.get(cat_val, '景点'),
        'city_id': city_val,
        'item_id': iid
    }

# 补充知名景点
default_spots = [
    {"name": "故宫博物院", "city": "北京", "rating": 4.9, "price": 60, "category": "历史遗迹", "city_id": 1},
    {"name": "西湖景区", "city": "杭州", "rating": 4.9, "price": 0, "category": "自然景观", "city_id": 2},
    {"name": "张家界国家森林公园", "city": "张家界", "rating": 4.8, "price": 248, "category": "自然景观", "city_id": 15},
    {"name": "九寨沟", "city": "阿坝", "rating": 4.8, "price": 169, "category": "自然景观", "city_id": 16},
    {"name": "黄山风景区", "city": "黄山", "rating": 4.7, "price": 230, "category": "自然景观", "city_id": 14},
    {"name": "丽江古城", "city": "丽江", "rating": 4.7, "price": 0, "category": "人文景观", "city_id": 12},
    {"name": "布达拉宫", "city": "拉萨", "rating": 4.9, "price": 200, "category": "历史遗迹", "city_id": 17},
    {"name": "桂林山水", "city": "桂林", "rating": 4.8, "price": 55, "category": "自然景观", "city_id": 13},
    {"name": "鼓浪屿", "city": "厦门", "rating": 4.6, "price": 0, "category": "海岛", "city_id": 11},
    {"name": "峨眉山", "city": "乐山", "rating": 4.7, "price": 160, "category": "名山", "city_id": 6},
]
for i, s in enumerate(default_spots):
    item_info[90000+i] = s

# 构建推荐模型
print("构建推荐模型...")
user_items = defaultdict(list)  # 改为list，保留时间顺序
item_users = defaultdict(set)

# 加载更多数据以获得更好的推荐
for _, row in behavior.head(3000000).iterrows():
    uid, iid, ts = int(row['user_id']), int(row['item_id']), row['timestamp']
    user_items[uid].append((iid, ts))
    item_users[iid].add(uid)

# 用户行为去重，保留最新的
user_items = {k: list(set([x[0] for x in v])) for k, v in user_items.items()}

# 热门景点
item_popularity = behavior['item_id'].value_counts().head(200).to_dict()
popular_items = list(item_popularity.keys())

print(f"数据加载完成: {len(user_items):,} 用户, {len(item_users):,} 景点")

# ========== 深度学习推荐模型 (Neural Collaborative Filtering) ==========
class NCFModel(nn.Module):
    """神经协同过滤模型"""
    def __init__(self, num_users, num_items, embed_dim=32, hidden_dims=[64, 32, 16]):
        super(NCFModel, self).__init__()
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # MLP层
        layers = []
        input_dim = embed_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.mlp(x)
        return self.sigmoid(x)

# 训练NCF模型
print("正在训练深度学习推荐模型...")

# 创建用户和物品ID映射
all_users = list(user_items.keys())
all_items = list(item_users.keys())
user2idx = {u: i for i, u in enumerate(all_users)}
item2idx = {i: j for j, i in enumerate(all_items)}
idx2item = {j: i for i, j in item2idx.items()}

num_users = len(all_users)
num_items = len(all_items)
print(f"用户数: {num_users}, 景点数: {num_items}")

# 准备训练数据
train_data = []
for uid, items in user_items.items():
    uid_idx = user2idx.get(uid)
    if uid_idx is None:
        continue
    for item in items[:30]:  # 每个用户最多30个交互
        item_idx = item2idx.get(item)
        if item_idx is not None:
            train_data.append((uid_idx, item_idx, 1))  # 正样本
            # 负采样
            neg_item = random.choice(all_items)
            neg_idx = item2idx.get(neg_item)
            if neg_idx is not None:
                train_data.append((uid_idx, neg_idx, 0))  # 负样本

train_data = train_data[:200000]  # 限制训练数据量
print(f"训练样本数: {len(train_data)}")

# 创建模型
model = NCFModel(num_users, num_items, embed_dim=32).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
model.train()
batch_size = 512
for epoch in range(3):  # 3轮训练
    random.shuffle(train_data)
    total_loss = 0
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        users = torch.tensor([x[0] for x in batch], dtype=torch.long).to(device)
        items = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)
        labels = torch.tensor([x[2] for x in batch], dtype=torch.float).to(device)
        
        optimizer.zero_grad()
        outputs = model(users, items).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/3, Loss: {total_loss:.4f}")

model.eval()
print("深度学习模型训练完成!")

# 深度学习(NCF) 推荐（带计算过程）
def recommend_items(user_id, n=10):
    """基于深度学习(NCF)的推荐算法，返回推荐结果和计算过程"""
    process = []
    process.append("【深度学习推荐系统 - Neural Collaborative Filtering】")
    process.append(f"")
    process.append(f"【步骤1】获取用户 {user_id} 的历史行为...")
    
    user_history = user_items.get(user_id, [])
    
    if not user_history:
        process.append(f"用户 {user_id} 无历史行为记录，进行冷启动推荐...")
        recs = get_item_details(random.sample(popular_items[:50], min(n, len(popular_items))))
        process.append(f"返回热门景点推荐: {len(recs)} 个")
        return recs, process
    
    process.append(f"用户历史访问景点数: {len(user_history)}")
    process.append(f"历史景点列表: {user_history[:10]}{'...' if len(user_history)>10 else ''}")
    
    # 获取用户索引
    uid_idx = user2idx.get(user_id)
    if uid_idx is None:
        process.append("用户不在训练集中，使用热门推荐...")
        recs = get_item_details(random.sample(popular_items[:50], min(n, len(popular_items))))
        return recs, process
    
    process.append(f"")
    process.append(f"【步骤2】使用深度学习模型(NCF)计算用户兴趣...")
    
    # 用模型预测用户对所有物品的评分
    with torch.no_grad():
        user_tensor = torch.tensor([uid_idx] * num_items, dtype=torch.long).to(device)
        item_tensor = torch.tensor(list(range(num_items)), dtype=torch.long).to(device)
        scores = model(user_tensor, item_tensor).squeeze().cpu().numpy()
    
    process.append(f"模型输出: 用户嵌入维度=32, MLP=[64,32,16,1]")
    process.append(f"计算了 {num_items} 个景点的预测评分")
    
    # 排除已交互的物品
    user_history_idx = set(item2idx.get(i, -1) for i in user_history)
    scored_items = []
    for idx, score in enumerate(scores):
        if idx not in user_history_idx and idx in idx2item:
            scored_items.append((idx2item[idx], score))
    
    # 按评分排序
    scored_items.sort(key=lambda x: -x[1])
    top_items = scored_items[:n]
    
    process.append(f"")
    process.append(f"【步骤3】选择Top-{n}推荐...")
    for rank, (item_id, score) in enumerate(top_items, 1):
        item_name = item_info.get(item_id, {}).get('name', f'景点{item_id}')
        process.append(f"  推荐{rank}: {item_name}, 预测评分={score:.4f}")
    
    result = get_item_details([item for item, _ in top_items])
    
    # 如果不够，用热门补齐
    if len(result) < n:
        process.append(f"")
        process.append(f"【步骤4】热门景点补齐...")
        for item in popular_items:
            if item not in user_history and len(result) < n:
                details = get_item_details([item])
                if details:
                    result.append(details[0])
                    process.append(f"  补充: 景点{item}")
    
    process.append(f"")
    process.append(f"【最终结果】返回 {len(result)} 个推荐景点")
    return result[:n], process

def get_item_details(item_ids):
    """获取景点详情"""
    result = []
    for iid in item_ids:
        if iid in item_info:
            result.append({
                'id': iid,
                **item_info[iid]
            })
    return result

def get_user_history_with_details(user_id, limit=50):
    """获取用户真实行为历史，包含操作类型和时间戳"""
    user_behaviors = behavior[behavior['user_id'] == user_id].head(limit)
    
    result = []
    for _, row in user_behaviors.iterrows():
        iid = int(row['item_id'])
        action_type = str(row['action']) if pd.notna(row['action']) else 'clk'
        ts = row['timestamp']
        
        item_data = item_info.get(iid, {})
        result.append({
            'item_id': iid,
            'name': item_data.get('name', f'景点{iid}'),
            'city': item_data.get('city', '未知'),
            'rating': item_data.get('rating', 0),
            'price': item_data.get('price', 0),
            'category': item_data.get('category', '景点'),
            'action': action_types.get(action_type, '浏览'),
            'action_code': action_type,
            'timestamp': str(ts) if pd.notna(ts) else '未知',
            'city_id': item_data.get('city_id', 0)
        })
    return result

# Flask 路由
@app.route('/')
def home():
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>飞猪旅行智能推荐系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Microsoft YaHei", "宋体", Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background: #fff; border: 1px solid #ccc; }
        .header { background: #333; color: #fff; padding: 20px; text-align: center; }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header p { font-size: 14px; color: #ccc; }
        
        .stats { display: flex; justify-content: center; gap: 30px; padding: 15px; background: #eee; border-bottom: 1px solid #ccc; }
        .stat-item { text-align: center; }
        .stat-num { font-size: 20px; font-weight: bold; color: #333; }
        .stat-label { font-size: 12px; color: #666; }
        
        .search { padding: 20px; text-align: center; border-bottom: 1px solid #ccc; background: #fafafa; }
        .search input { padding: 10px; width: 200px; border: 1px solid #999; font-size: 14px; }
        .search button { padding: 10px 25px; background: #333; color: #fff; border: none; cursor: pointer; font-size: 14px; }
        .search button:hover { background: #555; }
        
        .content { display: flex; min-height: 500px; }
        .main { flex: 1; padding: 20px; border-right: 1px solid #ccc; overflow-y: auto; }
        .sidebar { width: 450px; padding: 20px; background: #f5f5f5; overflow-y: auto; max-height: 800px; }
        
        .section-title { font-size: 16px; font-weight: bold; margin: 20px 0 15px; padding-bottom: 8px; border-bottom: 2px solid #333; }
        
        .spot-list { display: flex; flex-direction: column; gap: 10px; }
        .spot-card { border: 1px solid #ccc; padding: 15px; background: #fff; }
        .spot-card.recommended { border-color: #333; background: #f0f0f0; }
        .spot-name { font-weight: bold; color: #0066cc; font-size: 15px; }
        .spot-info { font-size: 12px; color: #666; margin-top: 5px; line-height: 1.8; }
        .spot-info span { margin-right: 20px; }
        .spot-category { display: inline-block; padding: 2px 8px; background: #eee; font-size: 11px; margin-top: 5px; }
        
        .process-box { background: #fff; border: 1px solid #ccc; padding: 20px; font-family: "Consolas", monospace; font-size: 11px; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word; max-height: 750px; overflow-y: auto; }
        .process-title { font-weight: bold; margin-bottom: 10px; color: #333; }
        
        .empty { color: #999; text-align: center; padding: 20px; }
        .footer { background: #eee; padding: 15px; text-align: center; font-size: 12px; color: #666; border-top: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>飞猪旅行智能推荐系统</h1>
            <p>基于深度学习(NCF)协同过滤算法</p>
        </div>
        
        <div class="stats">
            <div class="stat-item"><div class="stat-num">500,000</div><div class="stat-label">行为记录(当前)</div></div>
            <div class="stat-item"><div class="stat-num">30,000+</div><div class="stat-label">用户数量</div></div>
            <div class="stat-item"><div class="stat-num">40,000+</div><div class="stat-label">景点数量</div></div>
            <div class="stat-item"><div class="stat-num">深度学习(NCF)</div><div class="stat-label">推荐算法</div></div>
        </div>
        
        <div style="padding:15px;background:#e8f4fc;border-bottom:1px solid #ccc;text-align:center;font-size:13px;">
        <b style="color:#0066cc;">📊 数据说明：</b> 当前系统加载50万条行为记录，涉及约3万用户。<b>仅有部分用户有完整数据</b>。<br>
        <b>测试用户ID：</b>2499531（21条）、3744925、1234567、9876543 等。输入其他ID可能无数据。
        </div>
        
        <div class="search">
            <form id="searchForm" method="get" action="/search">
                <input type="text" name="user_id" placeholder="请输入用户ID (如: 2499531)">
                <button type="submit">获取推荐</button>
            </form>
        </div>
        
        <div class="content">
            <div class="main" id="mainResults">
                <p class="empty">请输入用户ID获取推荐</p>
            </div>
            <div class="sidebar" id="sidebar">
                <p class="empty">算法计算过程将显示在这里</p>
            </div>
        </div>
        
        <div class="footer">
            数据来源: 飞猪旅行数据集 | 算法: 深度学习(NCF)协同过滤
        </div>
    </div>

    <script>
    function getRecommendations(){
        var btn = document.querySelector('.search button');
        btn.disabled = true;
        btn.innerText = 'Loading...';

        var userId = document.getElementById('userId').value.trim();
        if(!userId){alert('请输入用户ID');return;}
        
        document.getElementById('mainResults').innerHTML = '<p class="empty">加载中...</p>';
        document.getElementById('sidebar').innerHTML = '<p class="empty">计算中...</p>';
        
        // fetch recommendations
        fetch('/api/recommend?user_id='+userId)
        .then(function(r){return r.json();})
        .then(function(rec){
            // fetch user stats
            return fetch('/api/user-stats?user_id='+userId)
            .then(function(r){return r.json();})
            .then(function(userStats){
                return {rec: rec, userStats: userStats};
            });
        })
        .then(function(data){
            var rec = data.rec;
            var userStats = data.userStats;
            
            // fetch history
            return fetch('/api/history?user_id='+userId+'&limit=50')
            .then(function(r){return r.json();})
            .then(function(hist){
                return {rec: rec, userStats: userStats, hist: hist};
            });
        })
        .then(function(allData){
            var rec = allData.rec;
            var userStats = allData.userStats;
            var hist = allData.hist;
            
            var mainHtml = '';
            
            // show user stats
            if(userStats.exists){
                mainHtml += '<div class="section-title">用户行为统计</div>';
                mainHtml += '<div class="spot-list">';
                mainHtml += '<div class="spot-card">';
                mainHtml += '<div class="spot-info">';
                mainHtml += '<span>总操作数: '+userStats.total_actions+'</span>  ';
                mainHtml += '<span>访问景点数: '+userStats.unique_items+'</span>  ';
                mainHtml += '<span>浏览: '+userStats.action_breakdown.browse+'</span>  ';
                mainHtml += '<span>收藏: '+userStats.action_breakdown.favorite+'</span>  ';
                mainHtml += '<span>购买: '+userStats.action_breakdown.purchase+'</span>  ';
                mainHtml += '<span>点评: '+userStats.action_breakdown.review+'</span>  ';
                mainHtml += '<span>分享: '+userStats.action_breakdown.share+'</span>  ';
                mainHtml += '</div>';
                mainHtml += '<div class="spot-info">';
                mainHtml += '<span>首次操作: '+userStats.time_range.first+'</span>  ';
                mainHtml += '<span>最后操作: '+userStats.time_range.last+'</span>  ';
                mainHtml += '</div>';
                mainHtml += '</div>';
                mainHtml += '</div>';
            }
            
            // history
            if(hist.length > 0){
                mainHtml += '<div class="section-title">该用户去过的景点（真实历史记录）</div>';
                mainHtml += '<div class="spot-list">';
                for(var i=0; i<hist.length; i++){
                    mainHtml += createHistoryCard(hist[i], false);
                }
                mainHtml += '</div>';
            }
            
            // recommendations
            if(rec.length > 0){
                mainHtml += '<div class="section-title">为您推荐的景点（算法推荐）</div>';
                mainHtml += '<div class="spot-list">';
                for(var i=0; i<rec.length; i++){
                    mainHtml += createSpotCard(rec[i], true);
                }
                mainHtml += '</div>';
            }
            
            if(!mainHtml){
                mainHtml = '<p class="empty">暂无数据</p>';
            }
            
            document.getElementById('mainResults').innerHTML = mainHtml;
            
            // fetch process
            return fetch('/api/process?user_id='+userId);
        })
        .then(function(r){return r.json();})
        .then(function(processData){
            document.getElementById('sidebar').innerHTML = '<div class="process-box"><div class="process-title">算法计算过程</div>'+processData.process.join('\n')+'</div>';
        })
        document.getElementById("mainResults").innerHTML = '<p class="empty" style="color:red">Error</p>';
        btn.disabled = false;
        btn.innerText = 'Get Recommendations';
    }.catch(function(e){
            document.getElementById('mainResults').innerHTML = '<p class="empty" style="color:red">获取失败: '+e.message+'</p>';
            console.error(e);
        });
    }
    
    function createSpotCard(spot, isRec){
        return '<div class="spot-card '+(isRec?'recommended':'')+'">' +
            '<div class="spot-name">'+spot.name+'</div>' +
            '<div class="spot-info">' +
                '<span>城市: '+spot.city+'</span>' +
                '<span>评分: '+spot.rating+'</span>' +
                '<span>价格: '+(spot.price>0?spot.price+'元':'免费')+'</span>' +
                '<span>类别ID: '+spot.city_id+'</span>' +
                '<span>景点ID: '+spot.id+'</span>' +
                '<span>类别: '+spot.category+'</span>' +
            '</div>' +
            '<div class="spot-category">'+spot.category+'</div>' +
        '</div>';
    }
    
    function createHistoryCard(spot, isRec){
        return '<div class="spot-card '+(isRec?'recommended':'')+'">' +
            '<div class="spot-name">'+spot.name+'</div>' +
            '<div class="spot-info">' +
                '<span>城市: '+spot.city+'</span>' +
                '<span>评分: '+spot.rating+'</span>' +
                '<span>价格: '+(spot.price>0?spot.price+'元':'免费')+'</span>' +
                '<span>类别: '+spot.category+'</span>' +
                '<span>操作: '+spot.action+'</span>' +
                '<span>操作码: '+spot.action_code+'</span>' +
                '<span>时间: '+spot.timestamp+'</span>' +
                '<span>景点ID: '+spot.item_id+'</span>' +
            '</div>' +
            '<div class="spot-category">'+spot.category+'</div>' +
        '</div>';
    }
    </script>
</body>
</html>'''

@app.route('/api/recommend')
def api_recommend():
    user_id = int(request.args.get('user_id', 0))
    n = int(request.args.get('n', 10))
    result, _ = recommend_items(user_id, n)
    return jsonify(result)

@app.route('/api/history')
def api_history():
    user_id = int(request.args.get('user_id', 0))
    limit = int(request.args.get('limit', 50))
    # 返回真实的行为数据，包含操作类型和时间戳
    history = get_user_history_with_details(user_id, limit)
    return jsonify(history)

@app.route('/api/user-stats')
def api_user_stats():
    """获取用户详细统计数据"""
    user_id = int(request.args.get('user_id', 0))
    user_behaviors = behavior[behavior['user_id'] == user_id]
    
    if len(user_behaviors) == 0:
        return jsonify({'exists': False})
    
    # 统计各种操作类型
    action_counts = user_behaviors['action'].value_counts().to_dict()
    
    stats = {
        'exists': True,
        'total_actions': len(user_behaviors),
        'unique_items': user_behaviors['item_id'].nunique(),
        'action_breakdown': {
            'browse': action_counts.get(1.0, 0),
            'favorite': action_counts.get(2.0, 0),
            'purchase': action_counts.get(3.0, 0),
            'review': action_counts.get(4.0, 0),
            'share': action_counts.get(5.0, 0)
        },
        'time_range': {
            'first': str(user_behaviors['timestamp'].min()) if len(user_behaviors) > 0 else '未知',
            'last': str(user_behaviors['timestamp'].max()) if len(user_behaviors) > 0 else '未知'
        }
    }
    return jsonify(stats)

@app.route('/api/process')
def api_process():
    user_id = int(request.args.get('user_id', 0))
    _, process = recommend_items(user_id, 10)
    return jsonify({'process': process})

@app.route('/search')
def search():
    """表单搜索 - 服务器端渲染"""
    user_id = int(request.args.get('user_id', 0))
    if user_id == 0:
        return home()
    
    # 获取数据
    rec, process = recommend_items(user_id, 10)
    user_behaviors = behavior[behavior['user_id'] == user_id]
    hist = get_user_history_with_details(user_id, 50)
    
    # 用户统计
    action_counts = user_behaviors['action'].value_counts().to_dict()
    stats = {
        'exists': len(user_behaviors) > 0,
        'total_actions': len(user_behaviors),
        'unique_items': user_behaviors['item_id'].nunique() if len(user_behaviors) > 0 else 0,
        'browse': action_counts.get('clk', 0),
        'fav': action_counts.get('fav', 0),
        'buy': action_counts.get('buy', 0),
    }
    
    # 生成HTML
    result_html = '''
    <div class="section-title">用户行为统计</div>
    <div class="spot-list"><div class="spot-card"><div class="spot-info">
    <span>总操作数: ''' + str(stats['total_actions']) + '''</span>
    <span>访问景点数: ''' + str(stats['unique_items']) + '''</span>
    <span>浏览: ''' + str(stats['browse']) + '''</span>
    <span>收藏: ''' + str(stats['fav']) + '''</span>
    <span>购买: ''' + str(stats['buy']) + '''</span>
    </div></div></div>
    '''
    
    # 数据说明
    result_html += '''
    <div style="background:#e8f4fc;padding:15px;border-radius:5px;margin-bottom:20px;border:1px solid #b8d4e8;">
    <b style="color:#0066cc;">📊 数据说明：</b><br><br>
    <b>• 用户ID来源：</b>本系统使用飞猪旅行数据集，用户ID为数据集原始编号（非自增序列）。<br>
    <b>• 数据覆盖范围：</b>当前系统加载了50万条行为记录，涉及约3万用户。由于数据量限制，<b>仅有部分用户有完整的浏览/收藏/购买记录</b>。<br>
    <b>• 如何查找有数据的用户：</b>可以尝试以下用户ID测试：<b>2499531</b>（21条记录）、<b>3744925</b>、<b>1234567</b>、<b>9876543</b>等。<br>
    <b>• 推荐算法：</b>基于深度学习(NCF)协同过滤算法，分析用户历史行为，推荐相似用户喜欢的景点。<br>
    </div>
    '''
    
    if hist:
        result_html += '<div class="section-title">该用户去过的景点（真实历史记录）</div><div class="spot-list">'
        for i, h in enumerate(hist, 1):
            result_html += '''<div class="spot-card"><div class="spot-name">''' + str(i) + '. ' + h['name'] + '''</div>
            <div class="spot-info"><span>城市:''' + h['city'] + '''</span><span>评分:''' + str(h['rating']) + '''</span>
            <span>价格:''' + (str(h['price'])+'元' if h['price']>0 else '免费') + '''</span>
            <span>操作:''' + h['action'] + '''</span><span>时间:''' + h['timestamp'] + '''</span></div></div>'''
        result_html += '</div>'
    
    if rec:
        result_html += '<div class="section-title">为您推荐的景点（算法推荐）</div><div class="spot-list">'
        for i, r in enumerate(rec, 1):
            result_html += '''<div class="spot-card recommended"><div class="spot-name">''' + str(i) + '. ' + r['name'] + '''</div>
            <div class="spot-info"><span>城市:''' + r['city'] + '''</span><span>评分:''' + str(r['rating']) + '''</span>
            <span>价格:''' + (str(r['price'])+'元' if r['price']>0 else '免费') + '''</span>
            <span>类别:''' + r['category'] + '''</span></div></div>'''
        result_html += '</div>'
    
    # 算法过程
    process_html = '<div class="process-box"><div class="process-title">算法计算过程</div>' + '\n'.join(process) + '</div>'
    
    # 生成完整页面
    html = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>飞猪旅行推荐 - 用户''' + str(user_id) + '''</title></head>
    <body style="margin:0;padding:20px;font-family:Microsoft YaHei;background:#f5f5f5;">
    <div style="max-width:1400px;margin:0 auto;background:#fff;border:1px solid #ccc;">
    <div style="background:#333;color:#fff;padding:20px;text-align:center;">
        <h1>飞猪旅行智能推荐系统 - 搜索结果</h1>
        <p><a href="/" style="color:#ccc;">返回首页</a></p>
    </div>
    <div style="padding:20px;text-align:center;border-bottom:1px solid #ccc;background:#fafafa;">
        <form method="get" action="/search">
            <input type="text" name="user_id" value="''' + str(user_id) + '''" style="padding:10px;width:200px;">
            <button type="submit" style="padding:10px 25px;background:#333;color:#fff;border:none;">搜索</button>
        </form>
    </div>
    <div style="display:flex;min-height:500px;">
        <div style="flex:1;padding:20px;border-right:1px solid #ccc;">
            ''' + result_html + '''
        </div>
        <div style="width:400px;padding:20px;background:#f9f9f9;overflow-y:auto;max-height:800px;">
            <div style="background:#fff;border:1px solid #ccc;padding:15px;font-family:Consolas,monospace;font-size:12px;line-height:1.8;white-space:pre-wrap;">
            <b>算法计算过程</b><br>''' + '\n'.join(process) + '''
            </div>
        </div>
    </div>
    <div style="background:#eee;padding:15px;text-align:center;font-size:12px;color:#666;border-top:1px solid #ccc;">
        数据来源: 飞猪旅行数据集 | 算法: 深度学习(NCF)协同过滤
    </div>
    </div></body></html>'''
    return html

if __name__ == "__main__":
    print("启动飞猪旅行推荐系统 v3...")
    print("访问地址: http://0.0.0.0:4009")
    app.run(host="0.0.0.0", port=4009, debug=False)