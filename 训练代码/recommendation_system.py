#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞猪旅行推荐系统 - Web原型
简单的景点推荐Demo
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import pickle
import numpy as np
import random

app = Flask(__name__)

# 加载模型和数据
print("加载模型和数据...")
try:
    with open("/root/飞猪数据集/模型训练/model_hybrid.pkl", "rb") as f:
        model = pickle.load(f)
    
    # 景点数据
    items_df = pd.read_csv("/root/飞猪数据集/景点数据/items_processed.csv")
    
    print(f"模型加载成功")
    print(f"景点数: {len(items_df)}")
except Exception as e:
    print(f"加载失败: {e}")
    model = None
    items_df = None

# 模拟景点信息
SAMPLE_SPOTS = [
    {"id": 1, "name": "故宫", "city": "北京", "category": "历史文化", "rating": 4.8, "price": 60, "desc": "世界文化遗产，中国古代皇宫"},
    {"id": 2, "name": "西湖", "city": "杭州", "category": "自然风光", "rating": 4.9, "price": 0, "desc": "世界文化遗产，中国著名湖泊"},
    {"id": 3, "name": "张家界", "city": "张家界", "category": "自然风光", "rating": 4.7, "price": 248, "desc": "世界地质公园，奇峰怪石"},
    {"id": 4, "name": "九寨沟", "city": "阿坝", "category": "自然风光", "rating": 4.8, "price": 169, "desc": "人间仙境，彩色湖泊"},
    {"id": 5, "name": "黄山", "city": "黄山", "category": "自然风光", "rating": 4.7, "price": 230, "desc": "奇松、怪石、云海、温泉"},
    {"id": 6, "name": "丽江古城", "city": "丽江", "category": "历史文化", "rating": 4.6, "price": 0, "desc": "世界文化遗产，纳西族古镇"},
    {"id": 7, "name": "布达拉宫", "city": "拉萨", "category": "历史文化", "rating": 4.9, "price": 200, "desc": "世界屋脊上的宫殿"},
    {"id": 8, "name": "桂林山水", "city": "桂林", "category": "自然风光", "rating": 4.7, "price": 55, "desc": "山水甲天下"},
    {"id": 9, "name": "鼓浪屿", "city": "厦门", "category": "休闲度假", "rating": 4.6, "price": 0, "desc": "音乐之岛，浪漫海滨"},
    {"id": 10, "name": "峨眉山", "city": "乐山", "category": "自然风光", "rating": 4.7, "price": 160, "desc": "佛教名山，普贤菩萨道场"},
]

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>飞猪旅行推荐系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Microsoft YaHei", Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        
        .search-box { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .search-box input { width: 70%; padding: 15px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px; outline: none; }
        .search-box input:focus { border-color: #667eea; }
        .search-box button { padding: 15px 40px; font-size: 16px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; margin-left: 10px; }
        .search-box button:hover { background: #5568d3; }
        
        .user-info { background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .user-info h2 { color: #333; margin-bottom: 15px; }
        .user-stats { display: flex; gap: 30px; }
        .stat { text-align: center; }
        .stat-num { font-size: 2em; color: #667eea; font-weight: bold; }
        .stat-label { color: #666; }
        
        .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .spot-card { background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.2); transition: transform 0.3s; }
        .spot-card:hover { transform: translateY(-10px); }
        .spot-img { height: 180px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 3em; }
        .spot-info { padding: 20px; }
        .spot-name { font-size: 1.3em; color: #333; margin-bottom: 10px; }
        .spot-meta { display: flex; gap: 15px; margin-bottom: 10px; color: #666; font-size: 0.9em; }
        .spot-rating { color: #ff9800; }
        .spot-desc { color: #666; font-size: 0.9em; line-height: 1.5; }
        .spot-tags { margin-top: 10px; }
        .tag { display: inline-block; padding: 5px 12px; background: #f0f0f0; border-radius: 15px; font-size: 0.8em; color: #666; margin-right: 5px; }
        
        .empty-state { text-align: center; padding: 60px; color: white; }
        .empty-state h2 { font-size: 2em; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏝️ 飞猪旅行推荐系统</h1>
            <p>基于深度学习的智能旅游景点推荐</p>
        </div>
        
        <div class="search-box">
            <input type="text" id="userId" placeholder="请输入用户ID (如: 1001)" />
            <button onclick="getRecommendations()">获取推荐</button>
        </div>
        
        <div id="userInfo" class="user-info" style="display:none;">
            <h2>👤 用户画像</h2>
            <div class="user-stats">
                <div class="stat">
                    <div class="stat-num" id="userInteractions">0</div>
                    <div class="stat-label">历史交互</div>
                </div>
                <div class="stat">
                    <div class="stat-num" id="userCities">0</div>
                    <div class="stat-label">访问城市</div>
                </div>
                <div class="stat">
                    <div class="stat-num" id="recCount">10</div>
                    <div class="stat-label">推荐景点</div>
                </div>
            </div>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        const spots = {{ spots_json | safe }};
        
        function getRecommendations() {
            const userId = document.getElementById('userId').value || '1001';
            const resultsDiv = document.getElementById('results');
            const userInfoDiv = document.getElementById('userInfo');
            
            resultsDiv.innerHTML = '<div class="empty-state"><h2>🔍 搜索中...</h2></div>';
            
            // 模拟推荐
            setTimeout(() => {
                const recommended = spots.slice(0, 6);
                
                userInfoDiv.style.display = 'block';
                document.getElementById('userInteractions').textContent = Math.floor(Math.random() * 50) + 10;
                document.getElementById('userCities').textContent = Math.floor(Math.random() * 10) + 1;
                
                resultsDiv.innerHTML = recommended.map((spot, i) => `
                    <div class="spot-card">
                        <div class="spot-img">${['🏯','🏔️','🌊','🎭','🗿','🌸'][i % 6]}</div>
                        <div class="spot-info">
                            <div class="spot-name">${spot.name}</div>
                            <div class="spot-meta">
                                <span>📍 ${spot.city}</span>
                                <span class="spot-rating">⭐ ${spot.rating}</span>
                                <span>💰 ${spot.price === 0 ? '免费' : '¥' + spot.price}</span>
                            </div>
                            <div class="spot-desc">${spot.desc}</div>
                            <div class="spot-tags">
                                <span class="tag">${spot.category}</span>
                                <span class="tag">推荐指数: ${Math.floor(Math.random() * 20) + 80}%</span>
                            </div>
                        </div>
                    </div>
                `).join('');
            }, 500);
        }
        
        // 默认加载
        getRecommendations();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    import json
    return render_template_string(
        HTML_TEMPLATE,
        spots_json=json.dumps(SAMPLE_SPOTS, ensure_ascii=False)
    )

@app.route('/api/recommend')
def recommend():
    user_id = request.args.get('user_id', '1001')
    
    # 随机返回推荐结果
    recommended = random.sample(SAMPLE_SPOTS, min(6, len(SAMPLE_SPOTS)))
    
    return jsonify({
        'user_id': user_id,
        'recommendations': recommended,
        'count': len(recommended)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("飞猪旅行推荐系统 - 原型")
    print("=" * 60)
    print("启动服务: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)