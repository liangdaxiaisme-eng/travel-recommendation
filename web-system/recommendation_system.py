#!/usr/bin/env python3
from flask import Flask, render_template_string
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>景点推荐系统</title>
    <style>
        body { font-family: 微软雅黑, 宋体, Arial; background: #f0f0f0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: #fff; border: 1px solid #ccc; padding: 20px; }
        h1 { text-align: center; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        .tips { background: #ffffcc; border: 1px solid #ffcc00; padding: 15px; margin: 15px 0; font-size: 14px; }
        .tips h3 { margin: 0 0 10px 0; color: #666; }
        .tips ol { margin: 0; padding-left: 20px; }
        .tips li { margin: 5px 0; color: #333; }
        .search { text-align: center; margin: 20px 0; }
        .search input { padding: 8px; width: 150px; border: 1px solid #999; }
        .search button { padding: 8px 20px; background: #333; color: #fff; border: none; cursor: pointer; }
        .section { margin: 20px 0; }
        .section h3 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .spot-list { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .spot { border: 1px solid #ddd; padding: 10px; font-size: 13px; }
        .spot-name { color: #0066cc; font-weight: bold; }
        .spot-info { color: #666; margin-top: 3px; }
        .rec-spot { background: #e6f7ff; border-color: #91d5ff; }
        .empty { color: #999; font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>景点推荐系统</h1>
        
        <div class="tips">
            <h3>使用说明：</h3>
            <ol>
                <li>在下方输入框中输入任意用户ID（如：1001）</li>
                <li>点击"获取推荐"按钮</li>
                <li>系统将显示您的历史访问景点，并推荐6个新景点</li>
            </ol>
        </div>
        
        <div class="search">
            <input type="text" id="userId" placeholder="请输入用户ID">
            <button onclick="getRecommendations()">获取推荐</button>
        </div>
        
        <!-- 历史访问 -->
        <div class="section" id="historySection" style="display:none;">
            <h3>📍 您历史访问的景点</h3>
            <div class="spot-list" id="historyList"></div>
        </div>
        
        <!-- 推荐结果 -->
        <div class="section" id="recSection" style="display:none;">
            <h3>✨ 为您推荐的景点</h3>
            <div class="spot-list" id="recList"></div>
        </div>
    </div>

    <script>
    // 全部景点数据
    var allSpots = [
        {name:"故宫",city:"北京",rating:4.8,price:60,desc:"世界文化遗产，中国古代皇宫"},
        {name:"西湖",city:"杭州",rating:4.9,price:0,desc:"世界文化遗产，中国著名湖泊"},
        {name:"张家界",city:"张家界",rating:4.7,price:248,desc:"世界地质公园，奇峰怪石"},
        {name:"九寨沟",city:"阿坝",rating:4.8,price:169,desc:"人间仙境，彩色湖泊"},
        {name:"黄山",city:"黄山",rating:4.7,price:230,desc:"奇松、怪石、云海、温泉"},
        {name:"丽江古城",city:"丽江",rating:4.6,price:0,desc:"世界文化遗产，纳西族古镇"},
        {name:"布达拉宫",city:"拉萨",rating:4.9,price:200,desc:"世界屋脊上的宫殿"},
        {name:"桂林山水",city:"桂林",rating:4.7,price:55,desc:"山水甲天下"},
        {name:"鼓浪屿",city:"厦门",rating:4.6,price:0,desc:"音乐之岛，浪漫海滨"},
        {name:"峨眉山",city:"乐山",rating:4.7,price:160,desc:"佛教名山，普贤菩萨道场"},
        {name:"泰山",city:"泰安",rating:4.7,price:115,desc:"五岳之首，登泰山而小天下"},
        {name:"凤凰古城",city:"湘西",rating:4.5,price:0,desc:"中国最美小城"},
        {name:"长白山",city:"吉林",rating:4.8,price:199,desc:"关东第一山"},
        {name:"乌镇",city:"嘉兴",rating:4.6,price:0,desc:"中国最后的枕水人家"},
        {name:"平遥古城",city:"晋中",rating:4.6,price:125,desc:"保存最完整的古城之一"}
    ];
    
    // 打乱数组顺序（洗牌算法）
    function shuffle(array) {
        var arr = array.slice();
        for(var i=arr.length-1; i>0; i--){
            var j = Math.floor(Math.random()*(i+1));
            var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
        return arr;
    }
    
    // 生成随机历史记录（2-4个景点）
    function getRandomHistory(){
        var shuffled = shuffle(allSpots);
        var count = 2 + Math.floor(Math.random()*3); // 2-4个
        return shuffled.slice(0, count);
    }
    
    // 渲染景点卡片
    function renderSpots(spots, elementId, isRecommend){
        var html = "";
        for(var i=0; i<spots.length; i++){
            var s = spots[i];
            var cls = isRecommend ? "spot rec-spot" : "spot";
            html += '<div class="'+cls+'">';
            html += '<div class="spot-name">'+s.name+'</div>';
            html += '<div class="spot-info">'+s.city+' | 评分:'+s.rating;
            html += ' | '+(s.price>0?s.price+'元':'免费')+'</div>';
            html += '<div class="spot-info" style="margin-top:5px;">'+s.desc+'</div>';
            html += '</div>';
        }
        document.getElementById(elementId).innerHTML = html;
    }
    
    function getRecommendations(){
        var uid = document.getElementById("userId").value;
        if(!uid){alert("请输入用户ID");return;}
        
        // 获取随机历史记录（本次新随机）
        var history = getRandomHistory();
        
        // 从剩余景点中随机推荐6个
        var historyNames = history.map(function(s){return s.name;});
        var remaining = allSpots.filter(function(s){return historyNames.indexOf(s.name)<0;});
        var shuffled = shuffle(remaining);
        var recommendations = shuffled.slice(0, 6);
        
        // 显示历史
        document.getElementById("historySection").style.display = "block";
        renderSpots(history, "historyList", false);
        
        // 显示推荐
        document.getElementById("recSection").style.display = "block";
        renderSpots(recommendations, "recList", true);
    }
    </script>
</body>
</html>"""

@app.route("/")
def home():
    return HTML

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4009, debug=False)
