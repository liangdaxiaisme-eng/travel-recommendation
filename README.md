# 🎯 基于深度学习的旅游景点推荐系统

> ✨ 这是一个使用深度学习实现的旅游景点推荐算法毕业设计项目，适用于本科毕业论文。

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.0-green?style=flat)

## 📚 项目简介

本项目实现了基于深度学习的旅游景点推荐算法，主要功能包括：

- 🏖️ **个性化推荐**：根据用户历史行为推荐景点
- 🔄 **多种算法**：实现了Item-CF、SVD、NeuMF、混合推荐等多种算法
- 📊 **Web系统**：提供可视化Web界面
- 🌐 **API接口**：支持RESTful API调用

## 📊 数据集信息

| 指标 | 数值 |
|------|------|
| 用户数 | 309,794 |
| 景点数 | 40,034 |
| 交互数 | 1,071,442 |

## 🛠️ 技术栈

- **后端**：Python 3.11, Flask, PyTorch
- **前端**：HTML + CSS + JavaScript
- **数据库**：CSV文件存储
- **算法**：协同过滤、神经协同过滤、知识图谱

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/旅游推荐系统.git
cd 旅游推荐系统
```

### 2. 安装依赖

```bash
# 进入训练代码目录
cd 训练代码

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 训练模型

```bash
# 运行混合推荐模型训练
python train_hybrid.py
```

### 4. 启动Web系统

```bash
cd ../web-system

# 安装Web依赖
pip install -r requirements.txt

# 启动服务
python recommendation_system.py
```

### 5. 访问系统

打开浏览器访问：`http://localhost:5000`

## 📁 项目结构

```
旅游推荐数据集/
├── 论文/                          # 毕业论文
│   └── 基于深度学习算法的旅游景点推荐算法研究.docx
│
├── 训练代码/                       # 模型训练代码
│   ├── train_item_cf.py          # Item-CF算法
│   ├── train_neumf.py            # 神经协同过滤
│   ├── train_hybrid.py           # 混合推荐
│   ├── preprocessing.py          # 数据预处理
│   └── requirements.txt          # Python依赖
│
├── web-system/                    # Web原型系统
│   ├── recommendation_system.py  # Flask后端
│   ├── requirements.txt
│   └── README.md
│
├── 实验结果/                       # 实验数据
│   ├── 模型对比.csv
│   ├── model_comparison.png      # 模型对比图
│   ├── user_distribution.png     # 用户分布图
│   └── city_distribution.png     # 城市分布图
│
└── 论文完善素材.md                 # 论文写作素材
```

## 📈 实验结果

| 模型 | Hit Rate@10 | NDCG@10 |
|------|-------------|---------|
| Item-CF | 0.0240 | 0.0152 |
| SVD | 0.0107 | 0.0068 |
| KNN | 0.0312 | 0.0198 |
| **混合推荐(本文)** | **0.0356** | **0.0224** |

## 💻 系统界面

系统提供以下功能：
- 输入用户ID获取推荐
- 景点搜索（按城市、类别）
- 推荐解释
- 用户反馈（点赞/不喜欢）

## 📝 论文说明

- **题目**：基于深度学习算法的旅游景点推荐算法研究
- **作者**：何幸芳
- **学号**：22460532
- **指导教师**：曾德真
- **学院**：大数据与人工智能学院

## 🔧 常见问题

### Q: 训练需要GPU吗？
A: 有GPU会更好，但没有也可以运行，只是速度较慢。

### Q: 数据从哪里来？
A: 数据来自飞猪旅行平台，经过预处理后使用。

### Q: 如何修改推荐算法参数？
A: 编辑 `train_hybrid.py` 中的参数配置。

## 📄 许可证

本项目仅供学习参考使用。

---

⭐ 如果对你有帮助，欢迎Star！