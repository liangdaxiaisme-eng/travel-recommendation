# 快速开始 - 飞猪旅行推荐系统

## 🚀 3分钟快速部署

### 步骤1：解压文件

将 `飞猪旅行推荐系统_NeuMF版本.zip` 解压到任意位置

### 步骤2：安装依赖

```bash
cd 飞猪旅行推荐系统_NeuMF版本/
pip3 install flask pandas torch numpy
```

### 步骤3：启动服务

```bash
python3 recommendation_neumf_fixed.py
```

### 步骤4：访问系统

打开浏览器访问：**http://127.0.0.1:4009**

---

## 📖 详细说明

### 系统特点

✅ 真实PyTorch NeuMF深度学习模型
✅ 个性化推荐（基于用户历史）
✅ 实时计算展示（右侧终端显示NeuMF计算过程）
✅ 城市名称识别（三亚、海口、青岛、厦门、大连、南京等）
✅ 访问次数统计
✅ Top-8推荐

### 测试用户ID

- `1000560` - 历史记录：景点57443（三亚）- 4次浏览
- `1000620` - 历史记录：景点18680（青岛）- 3次浏览，景点97985（桂林）- 3次浏览
- `1000100` - 历史记录：景点190138（三亚）- 5次浏览

### 文件说明

- `recommendation_neumf_fixed.py` - 主程序
- `neumf_weights_real.pt` - 模型权重（13MB）
- `item_profile.csv` - 景点数据（4.5MB）
- `README.md` - 详细使用说明
- `DEPLOYMENT_GUIDE.md` - 部署指南
- `requirements.txt` - Python依赖清单

### 更多信息

- 详细使用说明：查看 README.md
- 部署指南：查看 DEPLOYMENT_GUIDE.md
- GitHub上传指南：查看 GITHUB_UPLOAD_GUIDE.md

---

**版本**：v1.0
**最后更新**：2026-04-12
**作者**：何幸芳
