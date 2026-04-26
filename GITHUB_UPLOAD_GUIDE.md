# 飞猪旅行推荐系统 - GitHub 仓库

## 📌 重要提示

**本仓库包含真实用户数据，请勿公开发布！**

- ✅ 可以公开：代码、模型权重、项目说明、使用文档
- ❌ 不可公开：真实用户数据（user_item_behavior_history.csv 等）
- ⚠️ 使用前请删除或脱敏真实用户数据

## 🚀 如何上传到 GitHub

### 步骤 1：创建 GitHub 仓库

1. 登录 GitHub：https://github.com
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name：`feizhu-travel-recommendation`
   - Description：基于 NeuMF 的旅游景点推荐系统
   - Public/Private：选择 Public（如果数据已脱敏）
4. 点击 "Create repository"

### 步骤 2：删除敏感数据

**重要**：在上传前，必须删除或脱敏真实用户数据！

```bash
# 删除或重命名敏感文件
mv user_item_behavior_history.csv user_item_behavior_history.csv.backup
mv user_profile.csv user_profile.csv.backup

# 保留：
# - recommendation_v6_pure.py
# - recommendation_neumf_fixed.py
# - neumf_weights_real.pt
# - item_profile.csv（这个可以保留，因为不包含用户信息）
# - README.md
```

### 步骤 3：初始化 Git 仓库

```bash
# 在项目目录下执行
cd /home/asd/论文资料/4/旅游推荐数据集/

# 初始化 Git
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: 飞猪旅行推荐系统"
```

### 步骤 4：连接 GitHub

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/feizhu-travel-recommendation.git

# 查看远程仓库
git remote -v
```

### 步骤 5：推送到 GitHub

```bash
# 推送到主分支
git branch -M main
git push -u origin main
```

### 步骤 6：验证上传成功

1. 打开浏览器访问：https://github.com/YOUR_USERNAME/feizhu-travel-recommendation
2. 应该能看到仓库中的文件

## 📦 上传的文件清单

### ✅ 可以上传的文件

- `README.md` - 项目说明文档
- `recommendation_v6_pure.py` - 主程序
- `recommendation_neumf_fixed.py` - NeuMF版本
- `neumf_weights_real.pt` - 模型权重（13MB）
- `item_profile.csv` - 景点数据（可以保留）

### ❌ 不可上传的文件

- `user_item_behavior_history.csv` - 真实用户行为数据（1.47GB）
- `user_profile.csv` - 真实用户画像数据（161MB）

## 🔒 数据脱敏方案

### 方案 1：删除敏感文件

```bash
# 直接删除
rm user_item_behavior_history.csv
rm user_profile.csv
```

### 方案 2：脱敏处理

```bash
# 创建脱敏脚本
cat > desensitize.py << 'EOF'
import pandas as pd

# 脱敏用户行为数据
df = pd.read_csv('user_item_behavior_history.csv')
df['user_id'] = df['user_id'] % 10000  # 保留最后4位
df['item_id'] = df['item_id'] % 10000  # 保留最后4位
df.to_csv('user_item_behavior_history_desensitized.csv', index=False)

# 脱敏用户画像数据
df = pd.read_csv('user_profile.csv')
df['user_id'] = df['user_id'] % 10000  # 保留最后4位
df.to_csv('user_profile_desensitized.csv', index=False)

print("脱敏完成！")
EOF

python3 desensitize.py

# 上传脱敏后的文件
git add user_item_behavior_history_desensitized.csv
git add user_profile_desensitized.csv
```

## 📝 GitHub 仓库内容

上传后，你的 GitHub 仓库应该包含：

```
feizhu-travel-recommendation/
├── README.md                          # 项目说明（已上传）
├── recommendation_v6_pure.py          # 主程序（已上传）
├── recommendation_neumf_fixed.py      # NeuMF版本（已上传）
├── neumf_weights_real.pt              # 模型权重（13MB，已上传）
├── item_profile.csv                   # 景点数据（已上传）
├── user_item_behavior_history.csv     # （已删除或脱敏）
└── user_profile.csv                   # （已删除或脱敏）
```

## 🎯 使用说明（小白版）

### 对于初学者

1. **克隆仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/feizhu-travel-recommendation.git
   cd feizhu-travel-recommendation
   ```

2. **安装依赖**
   ```bash
   pip3 install flask pandas torch numpy
   ```

3. **启动服务**
   ```bash
   python3 recommendation_v6_pure.py
   ```

4. **访问系统**
   - 本地：http://127.0.0.1:4009
   - 远程：http://10.151.165.67:4009

### 对于开发者

1. **查看代码**
   - 阅读 `recommendation_v6_pure.py` 了解主程序逻辑
   - 阅读 `recommendation_neumf_fixed.py` 了解 NeuMF 模型

2. **修改配置**
   - 修改 `cities` 字典添加新城市
   - 修改 `recommend_items` 函数调整推荐逻辑

3. **测试系统**
   - 使用测试用户ID：1000560、1000620
   - 查看推荐结果和 NeuMF 计算过程

## 📊 项目亮点

### 技术亮点

1. **真实 PyTorch 模型** - 使用 NeuMF 深度学习模型
2. **真实用户数据** - 基于真实的用户行为历史
3. **实时计算展示** - 右侧终端显示 NeuMF 计算过程
4. **城市名称识别** - 自动识别景点所在城市
5. **访问次数统计** - 显示每个景点的历史访问次数

### 学术价值

1. **毕业设计** - 适合作为毕业设计项目
2. **课程项目** - 适合推荐系统、深度学习课程
3. **数据展示** - 展示真实数据集的使用方法
4. **模型部署** - 展示深度学习模型的部署流程

## 🎓 适用场景

- ✅ 毕业设计答辩演示
- ✅ 推荐系统课程项目
- ✅ 深度学习实践
- ✅ 数据分析学习
- ✅ 系统架构学习

## ⚠️ 注意事项

### 数据隐私

- **绝对不要**上传真实用户数据
- 上传前必须删除或脱敏
- 如果数据已脱敏，可以公开

### 代码版权

- 本项目仅供学习和研究使用
- 请勿用于商业用途
- 如需商用，请联系作者

### 模型权重

- `neumf_weights_real.pt` 是训练得到的权重
- 可以公开，但请说明是训练得到的
- 不要修改权重文件

## 📞 联系方式

- GitHub Issues：https://github.com/YOUR_USERNAME/feizhu-travel-recommendation/issues

## 📄 许可证

MIT License - 仅供学习和研究使用

---

**版本**：v1.0
**最后更新**：2026-04-12
**作者**：何幸芳
