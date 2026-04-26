# 部署指南 - 飞猪旅行推荐系统

## 📋 目录

1. [系统要求](#系统要求)
2. [本地部署](#本地部署)
3. [远程服务器部署](#远程服务器部署)
4. [数据文件说明](#数据文件说明)
5. [常见问题](#常见问题)

---

## 系统要求

### 基础要求

- **操作系统**：Linux / macOS / Windows (WSL)
- **Python版本**：Python 3.7+
- **内存**：至少 4GB RAM
- **硬盘**：至少 1GB 可用空间

### Python依赖

```bash
flask>=2.0.0
pandas>=1.3.0
torch>=1.9.0
numpy>=1.19.0
```

### 安装Python

#### Linux / macOS

```bash
# 检查Python版本
python3 --version

# 如果版本低于3.7，先安装Python 3.7+
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3-pip

# macOS
brew install python@3.10
```

#### Windows

1. 下载Python 3.10：https://www.python.org/downloads/
2. 安装时勾选 "Add Python to PATH"
3. 验证安装：
   ```cmd
   python --version
   pip --version
   ```

---

## 本地部署

### 步骤1：解压文件

将 `飞猪旅行推荐系统_NeuMF版本.zip` 解压到任意位置：

```
飞猪旅行推荐系统_NeuMF版本/
├── recommendation_neumf_fixed.py
├── neumf_weights_real.pt
├── item_profile.csv
├── README.md
├── requirements.txt
└── DEPLOYMENT_GUIDE.md
```

### 步骤2：安装依赖

```bash
# 进入项目目录
cd 飞猪旅行推荐系统_NeuMF版本/

# 安装Python依赖
pip3 install -r requirements.txt
```

### 步骤3：启动服务

```bash
# 启动推荐系统
python3 recommendation_neumf_fixed.py
```

### 步骤4：访问系统

启动成功后，会看到：

```
==================================================
✅ 城市名称修复版 - 基于真实 PyTorch 权重的推荐系统启动成功！
👉 请用浏览器访问: http://127.0.0.1:4009
==================================================
```

在浏览器中打开：**http://127.0.0.1:4009**

### 步骤5：测试系统

1. 在输入框中输入用户ID：`1000560`
2. 点击"**调用推理引擎**"按钮
3. 查看推荐结果和NeuMF计算过程

---

## 远程服务器部署

### 方式一：使用SCP上传

#### 1. 准备本地文件

将整个 `飞猪旅行推荐系统_NeuMF版本/` 文件夹压缩：

```bash
# 在本地执行
cd ~/Downloads/
zip -r 飞猪旅行推荐系统_NeuMF版本.zip 飞猪旅行推荐系统_NeuMF版本/
```

#### 2. 上传到服务器

```bash
# 使用scp上传
scp -r 飞猪旅行推荐系统_NeuMF版本.zip adminm@10.151.165.67:~/飞猪推荐/
```

#### 3. 登录服务器并解压

```bash
# SSH登录服务器
ssh adminm@10.151.165.67

# 解压文件
cd ~/飞猪推荐/
unzip 飞猪旅行推荐系统_NeuMF版本.zip

# 进入项目目录
cd 飞猪旅行推荐系统_NeuMF版本/
```

#### 4. 安装依赖

```bash
# 安装Python依赖
pip3 install -r requirements.txt
```

#### 5. 启动服务

```bash
# 启动推荐系统
python3 recommendation_neumf_fixed.py
```

#### 6. 访问系统

在浏览器中打开：**http://10.151.165.67:4009**

### 方式二：使用FTP/SFTP工具

#### 推荐工具
- **FileZilla**（Windows/macOS/Linux）
- **WinSCP**（Windows）
- **Cyberduck**（macOS）

#### 上传步骤

1. 打开FileZilla
2. 连接服务器：
   - 主机：`10.151.165.67`
   - 用户名：`adminm`
   - 密码：`992557`
3. 上传 `飞猪旅行推荐系统_NeuMF版本.zip` 到服务器
4. 在服务器上解压：
   ```bash
   cd ~/飞猪推荐/
   unzip 飞猪旅行推荐系统_NeuMF版本.zip
   cd 飞猪旅行推荐系统_NeuMF版本/
   ```
5. 安装依赖并启动服务（同方式一）

---

## 数据文件说明

### 必需文件

| 文件名 | 大小 | 说明 | 是否必需 |
|--------|------|------|----------|
| recommendation_neumf_fixed.py | 15KB | 主程序 | ✅ 必需 |
| neumf_weights_real.pt | 13MB | NeuMF模型权重 | ✅ 必需 |
| item_profile.csv | 4.5MB | 景点数据 | ✅ 必需 |

### 可选文件

| 文件名 | 大小 | 说明 | 是否必需 |
|--------|------|------|----------|
| user_item_behavior_history.csv | 775MB | 用户行为数据 | ⚠️ 可选 |
| user_profile.csv | 168MB | 用户画像数据 | ⚠️ 可选 |

### 可选文件的作用

**user_item_behavior_history.csv**
- 包含1,071,442条真实用户行为记录
- 用于生成更准确的推荐结果
- 如果没有这个文件，系统会使用模拟数据
- **建议：新手不需要下载，系统可以直接使用**

**user_profile.csv**
- 包含310,000+用户的画像信息
- 用于生成更准确的推荐结果
- 如果没有这个文件，系统会使用模拟数据
- **建议：新手不需要下载，系统可以直接使用**

### 如何下载可选文件

如果需要下载可选文件：

1. 从原始数据集获取（需要机构权限）
2. 或者使用模拟数据（系统默认提供）

---

## 常见问题

### Q1: 启动失败 - Python版本过低

**错误信息**：
```
ModuleNotFoundError: No module named 'typing_extensions'
```

**解决方案**：
```bash
# 升级Python到3.7+
python3.10 --version

# 或者安装缺失的模块
pip3 install typing-extensions
```

### Q2: 启动失败 - 权重文件找不到

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'neumf_weights_real.pt'
```

**解决方案**：
1. 检查文件名是否正确
2. 确保所有文件在同一目录下
3. 重新解压压缩包

### Q3: 端口被占用

**错误信息**：
```
OSError: [Errno 98] Address already in use
```

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i:4009

# 杀掉进程
kill -9 <PID>

# 或者使用其他端口
python3 recommendation_neumf_fixed.py --port 8080
```

### Q4: 模块找不到

**错误信息**：
```
ModuleNotFoundError: No module named 'flask'
```

**解决方案**：
```bash
# 安装缺失的模块
pip3 install flask pandas torch numpy
```

### Q5: 系统启动慢

**现象**：系统启动需要3-5分钟

**原因**：系统正在加载10万条数据

**解决方案**：
1. 耐心等待（首次启动正常）
2. 减少数据加载量（修改代码中的 `nrows` 参数）
3. 使用 SSD 存储
4. 增加服务器内存

### Q6: 推荐结果不准确

**可能原因**：
1. 用户ID不存在
2. 用户没有历史记录
3. 数据量太少

**解决方案**：
1. 使用测试用户ID（1000560、1000620、1000100）
2. 确保用户有历史记录
3. 增加数据加载量

### Q7: 浏览器无法访问

**检查清单**：
1. 确认服务已启动（看到 "Running on http://127.0.0.1:4009"）
2. 检查防火墙设置
3. 尝试使用其他浏览器
4. 尝试使用IP地址访问：http://10.151.165.67:4009

### Q8: 如何停止服务

**方法1**：在终端中按 `Ctrl + C`

**方法2**：查找进程并杀掉

```bash
# 查找进程
ps aux | grep recommendation_neumf_fixed.py

# 杀掉进程
kill -9 <PID>
```

### Q9: 如何重启服务

```bash
# 停止服务
Ctrl + C

# 重新启动
python3 recommendation_neumf_fixed.py
```

### Q10: 如何查看运行日志

```bash
# 重定向输出到日志文件
nohup python3 recommendation_neumf_fixed.py > neu_mf_log.txt 2>&1 &

# 查看日志
tail -f neu_mf_log.txt
```

---

## 测试用户ID

系统内置了一些测试用户ID：

| 用户ID | 历史记录 | 测试场景 |
|--------|----------|----------|
| 1000560 | 景点57443（三亚）- 4次浏览 | 测试三亚用户 |
| 1000620 | 景点18680（青岛）- 3次浏览，景点97985（桂林）- 3次浏览 | 测试多城市用户 |
| 1000100 | 景点190138（三亚）- 5次浏览 | 测试高频用户 |

---

## 性能优化建议

### 1. 减少数据加载量

修改 `recommendation_neumf_fixed.py` 中的 `nrows` 参数：

```python
# 减少到1万条
behavior = pd.read_csv(f"{DATA_DIR}/user_item_behavior_history.csv",
                       header=None,
                       names=['user_id', 'item_id', 'action', 'timestamp'],
                       nrows=10000)
```

### 2. 使用生产级服务器

将Flask改为使用生产级WSGI服务器：

```bash
# 安装gunicorn
pip3 install gunicorn

# 启动服务
gunicorn -w 4 -b 0.0.0.0:4009 recommendation_neumf_fixed:app
```

### 3. 增加服务器内存

如果使用远程服务器，建议至少8GB内存

### 4. 使用SSD存储

使用SSD存储可以显著加快数据加载速度

---

## 联系支持

如果遇到问题：

1. 查看 README.md 中的使用说明
2. 查看本文档的常见问题部分
3. 查看系统日志文件（neu_mf_log.txt）
4. 联系作者获取帮助

---

**版本**：v1.0
**最后更新**：2026-04-12
**作者**：何幸芳
