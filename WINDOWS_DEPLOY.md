# Windows 本地部署教程

## 一、检查 Python 是否已安装

打开 CMD（命令提示符）或 PowerShell，输入：

```cmd
python --version
```

如果显示类似 `Python 3.12.0` 说明已安装。

如果显示 `'python' 不是内部或外部命令`，说明没安装，请看下一步。

---

## 二、安装 Python

### 1. 下载 Python
访问：https://www.python.org/downloads/

点击 `Download Python 3.12.x`（或最新版本）

### 2. 安装 Python
运行下载的安装包，**务必勾选**：
- ✅ `Add Python to PATH`（添加到环境变量）
- ✅ `Install launcher for all users`（可选）

点击 `Install Now` 等待安装完成。

### 3. 验证安装
重新打开 CMD，输入：
```cmd
python --version
```
应该显示 `Python 3.12.x`

---

## 三、安装依赖库

在 CMD 中执行：

### 方法1：直接安装（推荐）
```cmd
pip install flask pandas numpy torch
```

### 方法2：国内镜像（如果卡顿）
```cmd
pip install flask pandas numpy torch -i https://mirrors.aliyun.com/pypi/simple/
```

### 方法3：PyTorch CPU 版本（如果安装失败）
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

等待安装完成，显示 `Successfully installed xxx` 即成功。

### 4. 验证依赖
```cmd
pip show flask pandas numpy torch
```
应该显示每个库的版本信息。

---

## 四、下载项目

### 方法1：下载 ZIP（简单）
1. 打开浏览器访问：https://github.com/liangdaxiaisme-eng/travel-recommendation
2. 点击绿色的 `Code` 按钮
3. 点击 `Download ZIP`
4. 把下载的 `travel-recommendation-main.zip` 解压到桌面
5. 文件夹重命名为 `feizhu`（可选）

### 方法2：使用 Git（需要安装 Git）
```cmd
git clone https://github.com/liangdaxiaisme-eng/travel-recommendation.git
```

---

## 五、运行系统

### 1. 进入项目目录
```cmd
cd C:\Users\你的用户名\Desktop\feizhu\web-system
```

（路径根据你解压的位置调整，可以用 Tab 自动补全）

### 2. 启动服务
```cmd
python recommendation_neumf_fixed.py
```

### 3. 等待启动
首次运行会显示：
```
==================================================
🚀 城市名称修复版 - 飞猪旅行推荐系统
==================================================
模型参数: NUM_USERS=63161, NUM_ITEMS=69525, NUM_KG_NODES=65
✅ 模型加载成功！

正在加载数据（快速模式）...
✓ 用户行为数据: 100,000 条

正在加载景点数据...
✓ 景点数据: 69,525 个

==================================================
🌐 服务地址: http://0.0.0.0:4009
   按 Ctrl+C 停止服务
==================================================
 * Running on http://0.0.0.0:4009 (Press CTRL+C to quit)
```

---

## 六、使用系统

1. 打开浏览器（Chrome/Edge 等）
2. 地址栏输入：**http://127.0.0.1:4009**
3. 会看到飞猪旅行推荐系统的网页界面

### 测试用户 ID
在输入框中输入以下 ID 测试：

| 用户ID | 历史记录 |
|--------|----------|
| 1000560 | 去过三亚（景点57443）|
| 1000620 | 去过青岛、桂林 |
| 1000100 | 去过三亚（景点190138）|
| 1000 | 随机推荐 |

输入后点击"调用推理引擎"按钮，系统会：
- 左侧显示：该用户的历史访问记录
- 右侧显示：NeuMF 模型推荐的景点
- 右下角：显示推荐算法的计算过程

---

## 七、常见问题

### Q1: 提示 'python' 不是内部或外部命令？
**A**: Python 没添加到 PATH。重新安装 Python，务必勾选 `Add Python to PATH`，或者手动添加：
- 右键 `此电脑` → `属性` → `高级系统设置` → `环境变量`
- 找到 `Path`，添加 Python 安装路径，例如：`C:\Users\用户名\AppData\Local\Programs\Python\Python312`

### Q2: pip 安装超时或卡住？
**A**: 使用国内镜像源：
```cmd
pip install flask pandas numpy torch -i https://mirrors.aliyun.com/pypi/simple/
```

### Q3: PyTorch 安装失败？
**A**: 使用 CPU 版本（不需要显卡）：
```cmd
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q4: 端口 4009 被占用？
**A**: 打开 `recommendation_neumf_fixed.py`，找到最后一行：
```python
app.run(host="0.0.0.0", port=4009, debug=False)
```
把 `4009` 改成其他端口如 `5000` 或 `8080`

### Q5: 内存不足？
**A**: 系统已优化为快速模式，只加载 10 万条数据，内存需求约 1GB。如果还是卡，关闭其他程序。

### Q6: 浏览器打不开？
**A**: 确保服务已启动（CMD 窗口显示 `Running on http://0.0.0.0:4009`），尝试访问：
- http://localhost:4009
- http://127.0.0.1:4009

---

## 八、停止服务

在 CMD 窗口按 `Ctrl + C` 停止服务。

---

## 九、卸载

如果不再使用，卸载 Python：

```cmd
pip uninstall flask pandas numpy torch
```

然后在 `控制面板` → `程序和功能` 中卸载 Python。

---

---

## 十一、Docker 部署（推荐）

如果觉得安装 Python 环境麻烦，可以使用 Docker 一键部署：

### 1. 安装 Docker Desktop
下载：https://www.docker.com/products/docker-desktop/

### 2. 启动 Docker Desktop
（首次启动可能需要等待几分钟）

### 3. 部署项目

打开 PowerShell：

```powershell
# 克隆项目
git clone https://github.com/liangdaxiaisme-eng/travel-recommendation.git
cd travel-recommendation

# 启动服务
docker-compose up -d
```

### 4. 验证运行
```powershell
docker ps
curl http://localhost:4009
```

### 5. 访问
浏览器打开：http://localhost:4009

---

## 十二、技术说明

- **后端框架**：Flask
- **AI 模型**：PyTorch NeuMF（神经协同过滤）
- **模型权重**：真实训练的模型（约 12MB）
- **数据规模**：31 万用户、7 万景点、107 万条行为记录
- **推荐算法**：GMF + MLP + 知识图谱融合

---

有问题随时问我！