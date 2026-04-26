# 飞猪旅行推荐系统 - Windows 部署教程

> 作者：何幸芳

---

## 一、检查环境

打开 CMD（命令提示符），输入：

```cmd
python --version
```

- ✅ 显示 `Python 3.x.x` = 已安装
- ❌ 报错 = 没安装，见下一步

---

## 二、安装 Python（如果没装）

1. 下载：https://www.python.org/downloads/
2. 安装时**务必勾选** `Add Python to PATH`
3. 安装完成后重新打开 CMD，验证：
```cmd
python --version
```

---

## 三、安装依赖

在 CMD 中执行：

```cmd
pip install flask pandas numpy torch
```

**如果安装慢或失败**，用国内镜像：

```cmd
pip install flask pandas numpy torch -i https://mirrors.aliyun.com/pypi/simple/
```

**如果 PyTorch 还是失败**，用 CPU 版本：

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 四、下载项目

1. 打开：https://github.com/liangdaxiaisme-eng/travel-recommendation
2. 点击 `Code` → `Download ZIP`
3. 解压到桌面，文件夹名叫 `travel-recommendation-main`
4. 进入文件夹 → 进入 `web-system` 文件夹

---

## 五、检查依赖

在 `web-system` 文件夹的地址栏输入 `cmd` 打开 CMD：

```cmd
python check_deps.py
```

看到 `🎉 所有依赖已安装！` 就可以下一步了。

---

## 六、启动系统

```cmd
python recommendation_neumf_fixed.py
```

等待显示：
```
🚀 城市名称修复版 - 飞猪旅行推荐系统
✅ 模型加载成功！
 * Running on http://0.0.0.0:4009
```

---

## 七、使用

浏览器打开：**http://127.0.0.1:4009**

测试用户 ID：`1000`、`1000560`、`1000620`、`1000100`

---

## 常见问题

| 问题 | 解决 |
|------|------|
| python 不是命令 | 重新安装 Python，勾选 Add to PATH |
| pip 找不到 | CMD 输入 `python -m pip install --upgrade pip` |
| PyTorch 安装失败 | 用 CPU 版本：`pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| 端口被占用 | 改端口：编辑代码把 `port=4009` 改成 `port=5000` |
| 内存不够 | 关闭其他程序，系统已优化为轻量模式 |

---

## 停止服务

CMD 窗口按 `Ctrl + C`

---

## 技术栈

- Flask + PyTorch NeuMF
- 31万用户、7万景点、107万条行为数据
- 真实训练的推荐模型

---

有问题请提交 Issue：https://github.com/liangdaxiaisme-eng/travel-recommendation/issues