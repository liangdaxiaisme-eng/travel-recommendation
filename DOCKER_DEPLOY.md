# 快速启动 Docker 部署

## 前置要求

- **Windows**: 安装 Docker Desktop：https://www.docker.com/products/docker-desktop/
- **Linux**: 安装 Docker 和 Docker Compose

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker
```

---

## 一键启动（Linux/Mac）

```bash
# 克隆项目
git clone https://github.com/liangdaxiaisme-eng/travel-recommendation.git
cd travel-recommendation

# 启动服务
docker-compose up -d
```

---

## 一键启动（Windows）

### 方法1：PowerShell

```powershell
# 克隆项目
git clone https://github.com/liangdaxiaisme-eng/travel-recommendation.git
cd travel-recommendation

# 启动服务
docker-compose up -d
```

### 方法2：Docker Desktop GUI
1. 下载项目 ZIP 并解压
2. 打开 Docker Desktop
3. 导入 `docker-compose.yml`
4. 点击启动

---

## 验证运行

```bash
# 查看容器状态
docker ps

# 查看日志
docker-compose logs -f

# 测试访问
curl http://localhost:4009
```

---

## 访问系统

打开浏览器：http://localhost:4009

---

## 常用命令

| 命令 | 说明 |
|------|------|
| `docker-compose up -d` | 启动服务（后台） |
| `docker-compose down` | 停止服务 |
| `docker-compose restart` | 重启服务 |
| `docker-compose logs -f` | 查看实时日志 |
| `docker-compose ps` | 查看服务状态 |
| `docker exec -it feizhu-recommendation bash` | 进入容器 |

---

## 常见问题

### Q: Docker 启动失败？
**A**: 
- Windows: 确保已启动 Docker Desktop
- Linux: 检查 Docker 服务状态 `sudo systemctl status docker`

### Q: 端口被占用？
**A**: 修改 `docker-compose.yml` 中的端口：
```yaml
ports:
  - "5000:4009"  # 改为你想要的端口
```

### Q: 模型文件不存在？
**A**: 确保 `web-system/neumf_weights_real.pt` 存在。如果缺失，需要先从 GitHub 下载：
```bash
git clone https://github.com/liangdaxiaisme-eng/travel-recommendation.git
```

### Q: 内存不足？
**A**: Docker 容器默认使用 1GB 内存，如需调整：
- Docker Desktop → Settings → Resources → Memory

---

## 生产环境部署（Linux 服务器）

```bash
# 1. 服务器上安装 Docker
curl -fsSL https://get.docker.com | sh

# 2. 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 3. 拉取项目
git clone https://github.com/liangdaxiaisme-eng/travel-recommendation.git
cd travel-recommendation

# 4. 后台启动
docker-compose up -d

# 5. 设置开机自启
sudo systemctl enable docker

# 6. 查看状态
docker-compose ps
curl http://localhost:4009
```

---

## 访问地址

- 本地：`http://localhost:4009`
- 服务器：`http://你的服务器IP:4009`

---

## 停止和卸载

```bash
# 停止服务
docker-compose down

# 停止并删除数据卷（慎用）
docker-compose down -v

# 卸载 Docker（Linux）
sudo apt-get remove docker.io docker-compose
```