# 可选的 MySQL 会话目录

项目默认使用 `data/users/<用户>/sessions.json` 保存会话名称、时间和统计。第一阶段可将**这部分元数据**改存 MySQL；PDF、FAISS 索引、聊天记录、任务快照及审批记录仍沿用当前存储。无需先安装 MySQL 才能开发或运行默认模式。

需要启用时，准备 MySQL 8 数据库并安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

先在 MySQL 中创建数据库及具有建表权限的应用账号。然后设置环境变量（密码里的特殊字符要进行 URL 编码）：

```bash
export PAPER_MYSQL_URL='mysql://USER:PASSWORD@127.0.0.1:3306/paper_rag'
export PAPER_SESSION_STORE=mysql
```

启动应用时会自动创建 `paper_sessions` 表。连接失败会直接报错，以免同一用户的数据分散到两个目录。未设置 `PAPER_SESSION_STORE` 时仍使用 JSON。

若已有本地会话，在**切换为 MySQL 之前**先执行导入：

```bash
PAPER_MYSQL_URL='mysql://USER:PASSWORD@127.0.0.1:3306/paper_rag' \
  python -m scripts.migrate_sessions_to_mysql
```

脚本只读取 `data/users/*/sessions.json`，不会删除本地文件；对已包含会话的 MySQL 用户拒绝覆盖。导入后再设置 `PAPER_SESSION_STORE=mysql` 并启动应用。当前还没有实际 MySQL 服务集成测试，切换生产数据前请备份数据库和 `data/`。
