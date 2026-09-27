# AGENTS.md — 项目工作文档

本仓库是 ASH（arXiv 2605.14211）方法论在《咸鱼喵喵》(Nyaruru Fishy Fight) 上的实现。

## 铁律

1. **不要引入里程碑系统或奖励函数**。进度信号必须来自 HDBSCAN 自动发现的
   关键时刻（`memory/kdm.py`），卡住 = 连续 Δ 步无新关键时刻。手工列进度节点、
   手写价值函数都与 ASH 方法论冲突，不得加入。
2. **凡修改下列组件，必须跑对应回归测试**（每个组件的测试都钉死了具体失败模式）：
   - 快照/回滚（`env/cdp_backend.py`、`memory/js_source.py`）→ 5 个已知致命缺陷：
     编码器预注册吞内容、单段路径赋值错、函数占位符覆盖真函数、整数键排序错位、
     删除循环误删运行时方法。详见 `docs/game-architecture.md`。
   - 策略接线（`src/ash/` 下的 `models/`、`loop/`）→ 静默失败类：logit 下标错位、state 维度不匹配、
     图像尺寸不匹配、异常被吞退回噪声。`tests/test_loop.py` 钉死训练目标形状。
   - 检索匹配（`retrieval/matching.py`）→ 贪心一对一必须与论文循环一致
     （`tests/test_matching.py` 钉死）。
3. **不确定就停下确认**，不要臆测。
4. **环境**：uv 管依赖，`export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python`；
   GPU 任务必须 `HSA_OVERRIDE_GFX_VERSION=11.0.0`；256 分辨率训练用
   `--batch-size 16 --accum-steps 2`（游戏与训练共享 12G 显存）。
5. **重启游戏**需 `rm -rf .nw-profile/SingletonLock`（`.nw-profile/` 是本地运行时目录，不在仓库里）；绝不要 `pkill -f 'nw.exe'`
   （宽匹配会误杀进程）。

## 架构速览

- 推理循环：`loop/runner.py`（论文 Algorithm 2）——双记忆（短期 w_s 对 + 长期 w_l 个
  关键时刻）、卡死计时器、逐 4 帧嵌入检测。
- 自举：`loop/bootstrap.py`（Algorithm 4）——先更新 K，再在自博弈轨迹上更新 IDM，
  最后用 IDM 伪标签 + K 记忆更新 π。
- 编排：`loop/orchestrator.py`（Algorithm 1）——infer → retrieve → bootstrap 循环。
- 关键时刻：`memory/kdm.py`（HDBSCAN，多轨迹过滤）+ `memory/embeddings.py`（DINOv2）。
- 检索：`retrieval/matching.py`（贪心一对一窗口匹配）。

## 与论文的有意偏差

见 `README.md` 的偏差表与 `docs/ai-architecture.md`。改动偏差前先登记并确认。

## 测试

```bash
# 克隆后先建环境（`data/`、`runs/`、`.venv/` 等都不在仓库里）
export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
uv sync
.venv/bin/python -m pytest tests -q
```

14 项全过才算改动成立。`tests/test_loop.py` 是 fake 后端端到端 smoke。
