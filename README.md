# NyaruruASH — ASH 自主精进智能体玩《咸鱼喵喵》

基于论文 **ASH: Agents that Self-Hone via Embodied Learning**
(arXiv [2605.14211](https://arxiv.org/abs/2605.14211)) 的完整重写。
环境是 RPG Maker MZ 游戏《咸鱼喵喵 Nyaruru Fishy Fight》，通过 CDP（Chrome
DevTools Protocol）驱动 nwjs 运行时。

ASH 与传统 BC+手工里程碑路线的根本区别：**不写奖励
函数、不写里程碑系统**。进度信号由 HDBSCAN 在互联网视频嵌入上聚类自动
发现；卡住的定义是「连续 Δ 步没有遇到新的关键时刻」；卡住后自动检索相似
网络视频、用 IDM 打伪标签、增量更新策略——无需专家标注，无需奖励塑形。

## 前置条件

- Linux + Wayland，Python `>=3.13,<3.14`，包管理用 **uv**
- （真机运行）nwjs SDK 0.64.1 win-x64 + Proton，游戏本体，CDP 端口 `:9222`
- （AMD GPU）ROCm 版 torch，运行时必须导出 `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- （检索语料）速通视频帧存放在 `data/corpus/*.npz`（键 `frames`，(T,H,W,C) uint8）

## 安装

```bash
cd NyaruruASH
export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
uv venv && uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# AMD GPU: uv pip install --python .venv/bin/python torch --index-url https://rocm.nightlies.amd.com # 或项目内记录的精确 ROCm 轮子
```

## 快速验证（不需要游戏）

```bash
.venv/bin/python -m ash.cli.main doctor                 # 环境自检
.venv/bin/python -m pytest tests -q                     # 14 项回归测试
.venv/bin/python -m ash.cli.main run --backend fake --delta 30 --max-steps 200
```

`--backend fake` 跑内置的确定性玩具平台关卡，整条 ASH 循环
（推理 → 检索 → 自举）无需游戏即可端到端跑通。

## 接真游戏

1. 启动游戏并开 CDP：`scripts/launch_nw_proton.sh`（机器相关路径全部可用环境变量覆盖，见脚本头注释）。
   见 `docs/game-architecture.md`）。
2. `ash doctor --backend cdp` 确认后端探活。
3. `ash run --backend cdp --num-agents 1 --delta 600`。
   卡住阈值 Δ=600 步约对应 30 秒游戏时间（20 fps）。
4. 循环报告写在 `runs/ash/loop-report.json`，每轮自举的 checkpoint 在
   `runs/ash/bootstrap-NNN/`。

## 项目布局

```
src/ash/
  actions/    动作空间与策略建议（沿用）
  env/        SpeedrunEnv 契约 + cdp/fake 后端（沿用，快照已带回归测试）
  memory/     DINOv2 嵌入 + HDBSCAN 关键时刻模型（新，ASH 核心）
  retrieval/  贪心一对一匹配检索（新，论文 Algorithm 3）
  models/     ASH 双记忆 causal transformer 策略 + IDM（新+沿用；
              注：仓库根的 `models/` 被 gitignore，只存 checkpoint，两者不是一回事）
  loop/       推理 runner、自举、编排器（新，论文 Algorithms 1/2/4）
  train/      IDM 训练（沿用）
  cli/        ash doctor / run / eval
tests/        回归测试；凡 MEMORY.md 记过坑的组件都有针对性测试
docs/         游戏架构、AI 架构、注意事项
AGENTS.md     项目工作文档入口
```

## 与论文的偏差（有意为之）

| 论文 | 本项目 | 原因 |
| --- | --- | --- |
| SigLIP 图像 tokenizer（冻结） | IMPALA-CNN tokenizer（随策略训练） | 单卡 12G 与游戏共享，SigLIP 常驻代价过高；只有 DINOv2 保持冻结（论文同样如此） |
| 28 层 causal transformer | 6 层、hidden 256 | 同上，规模缩小见 `docs/ai-architecture.md` |
| N 个并行 agent | 默认 1（可 `--num-agents` 调） | 单游戏进程、单 GPU |
| DINOv2 ViT-S/14 | 相同 | 无偏差 |

## 复现步骤

1. `ash run --backend fake ...` 先在玩具环境验证循环收敛行为；
2. 准备 `data/corpus/*.npz`（yt-dlp 下载速通视频 → 抽帧，见
   `docs/ai-architecture.md#语料`）；
3. `ash run --backend cdp` 开始真机自博弈；
4. 每轮 `runs/ash/bootstrap-NNN/` 下的 `policy.pt` 即当前策略，
   随时可用 `ash eval runs/ash/loop-report.json` 查看进度。

## 状态

- [x] 核心模块与回归测试（14 passed）
- [x] fake 后端端到端 dry run
- [x] CDP 后端联调（真机 PASS：注入、帧泵、快照/回滚往返、方法存活）
- [x] CDP 真机快照/回滚往返验证（PASS）
