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
- （AMD GPU）ROCm 版 torch；**不要导出 `HSA_OVERRIDE_GFX_VERSION`**——wheel 是
  `device-gfx1101`，与 RX 7700 XT 原生匹配，设成 `11.0.0` 会 `hipErrorInvalidKernelFile`
- （检索语料）速通视频帧存放在 `data/corpus/*.npz`（键 `frames`，(T,H,W,C) uint8）

## 安装

```bash
cd NyaruruASH
export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
uv venv

# 默认（CPU）：PyPI 的 torch CPU 轮子，任何机器都能装
uv sync --extra cpu --extra dev

# AMD GPU（本项目实测环境）：ROCm 夜间仓库按设备分发内核包。
# gfx1101 = RX 7700 XT；换卡就换 device tag（gfx1100 / gfx1102 / device-gfx110X-all）。
uv sync --extra rocm --extra dev

# NVIDIA GPU
uv sync --extra nvidia --extra dev
```

三个 GPU extra 互斥（`[tool.uv] conflicts`），一次只能装一个。
ROCm 版实测为 `torch 2.15.0a0+rocm10.2.0a20260924`（HIP 7.17）。

> GPU 运行时需要访问 `/dev/kfd` 与 `/dev/dri`。在容器/沙箱里若设备数报 0，
> 先确认这两个节点已透传，再谈 torch 安装问题。

## 快速验证（不需要游戏）

```bash
.venv/bin/python -m ash.cli.main doctor                 # 环境自检
.venv/bin/python -m pytest tests -q                     # 21 项回归测试
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
3. **先在游戏里读档进入可操作的地图**，再 `ash run --backend cdp` 开始真机自博弈。
   在 MZ 里 `jump` 就是「确定」键，所以策略在标题/菜单画面上按一下跳跃就等于提交
   菜单选项；输入只允许在 `Scene_Map` 内派发，CLI 启动时会先查一次场景，不在游戏内
   直接拒绝启动（退出码 3）。**不要在标题画面或菜单里启动实机循环**——首次实机自举
   就是这样进入了玩家的存档。
4. 每轮 `runs/ash/bootstrap-NNN/` 下的 `policy.pt` 即当前策略，
   随时可用 `ash eval runs/ash/loop-report.json` 查看进度。

**注意退出状态**：实机跑完游戏会被**刻意留在暂停态**（ticker 停住），目的是不让角色
在无人操作时被敌人打死——所以「跑完游戏像卡住了」是预期行为，不是 bug。恢复方法：
页面里执行 `__ash.pump.resume()`，或重启游戏。

## 状态

- [x] 核心模块与回归测试（73 passed）
- [x] 实机输入安全护栏（场景探针 + 后端硬闸 + runner 中止 + **确认场景白名单**，
      `tests/test_safety.py`）；真机只读验证通过：游戏内 `__ash.safety()` →
      `{"scene":"Scene_Map","inGameplay":true}`，过场 → `{"scene":"Scene_Transport","confirm":true}`
- [x] 退出保持暂停是刻意的（防无人操作时被打死），且退出时明确告知留在暂停态与恢复方法
- [x] 语料 3 个视频，按 0.25 s 控制间隔重抽（9405 + 10406 + 13346 帧 @256×256，
      共 33157 帧，基本全不重复）
- [x] 时间尺度统一到 `control_interval_s`（0.25 s）：agent 15 帧/动作、语料 4 fps，
      CLI 在不一致时拒绝启动（`tests/test_time_scale.py`）
- [x] 随机策略采样补充 IDM（论文 Alg 4 步骤 2，原文档承诺但未实现）
- [x] fake 后端端到端自举循环（`ash run --backend fake`，3 轮：检索命中 3 个语料
      视频、K 拟合出 1 个保留簇、IDM 与 π 均收敛、checkpoint 落盘）
- [x] CDP 后端联调（真机 PASS：注入、帧泵、快照/回滚往返、方法存活）
- [x] 语料构建（`scripts/build_corpus.py`，ffmpeg 抽帧，实测 1176 帧全不重复）
- [x] 检索实机验证（`scripts/verify_retrieval.py`，真机帧 vs 噪声对照 PASS）
- [x] **完整自举循环实机跑通**（一次性 smoke：`--delta 60 --max-bootstraps 1
      --image-size 64 --corpus data/corpus-live`，47 秒一轮）——推理 40 步判 stuck、
      检测到 1 个关键时刻、检索命中 3/3、K 保留 2 个簇、IDM 与 π 训练完成、
      `policy.pt`/`idm.pt` 落盘；安全护栏全程未误拦（`aborted: None`）
- [ ] **IDM 还没学会依赖输入（当前主要障碍）**。实测：IDM 逐帧 logits 的方差只有
      0.008，而类间偏置的平方是 0.13 —— **argmax 完全由偏置决定、与画面无关**；
      随机初始化的 IDM 在语料上就已经 97% 输出同一个类。因此伪标签是常数，π 只能
      学成「永远输出同一个动作」（`policy_val ~1e-6` 看起来像收敛）。
      每轮 140 个转移（40 agent + 100 随机）对 16 类分类问题远远不够 —— 梯度下降最快
      的降损方式就是把类别偏置拟合好然后停住。需要的是数据规模（更长 episode、
      更多随机步、更多轮）或 IDM 预训练，不是接线。
- [x] 已加护栏：伪标签退化（某一类 >90%）时**拒绝更新 π** 并报错，而不是静默地把它
      训成常量。已在两个真实退化 checkpoint 上验证会触发
      （`tests/test_pseudo_labels.py`）
- [ ] 实机长跑（`--max-bootstraps > 1`）：等 IDM 的伪标签有信号后再谈
