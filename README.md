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
.venv/bin/python -m pytest tests -q                     # 248 项回归测试
.venv/bin/python -m ash.cli.main run --backend fake --delta 30 --max-steps 200
```

`--backend fake` 跑内置的确定性玩具平台关卡，整条 ASH 循环
（推理 → 检索 → 自举）无需游戏即可端到端跑通。

## 接真游戏

1. 启动游戏并开 CDP：`scripts/launch_nw_proton.sh`（机器相关路径全部可用环境变量覆盖，见脚本头注释）。
   见 `docs/game-architecture.md`）。
2. `ash doctor --backend cdp` 确认后端探活。
3. `ash run --backend cdp --num-agents 1 --delta 600`。
   卡住阈值 Δ=600 步 = 600 × `control_interval_s`(0.25 s) ≈ **150 秒游戏时间**。
4. 循环报告写在 `runs/ash/loop-report.json`，每轮自举的 checkpoint 在
   `runs/ash/bootstrap-NNN/`。

## 现状与证据链

**先读 [`docs/status.md`](docs/status.md)**：它把「已验证事实 / 已实测否证的假设 / 未决问题」
分开记录，每条都带测量数字与复现入口。当前一句话结论：

> 自进化循环的**管道全部打通并逐项真机验证过**（护栏、K、检索、裁剪、报告、录制）；
> 但**唯一的学习信号来源——IDM 对语料帧的伪标签——实测不可用**（跨场景准确率 45.5%，
> 多数类基线 50.4%，即等于瞎猜），所以 π 学不到「好行为」，表现为「尽量不动」。

**能力边界（2026-09-26 决定）：** 接受这条边界，停止学习信号方向的实验。
本项目**已交付、可复用的是「输入护栏 + 关键时刻记忆 K + 检索」这套框架**；
**「自我精进」那一段没有成立**——不要当作已实现，也不适合作为「ASH 复现成功」引用
（适合作为「ASH 的骨架 + 一个被测量过的失败点」引用）。决策记录与两条交接路径见
[`docs/status.md`](docs/status.md) 第五节。

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
| IDM 只由 agent 自己的转移更新（Alg 4） | 每轮再混入一批**录制的人类示范**转移（`--idm-replay`） | 实测：只用自己的转移时，一轮 220 条近似重复样本把预训练 IDM 从 `val 0.29` 打回 **`val 4.64`**（比均匀先验还差），伪标签 99.9–100% 单一类、π 从不更新；混入示范后同一轮 `val 0.28` |
| π 在 D^R 上训到收敛 | 每轮每视频上限 `--policy-steps`（默认 300） | 无上限时一个 9405 帧视频 = 3126 步、每步 8×64 帧编码，实测 **1.0 steps/s** → 近一小时，且 epoch 结束前没有任何日志。上界让一轮可迭代；实际步数写进报告的 `policy_steps` |

## 复现步骤

1. `ash run --backend fake ...` 先在玩具环境验证循环收敛行为；
2. 准备 `data/corpus/*.npz`（yt-dlp 下载速通视频 → 抽帧，见
   `docs/ai-architecture.md#语料`）；
3. **先在游戏里读档进入可操作的地图**，再 `ash run --backend cdp` 开始真机自博弈。
   在 MZ 里 `jump` 就是「确定」键，所以策略在标题/菜单画面上按一下跳跃就等于提交
   菜单选项；输入只允许在 `Scene_Map` 内派发，CLI 启动时会先查一次场景，不在游戏内
   直接拒绝启动（退出码 3）。**不要在标题画面或菜单里启动实机循环**——首次实机自举
   就是这样进入了玩家的存档。
4. **自己录一段人类操作**（可选，但这是让 IDM 变好的数据来源）：
   ```bash
   .venv/bin/ash record --out data/human-001.npz --fps 10 --size 128
   ```
   先在游戏里读档进到可操作状态，然后**开着终端直接玩**，玩完按 Ctrl-C。
   录制的按键状态是**在页面里读玩家自己的物理按键**（`keydown`/`keyup` 的 keyCode），
   与截图取自同一个事件循环，因此天然对齐——不需要旧项目那套 evdev + 录屏 + 时钟标定。
   一个键按在**静止画面**上会被降权（`weights`）而不是删除：运动分布是连续的，
   任何阈值都是任意切断，旧数据集也是这么处理的。结束后会打印帧数、按键直方图、
   以及「截图期间按键发生变化」的帧数（对齐误差），不会把这些藏起来。
5. **用带动作标签的示范预训练 IDM**（这一步是 π 能否被训练的前提）：
   ```bash
   uv run python scripts/pack_demos.py data/idm-human.npz   # npz → 可 mmap 的 .npy
   .venv/bin/ash pretrain-idm --demos data/idm-human.npz --out models/idm-demo.pt
   .venv/bin/ash run --backend cdp --corpus data/corpus \
       --idm models/idm-demo.pt --idm-replay data/idm-human.npz --policy-steps 200
   ```
   难度雕像的对话**关不掉**（只能选一个），所以必须有个预设答案：默认取
   `config/game.yaml:difficulty_preset`（**默认「简单」**——语料是速通录像，速通都在简单
   难度下跑，混着难度等于比较两个不同的游戏）。`--difficulty 普通`（选项文本，或从 1 数起的
   序号）可临时覆盖，`--difficulty off` 则回复成「拒绝并中止该轮」。其余普通对话选项
   仍由 agent 自己操作。
   没有 `--idm-replay`，每轮自博弈的近似重复转移会把 IDM 打回「预测类别先验」，
   伪标签退化成常数，π 永远不会更新（见上表）。实测带 replay 的一轮：
   `majority_share` 0.72–0.87（原先 0.999–1.000）、`classes_used` 7–10（原先 1–3）、
   `logit_temporal_std` 4.4–5.1（原先 0.09–0.12），4 个视频里 3 个通过守卫并真的更新了 π。
   自己录的文件同样走这两步（`pack_demos.py` → `pretrain-idm` / `--idm-replay`）。

6. **让录制也参与 K 的覆盖**：把录制文件（或它的 `pack_demos` 产物）放进
   `data/recordings/`，它会和 `data/corpus/` 一起构成 D^I（`--recordings` 可改路径）。
   K 是在 D^I 上拟合的，而抓来的语料是**速通录像**——没人会拍初始小屋或某个 NPC 房间。
   实测：`key_moments` 在每一轮都是 0，实机帧到最近语料帧的余弦只有 0.77–0.92，
   语料里**没有任何一帧**与实机帧相似度超过 0.95。录一段你实际待的房间，K 才能在那里响；
   K 一响，卡死计时器不再累积，回合长度会从 `delta + random_steps` 变成几千步。
   录制文件按 `frames` 或 `observations` 任一字段时间读入，两种写法都接受。
7. 每轮 `runs/ash/bootstrap-NNN/` 下的 `policy.pt` 即当前策略，
   随时可用 `ash eval runs/ash/loop-report.json` 查看进度。

**注意退出状态**：实机跑完游戏会被**刻意留在暂停态**（ticker 停住），目的是不让角色
在无人操作时被敌人打死——所以「跑完游戏像卡住了」是预期行为，不是 bug。恢复方法：
页面里执行 `__ash.pump.resume()`，或重启游戏。

## 状态

> 数字与证据的**唯一出处**是 [`docs/status.md`](docs/status.md)；这里只留终态判断。
> 曾经的逐条进度清单已删除——它与 `docs/status.md` 互相矛盾过（同一个 IDM 问题在两处
> 结论相反），两份状态文档的维护成本比它记录的信息更贵。

**已完成、真机验证过**：三层输入护栏（场景探针 / 后端硬闸 / runner 中止）；游戏异常
场景的自动处置（难度对话 / ESC 菜单最后一栏 / GAME OVER 只许读档 / 道具弹窗 / 菜单
cancel 逃生）；K（HDBSCAN + PCA-64 + 多轨迹过滤 + 缓存）；贪心一对一检索；语料裁剪；
边录边落盘的录制器；每轮在 bootstrap 前落盘的可观测报告；248 项回归测试。

**未完成、且经实测判定当前路线走不通**：π 的学习信号。IDM 对语料帧的伪标签退化为
「先验 ≈ 一半 noop」，π 用 BC 模仿它就学成尽量不动。根因是**跨场景不泛化**
（留出场景 45.5% vs 多数类基线 50.4%），三种架构都如此，第三种把训练集背到
loss 0.0000 而留出仍是 49.6% ⇒ **瓶颈是带标签数据太少太窄，不是架构**。
按 2026-09-26 的决策**停止在这条路上继续投入**；两条候选接手路径见
`docs/status.md` 第五节。

**因此当前可用的命令**：`ash doctor` / `ash record` / `ash assemble` / `ash pretrain-idm`
/ `ash run --backend fake`（玩具环境端到端跑通）。`ash run --backend cdp` 能跑、护栏
有效、报告完整，但**不要期待它变强**——它在语料覆盖不到的地方仍会退化成静止。
