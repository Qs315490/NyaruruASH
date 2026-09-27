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
4. 每轮 `runs/ash/bootstrap-NNN/` 下的 `policy.pt` 即当前策略，
   随时可用 `ash eval runs/ash/loop-report.json` 查看进度。

**注意退出状态**：实机跑完游戏会被**刻意留在暂停态**（ticker 停住），目的是不让角色
在无人操作时被敌人打死——所以「跑完游戏像卡住了」是预期行为，不是 bug。恢复方法：
页面里执行 `__ash.pump.resume()`，或重启游戏。

## 状态

- [x] 核心模块与回归测试（110 passed）
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
- [x] **实机驱动改为实时模式**：帧泵会改变游戏行为（同陷阱 A/B 实测：真实 ticker 会传送、
      帧泵 1200 帧钉在受伤态不传送），自博弈改用 `drive="realtime"`
- [ ] **实机操作已暂停（决定，2026-09-24）**：在「agent 的输入确实能驱动角色、
      且伤害/死亡/菜单等场景都有可靠处置」被证据验证之前，不再对游戏发任何输入。
      停止前的事故见 `docs/pitfalls.md` 第 14 条。
- [x] **实机驱动验证通过**（2026-09-24，用正确信号 `screenX()/screenY()` 观测）：
      `drive="realtime"` 下一轮 40 步，角色精灵累计移动 **3832 px**、位置变化 56 次，
      全程 `canMove=True` —— agent 的动作**确实在驱动角色**
      （此前用 `$gamePlayer.x/y` 观测得出「角色不动」的结论是**观测方法错误**：
      本游戏是动作平台游戏，位置记在**精灵**上，格子坐标 `x/y` 永不更新）
- [x] 语料嵌入缓存**真正被复用** + 加指纹失效（此前缓存路径默认 None，每次启动都重嵌整份语料）
- [x] **K 恒 0 的根因（已纠正过一次）**：当时跑的是 1800 帧的 smoke 子集 + 384 维聚类。
      实测三方对照（PCA-64）：
      | 语料 | `clusters_kept` |
      | --- | --- |
      | 3 视频 / 33157 帧 | **7** |
      | 6 视频 / 68440 帧 | **91** |
      | 6 视频 / `c_min=2` | 189 |
      **扩语料是主导因素（7→91）**；PCA 的作用是让拟合跑得完（384 维 6 分钟未完成 vs
      PCA-64 113 秒）。**注意**：3 视频并非「数学上不可能」满足 `c_min=3`（keep 了 7 个），
      只是非常苛刻——我先前写下的「不可能」是错的。
- [x] **语料扩充**：新增 3 个 B 站结局 3 速通录像，现为 **6 视频 / 68440 帧**（4 fps、256×256）
- [x] **K 不再为空**：6 视频语料 + PCA-64 拟合出 **91 个关键簇**（控制对照 3 视频仅 7 个）
- [x] **K 的拟合结果缓存**（`data/corpus-kdm.pkl`，`--kdm-cache` 可覆盖；已被 `.gitignore` 的
      `data/` 挡住，属可再生的派生产物）。实测全量 6 视频语料：
      **首次拟合 116.5 s → 命中缓存 0.0 s，随机 400 帧的簇标签 400/400 完全一致**。
      缓存带指纹（语料文件名+大小、嵌入分辨率、超参、hdbscan/scikit-learn 版本），
      指纹不符/文件损坏/版本不符一律当作「没有缓存」并重建——缓存只许是加速手段。
      另外：`embedder` **不进缓存**（它是调用方挂上去的，pickle torch 模块会让文件变大
      且把有效性绑死在 torch 版本上），加载后由调用方重新挂。
- [x] **修掉 `f0c5ad0` 引入的一处破坏性缺陷**：`bootstrap` 的关键时刻检测直接调用
      `approximate_predict(kdm._clusterer, 原始向量)`，而 clusterer 已改成活在 PCA 空间里，
      于是每个 bootstrap 轮都会抛 `ValueError: New points dimension does not match fit data!`
      ——即「跑完一轮推理、进 bootstrap 就崩」。现统一走 `kdm.cluster_of()`（内含同一投影），
      `tests/test_loop.py` 钉死。
- [ ] 用扩充后的语料跑实机：验证 `key_moments` 能否被真正触发（这一条还没验证）
- [x] **修掉「游戏永久冻结、且所有恢复路径都在撒谎」的缺陷**：帧泵把 `start()` 影子写在
      **ticker 实例**上，而重新注入 `agent.js` 会替换 `window.__ash`（并丢掉 `V`）——
      新 agent 手里没有 ticker 句柄，`unguard()` 静默无效，`resume()` 却返回
      `{resumed:true}`。实测复现：pump 安装 → 重新注入 → `resume()` 报成功，而
      `ticker.started` 仍为 `false`、`__ashGuarded` 仍为 `true`，游戏就此停死；
      `close(resume=True)` / `resume_game()` 同样谎报成功。修法：影子里的判断读**活的**
      `window.__ash`、`unguard(ticker)` 接受显式 ticker、`resume()` 用 `t.started`
      **核实**后才报成功。`tests/test_pump_guard.py` 钉死（三条里有一条在旧代码上必红）。
- [x] **全语料自举过去根本跑不起来（已修）**：`build_policy_dataset` 把训练窗口物化成
      `(n_win, w_s, H, W, 3)` float32 —— 一个 **9405 帧**的语料视频就是 **59 GB** 的
      `win_frames` + 14.7 GB 的 `win_mem`，而本机 16 GB。实测：全语料实机 run 的 bootstrap
      **18 分钟单核 100%、零产物**（此前一直被 600 帧的 `corpus-live` 子集掩盖）。
      改为 `WindowDataset`：只存一份 uint8 帧缓冲 + 每窗口索引，`batch(idx)` 现切。
      同一视频实测 **峰值 RSS 3.54 GB**（原需 73.7 GB 才会开始）。`ds["frames"]` /
      `ds["memories"]` 现在**显式抛 KeyError**——旧的静默分配才是真问题。
      `tests/test_lazy_windows.py` 钉死 batch 与 `np.stack` 的窗口**逐位相同**。
- [x] **K 批量查询**：`kdm.classify_sequence()` 一次 `approximate_predict` 覆盖整条轨迹，
      bootstrap 用它、runner 仍用单帧 `classify()`，`tests/test_kdm.py` 直接比对两条路径
      必须完全一致。**但只快 1.3×**（`approximate_predict` 本质是逐点 KD-tree 查询），
      实测干净负载 1.2 ms/帧 —— 早先报的 3.4 ms/帧是**在实机 run 抢 CPU 时测的**，已更正。
- [x] **每轮增量落盘**：报告原先只在 `run()` 返回后写一次，于是死在 bootstrap 里的 run
      把整轮推理数据全部抹掉。现在每轮在 **bootstrap 开始前**就写一次（未完成时带
      `bootstrap_pending: true`），原子替换。`tests/test_report_incremental.py` 钉死。
- [x] **菜单逃生（cancel-only）**：agent 会自己按进菜单（实测 `Scene_SkillSt`，一轮只跑了
      39 步就中止），而它永远出不来 → 整轮报废 + 需要人救。现在 `V.MENU_SCENES` +
      `escape_menu()` 只派发 cancel、只限清单场景、次数有上限，失败即中止该轮。
      **`MENU_SCENES` 与 `CONFIRM_SCENES` 是两份清单、不得有交集**（判据分别是
      「cancel 安不安全」与「ok 安不安全」）。实测 `Scene_SkillSt` 两次 cancel 回到 `Scene_Map`。
      另：中止还会连带丢掉该轮的随机探索（`random_transitions: 0`，因为中止后不再乱按）。
- [x] **动作空间加入战斗/物品四个动词**（16 → 20，**只追加**，原下标不变）：
      `special`(V 技能，耗 SP)、`ult`(A 武器大招，需黄条)、`weapon_switch`(S 换咸鱼武器)、
      `item`(F 就地用物品，唯一能即时回血的动作)。`docs/game-systems.md` 早已把 S/A 记为
      「不在动作空间里，可能是战斗能力受限的原因」；`F` 当年与 M/D 一起被有意排除，现按
      要求加入（D 仍排除——速通用它等于作弊）。
      **已验证「键到位」**（keydown 计数 + 引擎 `Input.keyMapper`：70→item / 83→cfish /
      86→subattack / 65→zxc）；**未验证「效果」**（当前 0 物品、不在战斗、
      `_equips` 为空即咸鱼武器非 MZ 标准装备系统）——效果需在战斗中另测。
- [x] **属性强化其实可逆**：商店的**猫退烧药**会「退回强化材料并重置强化」（洗点），
      所以此前「强化不可逆」的顾虑作废（但买药花钱本身仍不可逆）。
- [x] **帧泵时钟锚定（修掉一个真缺陷）**：泵原先合成时钟从 **0** 起算，而 ticker 的
      `lastTime` 是真实页面运行时间（实测 ~40000ms）。PIXI `Ticker.update()` 只在前进时才
      跑一帧，否则把 delta 全置 0 **且不通知 listener** → **第一泵帧被整个吞掉**，随后
      `lastTime = currentTime` 把时间轴重设到合成时钟上，卸载后引擎看到 ~40 秒跳变。
      现在 `anchorClock()` 在 `install()`/`resume()` 时把 `origin` 与 `ticker.lastTime`
      锚到真实时钟，喂 `origin + ticks*dt`：**dt 仍是固定 1/60 s**（确定性不丢），时间轴
      始终在真实时间轴内。实测首帧 `deltaMS=16.667`（旧为 0）、卸载后无跳变。
      **待验证**：这是否就是「后摇不结束 / 陷阱不传送」的病根——需在陷阱现场另测。
- [ ] IDM 伪标签仍退化（每轮仅 60 条转移）：需大幅提高 `--random-steps` 或更多轮次
- [ ] 实机长跑（`--max-bootstraps > 1`）：等 K / IDM 有信号后再谈
