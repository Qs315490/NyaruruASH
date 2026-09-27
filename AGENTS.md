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
   **不要设 `HSA_OVERRIDE_GFX_VERSION`**。本机是 RX 7700 XT（`gcnArchName=gfx1101`，12G），
   装的 torch wheel 是 `device-gfx1101`，原生检测即可。实测设 `11.0.0`/`11.0.2` 会
   `hipErrorInvalidKernelFile`（运行时声称 gfx1100，而 wheel 里只有 gfx1101 内核）；
   真要覆盖只有 `11.0.1` 可用。256 分辨率训练用 `--batch-size 16 --accum-steps 2`
   （游戏与训练共享 12G 显存）。
5. **重启游戏**需 `rm -rf .nw-profile/SingletonLock`（`.nw-profile/` 是本地运行时目录，不在仓库里）；绝不要 `pkill -f 'nw.exe'`
   （宽匹配会误杀进程）。
6. **实机输入必须过场景护栏，不许绕过**。在 MZ 里 `jump` 就是「确定」键（Z）、
   `attack` 就是「取消」键（X），所以策略在标题/菜单画面上按一下跳跃就等于提交菜单
   选项——**首次实机自举就是这样进入了玩家的存档**。因此：
   - 只在 `Scene_Map` 内派发按键：`agent.js` 的 `V.safety()` → `cdp_backend.unsafe_reason()`
     → `_check_safe()`（`apply_action`/`step_frame` 的硬闸），读不到场景一律**失败即拒**；
   - **`Scene_Map` 还不够**：对话框正在**等待选项**时（`$gameMessage._choiceCallback`
     或 `choices()` 非空且 `isBusy()`）也是决策点——确定键就是策略的跳跃键，自由输入会替你
     选中高亮项（难度雕像、yes/no 都在此列）。`safety().awaitingChoice` 为真时一并拒绝。
   - 不许为了「先跑起来」关掉护栏，也不许新开一条绕过 `_check_safe()` 的按键派发路径；
     诊断探针确需原始按键时必须显式 `enforce_safety=False` 并写明理由；
   - **`V.CONFIRM_SCENES` 目前是空列表，而且必须保持为空**，直到某个场景被**证明**
     只需要一次 ok 且 ok 不做任何提交。机制本身留着（`press_ok()`：只按 keymap 的确定键，
     有 `max_confirm_presses` 上限），但名单为空。
     **反例（务必记住）**：`Scene_Transport` 曾被放进这个名单，理由是「它是游戏的过场
     画面、需要按一次确定」。它其实是**选择传送点的菜单**——ok 在那里就是选传送目标，
     于是 agent 选了它根本没选的落点、把角色扔到伤害陷阱上。这正是这道门槛存在的意义。
     **判据是「ok 会不会提交选择」，不是「画面看起来无害」**。菜单类
     （`Scene_Title`/`Scene_Menu`/`Scene_File`/`Scene_Load`/`Scene_Save`/`Scene_Transport`…）
     永远不得进白名单；`tests/test_confirm_scenes.py` 直接在 node 里加载真实的
     `agent.js` 读这份名单（不是读一份副本），并逐个钉死这些菜单不在里面。
   - 实机启动前 CLI 会先查一次场景，不在游戏内且不在白名单里就拒绝启动（退出码 3）；
   - **退出时保持暂停是刻意的，不要「修」成自动恢复**。`close(resume=False)` 停掉
     ticker，为的是不让角色在无人操作时被敌人打死。代价是「跑完游戏像卡住了」，所以
     缺陷是它**太安静**、不是它发生了：CLI 退出时必须说明游戏被留在暂停态以及怎么
     恢复（页面里 `__ash.pump.resume()`，或 `close(resume=True)`，或重启游戏）。
     要交还控制权必须由人显式要求。
   三层护栏 + 暂停退出 + 确认白名单由 `tests/test_safety.py` 钉死。
7. **时间尺度只有一个来源：`config/game.yaml:control_interval_s`**（论文 Appendix G
   用 0.25 s）。agent 每步推进 `control_frame_skip = 0.25*60 = 15` 帧，语料按
   `corpus_fps = 4` 抽帧，两者都由它派生。**不许再写死第二处**：IDM 是在 agent 的
   (obs[t], obs[t+1]) 上训的、却用在语料帧上，两边 Δt 必须相同。实测踩过 120× 的
   错配（agent 1 帧 vs 语料 2 s）。`tests/test_time_scale.py` 钉死派生关系与
   CLI 的不一致拦截（退出码 4）。
8. **伪标签退化时必须拒绝更新 π，不许「先训了再说」**。实测：IDM 的逐帧 logits 方差
   0.008 而类间偏置高达 0.13，即 **argmax 完全由偏置决定、与输入无关**；随机初始化的
   IDM 就已经 97% 输出同一个类。在常数目标上训 π 只会把它教成「永远输出同一个动作」，
   而 `policy_val ~1e-6` 看起来像收敛。`bootstrap.pseudo_label_stats` +
   `max_pseudo_majority=0.9` 会跳过更新并报错，由 `tests/test_pseudo_labels.py` 钉死。

9. **实机自博弈必须用 `drive="realtime"`，不许用帧泵**。帧泵是「手工驱动 ticker」，
   为**确定性回放/搜索**而设，它会**改变被观测的游戏**。同一陷阱、同一角色实测：

   | 驱动方式 | `_pRealState` | 陷阱是否传送 |
   | --- | --- | --- |
   | 引擎自己的 ticker | `3→6→1→0→9→4→7…` 正常流转 | **会**（`map 8 (0,8)` → `map 14 (0,18)`）|
   | 帧泵（`ticker.update(合成时钟)`） | 1200 帧**全程钉在 6** | **不** |

   后果：帧泵驱动下采到的实机轨迹是「一个动不了的角色」，IDM 自然学不到输入相关的
   动态——**这比 IDM 数据量的问题更上游**。自博弈要的是「不改变系统」，不是确定性；
   确定性只对回放/搜索有意义。`tests/test_drive_mode.py` 钉死两种模式不得混同
   （realtime 不装帧泵、按真实时间按持键；pump 仍可用于回放）。

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

99 项全过才算改动成立。四个测试文件各钉死一类静默失败：
- `tests/test_loop.py`：三个动作维度（BUTTONS / ActionSpace / 策略类别数）必须同源、
  轨迹的 `act` 是「每个转移一个标签」（长度 T-1，不是 T）、策略训练目标是窗口内每个
  位置、bootstrap 必须按 D^R 裁剪语料（否则检索是空操作）。
- `tests/test_safety.py`：实机输入护栏三层（场景探针失败即拒 / 后端硬闸不派发按键 /
  runner 中止该轮）+ 退出保持暂停 + 等待选项的对话框被拒（难度雕像）。
- `tests/test_confirm_scenes.py`：`CONFIRM_SCENES` 为空，且传送菜单等菜单类不在其中。
- `tests/test_time_scale.py`：`control_interval_s` 派生出的 agent 步长与语料抽帧率必须
  一致，CLI 不一致时拒绝启动。
- `tests/test_pseudo_labels.py`：伪标签退化成常数时不得更新 π，且必须报告。
- `tests/test_drive_mode.py`：实机自博弈不得接管引擎 ticker（帧泵会改变游戏行为）。
