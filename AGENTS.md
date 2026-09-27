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
   - **菜单可以「只按取消键」退出，但这不等于放开护栏**。agent 会自己按进菜单
     （实测：一轮真实回合第 39 步按进了 `Scene_SkillSt`），而它永远出不来——于是整轮报废、
     每次都要人救。为此加了 `V.MENU_SCENES`（**与 `CONFIRM_SCENES` 是两份清单，判据不同**：
     那份问「ok 安不安全」，这份问「cancel 安不安全」，**两份不得有交集**）+
     `cdp_backend.escape_menu()`。约束：只派发 **cancel**、只限清单内场景、`awaitingChoice`
     或 `messageBusy` 或未知场景一律拒绝、有次数上限（`max_menu_escapes`/`max_menu_presses`）。
     **它不是绕过 `_check_safe()`**：`_check_safe(escape=True)` 仍是同一道闸门，只是多了一个
     窄模式；`enforce_safety=False` 依旧整体关闭。**逃生失败必须中止该轮**——
     第一版曾写漏 `break`，测试当场抓出：`kind == "menu"` 落空后会继续往下派发
     **游戏按键**，正是这道闸门要防的事。
     自定义场景只有在**实测 cancel 能退出且不提交任何东西**之后才准进 `MENU_SCENES`
     （`Scene_SkillSt` 实测：2 次 cancel 回到 `Scene_Map`，map/hp 不变）。
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
  **报告必须能说出「为什么」退化**：`logit_diagnosis()` 把 logits 拆成
  「每类随帧变化的幅度」（`logit_temporal_std`）与「类间偏置的跨度」
  （`logit_bias_spread`），比值 `bias_over_temporal ≥ 1` 就说明 argmax 由偏置决定、
  与输入无关。实测过的真 IDM 是 0.008 vs 0.13（比值 ~16）；假后端那轮是 0.11~0.15 vs
  0.55~0.67（比值 3.6~6.2）。这两个数字此前是**手写探针**跑了两次才拿到，
  现在每轮报告自带。

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

   **已找到机理并修复（2026-09-25）**：泵原先让合成时钟从 **0** 起算，而 ticker 自己的
   `lastTime` 是**真实页面运行时间**（实测 ~40000ms）。PIXI `Ticker.update(currentTime)`
   只在 `currentTime > lastTime` 时才跑一帧，否则把 `deltaTime/deltaMS/elapsedMS` 全置 0
   **并且不通知任何 listener**——于是**第一泵帧被整个吞掉**；紧接着它执行
   `this.lastTime = currentTime`，把**时间轴重设到合成时钟上**；卸载后引擎的下一帧看到
   的是 ~40 秒的跳变。
   修法：`V.pump.anchorClock()` 在 `install()` 与 `resume()` 时把 `origin` 与
   `ticker.lastTime` 都锚到真实时钟，之后喂 `origin + ticks*dt`——**dt 仍是固定 1/60 s
   （确定性不丢），但时间轴始终落在真实时间轴内**。
   实测：装好后首帧 `deltaMS = 16.667`、`deltaTime = 1.0`（旧实现是 0 ＝被吞），
   卸载后引擎 `deltaMS ≈ 20.8`、`lastTime ≈ now`（不再跳变）。
   `tests/test_pump_clock.py` 用会记录「喂进去的时间与算出的 delta」的假 ticker
   钉死三条：首帧不被吞、纯泵时钟不脱离真实时间轴、`resume()` 后引擎下一帧是正常帧。
   **尚未验证**：这能否治好「后摇不结束 / 陷阱不传送」那个症状——那是陷阱现场的另一场实验。

   **帧泵实验的铁律（实测踩过）**：**一臂一个全新进程**。泵装过一次后本进程内游戏就被
   弄坏了（后摇不结束 → 无敌帧不结束 → 陷阱不传送、角色无法操作），因此
   「先跑泵臂、再跑引擎臂」里的**引擎臂是无效对照**（它面对的是已经坏掉的游戏）。
   犯过这个错误并据此得出过「引擎也会卡在 6」的**错误结论，已撤回**。
   另：旧项目的 `__vpt.restore()` **恢复不完整**（只回位置，不回 hp/`_pRealState`），
   不能靠它在两臂之间复位。

## 架构速览

- 推理循环：`loop/runner.py`（论文 Algorithm 2）——双记忆（短期 w_s 对 + 长期 w_l 个
  关键时刻）、卡死计时器、逐 4 帧嵌入检测。
- 自举：`loop/bootstrap.py`（Algorithm 4）——先更新 K，再在自博弈轨迹上更新 IDM，
  最后用 IDM 伪标签 + K 记忆更新 π。
  **训练窗口必须是惰性的（`WindowDataset`）**：把窗口物化成
  `(n_win, w_s, H, W, 3)` float32，对一个 9405 帧的语料视频就是 **59 GB**，记忆库再加
  15 GB —— 16 GB 机器上全语料自举**永远跑不完**（实测：18 分钟单核 100%、零产物），
  此前一直被 600 帧的 `corpus-live` 子集掩盖。现在只存一份 uint8 帧缓冲 + 每窗口的
  索引，实测同一视频峰值 RSS **3.54 GB**。`ds["frames"]` / `ds["memories"]` 会**显式抛
  KeyError**（旧的静默分配才是真问题），要用 `ds.batch(idx)`；`ds["actions"]` 小到可以物化。
  `tests/test_lazy_windows.py` 钉死：不许出现第二份帧拷贝、batch 必须与
  `np.stack` 出来的窗口逐位相同、记忆槽必须指向真正在窗口之前的关键帧。
  **K 的批量查询用 `kdm.classify_sequence()`**（bootstrap 用），单帧在线查询用
  `classify()`（runner 用），两者定义必须一致——`tests/test_kdm.py` 直接比对两条路径。
  批量只快 ~1.3×（`approximate_predict` 本身仍是逐点 KD-tree 查询），**别指望它救性能**：
  实测干净负载下 1.2 ms/帧（早先报的 3.4 ms/帧是在实机 run 抢 CPU 时测的）。
- 编排：`loop/orchestrator.py`（Algorithm 1）——infer → retrieve → bootstrap 循环。
- 关键时刻：`memory/kdm.py`（HDBSCAN，多轨迹过滤）+ `memory/embeddings.py`（DINOv2）。
  **聚类前先 PCA 到 64 维**：384 维的 HDBSCAN 在 6 视频语料（68440 帧）上 6 分钟跑不完，
  64 维（保留 81% 方差）只要 113 秒。`fit` 与 `classify` **必须用同一个投影**，否则在线查询
  会被当成噪声、`key_moments` 静默恒为 0——`tests/test_kdm.py` 钉死这条不变量。
  K 的规模实测：3 视频 → keep 7 个簇；6 视频 → keep 91 个（`c_min=3`）。
  **拟合结果必须经 `cluster_of()` 查询，不许直接 `approximate_predict(kdm._clusterer, …)`**：
  clusterer 活在 PCA 空间里，喂原始 384 维向量会抛
  `ValueError: New points dimension does not match fit data!`（bootstrap 踩过，现已改回
  `cluster_of()`，`tests/test_loop.py` 钉死）。
  **K 的拟合结果缓存到 `data/corpus-kdm.pkl`（运行时生成，不在仓库里）**（`--kdm-cache` 可覆盖）：拟合是确定性的，
  但 6 视频语料要 113 秒 CPU，每次启动都付一遍毫无意义。缓存带指纹
  （语料文件表 + 嵌入分辨率 + 超参 + hdbscan/sklearn 版本），不匹配就重建；文件损坏/
  版本不符一律当作「没有缓存」，**缓存只许是加速手段，不许成为真相来源**
  （`tests/test_kdm.py` 钉死：往返一致性、指纹失效、损坏文件、embedder 不入盘）。
- 动作空间：`actions/space.py`。**只能追加，不能插入**——动作下标是写进轨迹、日志和
  checkpoint 的标签，`BUTTONS` 同理（mask 是按钮顺序上的位）。原 16 个 mask 保持原位，
  战斗/物品四个动词追加在后面：`special`(V 咸鱼技能，耗 SP)、`ult`(A 武器大招，需黄条)、
  `weapon_switch`(S 换咸鱼武器)、`item`(F 就地用物品——**唯一能即时回血的动作**)。
  `docs/game-systems.md` 早就把 S/A 记为缺口；`F` 当年与 M(地图)/D(快速读档) 一起被
  **有意排除**，现在按需加入（D 仍排除，速通用它等于作弊）。
  两个模型头的宽度都从 `len(action_space)` 来，`DEFAULT_NUM_ACTIONS` 必须等于
  `len(ActionSpace.minimal())`——`tests/test_action_space.py` 钉死这三条（下标不变、
  按钮位不变、常量等于真实空间）。
  **已验证的只是「键到位」**：四个键都真的派发到页面（keydown 计数实测），且引擎自己的
  `Input.keyMapper` 证实 `70→"item"`、`83→"cfish"`、`86→"subattack"`、`65→"zxc"`。
  **没验证的是「效果」**：当前存档 party 里 0 个物品（F 本就该无动作）、不在战斗中、
  且咸鱼武器系统不是 MZ 的 `equips`（`$gameParty.leader()._equips` 为空），所以
  S/V/A 的实际效果**仍需在战斗中验证**，不要当成已验证。
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

155 项全过才算改动成立。测试文件各钉死一类静默失败：
- `tests/test_loop.py`：三个动作维度（BUTTONS / ActionSpace / 策略类别数）必须同源、
  轨迹的 `act` 是「每个转移一个标签」（长度 T-1，不是 T）、策略训练目标是窗口内每个
  位置、bootstrap 必须按 D^R 裁剪语料（否则检索是空操作）、bootstrap 的关键时刻必须
  走 `cluster_of()`（PCA 空间）而不是拿原始向量喂 clusterer。
- `tests/test_safety.py`：实机输入护栏三层（场景探针失败即拒 / 后端硬闸不派发按键 /
  runner 中止该轮）+ 退出保持暂停 + 等待选项的对话框被拒（难度雕像）。
- `tests/test_confirm_scenes.py`：`CONFIRM_SCENES` 为空，且传送菜单等菜单类不在其中。
- `tests/test_time_scale.py`：`control_interval_s` 派生出的 agent 步长与语料抽帧率必须
  一致，CLI 不一致时拒绝启动。
- `tests/test_pseudo_labels.py`：伪标签退化成常数时不得更新 π，且必须报告。
- `tests/test_drive_mode.py`：实机自博弈不得接管引擎 ticker（帧泵会改变游戏行为）。
- `tests/test_pump_guard.py`：帧泵在 ticker 实例上留下的 `start()` 影子必须能被**后来注入
  的** agent 摘掉，且 `resume()` 必须**核实** ticker 真的起来了才报成功。
  反例（实测）：pump 安装过 → 重新注入 agent.js（`window.__ash` 被替换、`V` 被丢弃）
  → 新 agent 手里没有 ticker 句柄 → `unguard()` 静默无效 → `resume()` 返回
  `{resumed:true}` 而 ticker 仍是 stopped+shadowed：**游戏永久冻结，且所有恢复路径
  都在撒谎**（`close(resume=True)`、`resume_game()` 同样返回 true）。
  修法：影子里的判断读**活的** `window.__ash`；`unguard(ticker)` 接受显式 ticker；
  `resume()` 用 `t.started` 核实结果。**凡是「声称做了某事」的返回值，都必须来自事后的核实。**
- `tests/test_lazy_windows.py`：训练窗口不得被物化。反例（实测）：一个 9405 帧的语料视频
  物化出 59 GB 的 `win_frames` + 14.7 GB 的 `win_mem`，16 GB 机器上全语料自举永远跑不完。
  钉死：只存一份 uint8 帧缓冲（不许第二份拷贝）、`batch()` 与 `np.stack` 的窗口**逐位相同**、
  记忆槽指向真正在窗口之前的关键帧、惰性数据集上 π 仍能训练。
- `tests/test_pseudo_labels.py`：除「拒绝在常数目标上训 π」外，还钉死 `logit_diagnosis()`
  能把「偏置主导」与「输入驱动」分开，且这两个数字必须真的进报告。
- `tests/test_menu_escape.py`：菜单逃生的边界。钉死：`MENU_SCENES` 与 `CONFIRM_SCENES`
  **不相交**、`CONFIRM_SCENES` 仍为空、**只按 cancel**（记录实际派发的按键）、一到
  `Scene_Map` 立刻停手、`awaitingChoice`/未知场景**拒绝**、逃生失败**必须中止该轮**
  （不能落空派发游戏按键）、次数有上限。
- `tests/test_action_space.py`：动作空间**只能追加**。钉死：`BUTTONS` 前 13 项的下标与
  顺序不变（否则所有历史 mask 全部变义）、原 16 个 mask 的下标不变、四个新动词确实在
  尾部、`DEFAULT_NUM_ACTIONS == len(ActionSpace.minimal()) == 20`、以及四个键在
  `config/game.yaml` 里**真有绑定**（没有键的动作是静默空动作）。
- `tests/test_pump_clock.py`：手工时钟必须锚在真实时钟上。反例（实测）：合成时钟从 0 起算
  而 ticker 的 `lastTime` 是 ~40000ms 的真实运行时间 → 在 PIXI `Ticker.update()` 里
  `currentTime > lastTime` 不成立 → **首泵帧被整个吞掉**（`deltaTime=0`、listener 不触发）
  且 `lastTime` 被重设到合成时间轴上。钉死：首帧 delta 恰好等于 dt、纯泵时钟不脱离真实
  时间轴、`resume()` 之后引擎的下一帧是正常帧（不是 40 秒跳变）。
- `tests/test_report_incremental.py`：每轮必须在 bootstrap **开始前**就落盘。反例（实测）：
  报告只在 `run()` 返回后写一次，于是死在 bootstrap 里的 run 把整轮推理数据抹掉了——
  连「这轮看到过没有关键时刻」都无从得知。


