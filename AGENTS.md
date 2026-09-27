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
   **也不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（2026-09-26 实测）：
   设了之后同一个 fold 在 epoch 1 起 loss 全 NaN，不设则 `1.1664 → 0.8827 → 0.7488`
   干净收敛，两次 A/B 各观察一致。`clip_grad_norm_` **救不了它**——梯度里已经是 NaN 时
   `clip_coef = max_norm/(total_norm+1e-6)` 也是 NaN，而 `if clip_coef < 1` 判 False，
   于是裁剪静默不发生。**代价已经付过**：那次运行的日志（`runs/probe-diff-branch.log`，**运行时产物、不在仓库里**）epoch 3~7 都打印
   「非有限 loss，停止本轮」，那次的 49.6% 与「loss 到 0.0000 完全背下来」都不可信，
   结论已撤回（见 `docs/status.md`）。**训练命令里不要带这个变量**，也不要用它来"省显存"。
5. **重启游戏**需 `rm -rf .nw-profile/SingletonLock`（`.nw-profile/` 是本地运行时目录，不在仓库里）；绝不要 `pkill -f 'nw.exe'`
   （宽匹配会误杀进程）。
6. **实机输入必须过场景护栏，不许绕过**。在 MZ 里 `jump` 就是「确定」键（Z）、
   `attack` 就是「取消」键（X），所以策略在标题/菜单画面上按一下跳跃就等于提交菜单
   选项——**首次实机自举就是这样进入了玩家的存档**。因此：
   - 只在 `Scene_Map` 内派发按键：`agent.js` 的 `V.safety()` → `cdp_backend.unsafe_reason()`
     → `_check_safe()`（`apply_action`/`step_frame` 的硬闸），读不到场景一律**失败即拒**；
   - **对话选项：普通的选择由 agent 自己操作，难度选择必须拦住**（2026-09-25 放宽）。
     确定键就是策略的跳跃键，所以「等选项」时自由输入会替你选中高亮项；但原来**一律拒绝**
     等于让 agent 永远过不了剧情（对话选项正是游戏推进的方式）。现在的规则：
     - `safety()` 报出选项原文（`choices`）与 `guardedChoice`；
     - 拒绝名单 `V.GUARDED_CHOICES` 就一条：**`难度`**（子串匹配）。
       证据是**从运行中的游戏抓的真实原文**（`scripts/watch_choices.py`，只读探针）：
       `['按下简单难度按钮', '按下普通难度按钮', '\C[18]按下困难难度按钮']`——
       三条都含「难度」，所以匹配这个共同词比匹配整句更稳（改措辞也拦得住）；
       注意 MZ 文本带控制码（`\C[18]` 是设置颜色），**匹配前必须剥掉**，否则困难那条会漏过去。
     - **匹配前先剥控制码**；**选项读不到时一律拒绝**（fail closed）——分类不了的选项
       可能就是不许答的那个。
     - 误拦的代价是那一轮被中止（可恢复），漏拦的代价是 agent 永久替你选了难度：
       **错误方向必须是「多拦」**。其余不可逆选择（如 yes/no）目前**不拦**，这是明确选定的方向，残余风险记在此处。
     - **难度对话关不掉**（实测：只能选，没有取消键）——所以「一律拒绝」等于把游戏**永久
       卡在那一屏**，而且让它到达的每一轮都中止。为此加了 `--difficulty <文本|序号>`：
       **由操作者预置**答案，agent 触发时 `resolve_difficulty()` 移动高亮并确认。
       它是**第三种窄模式**（`_check_safe(difficulty=True)` → `difficulty_reason()`），约束：
       只允许**受保护的那条**难度选择、只允许移动到**预置的那个**选项（文本按 `choiceTexts`
       匹配——**已剥控制码**，序号**从 1 数起**）、按键有上限、且**必须核实选项真的消失**才算成功
       （不核实就等于「我按了所以成功了」）。未配置预设 / 不是受保护选择题 / 高亮项读不到 →
       一律拒绝；解析失败时 runner **中止该轮**，绝不落空把游戏按键派进对话框。
       预设默认值来自 `config/game.yaml:difficulty_preset`（**默认「简单」**，理由是
       **语料是速通录像，速通都在简单难度下跑**——让 agent 在别的难度上玩却拿简单难度的
       录像当监督信号，等于在比较两个不同的游戏；要按难度分开训练就改这一行或每次
       `--difficulty` 覆盖，**不是混着用**）。`--difficulty off` 可关掉（回复成「拒绝并中止」）。
   - **菜单可以被 agent 操作，但「光标下的条目」是判据**（2026-09-25 第二次放宽）。
     这个游戏的 ESC 菜单**不是 MZ 的 Window**，而是自绘 Sprite（`Sprite_MenuCommand` /
     `Sprite_MenuSystem` / `Sprite_Item` …），标准窗口探针**完全看不到**——必须按它自己的结构读：
     每个面板有 `_items` + `_selectIndex` 光标，条目带**稳定符号名**（实测抓取）：
     ```
     _menuCommand : STATIC_TEXT_MENU_ITEM / _SKILL / _MAP / _BOOK / _SYSTEM
     _systemPanel : ..._BACK_TOWN / _RETURN_TO_TITLE / _RETURN_LOAD_GAME / _OPTIONS / _EXIT_GAME
     _itemPanel   : STATIC_TEXT_MENU_ITEM_CONSUMABLES / _WEAPONS / _STONES / _ORNAMENTS
     ```
     **用符号名而不是显示文本**：文本是本地化的、会随补丁改；而且**口述里的「返回菜单」
     在游戏里其实叫 `..._BACK_TOWN`（返回城镇）**——照口述写拒绝名单会漏掉它。商店/存档界面
     是普通 MZ 场景（`Window_Command._list` + `index()`），所以两种结构都要读。
     - `V.OPERABLE_SCENES` 里**允许普通按键**；清单**只放实测见过的场景**
       （`Scene_Menu` / `Scene_Shop` / `Scene_SkillSt`）。**别按 MZ 惯例猜场景名**：
       第一版曾列了 `Scene_Item`/`Scene_Equip`/`Scene_Skill`/`Scene_Status`——
       **这些类确实存在**（枚举 `window.Scene_*` 能看到），但**截图显示**物品/宝珠/饰品
       是 `Scene_Menu` **内部的面板**，而它的「技能」页只是**演示按键**的信息页。
       列了没用的场景 = 看起来有覆盖、其实没有。
       其余 `V.MENU_SCENES` 仍是 **cancel-only 逃生**（两份清单**允许重叠**，判据不同：
       这份问「能不能操作」，那份问「cancel 安不安全」）；
     - **拒绝名单 `V.MENU_ENTRY_DENY` = 最后一栏里除「返回城镇」外的四项**
       （返回标题 / 读取存档 / 设置 / 退出游戏，全是单向操作）；
     - **任一**面板的高亮条目命中拒绝名单 → 拒绝（哪个面板在处理输入是游戏的事，
       「任一命中即拒」是安全的一侧：误拒只赔一轮，误放赔存档）；
     - **读不到高亮条目的可操作场景一律拒绝**（fail closed，分类不了的可能就是危险的那个）。
     - **死亡后的 `Scene_Gameover` 也放开，但只许「读取存档」**（第三次放宽，方案已选定）。
     **自动恢复（2026-09-26 实现）**：光标停在「返回小镇」时，闸门过去会**中止整轮**——
     实测一次 2500 步采集因此在 1314 步被掐断。现在有**窄模式** `resolve_gameover()`：
     **必须先按「上」把光标移到「读取存档」再按确定**（测试钉死「第一次派发的键必须是 up」，
     因此 ok 永远不会落在被拒绝的那一项上）、按键有上限、且**必须重新读场景确认真的离开
     GAME OVER** 才算成功，否则中止该轮（fail closed）。注意它走 `_dispatch` 而不是
     `step_frame`——后者会再过一次普通闸门（那张屏正是被拒的）。
     **同时删掉了旧的 `guard_scene()`**：它**没有任何调用者**，而且选的正是被拒绝的
     「返回小镇」，与规则矛盾——留着就是给下一个人的陷阱。
       它同样**不是 MZ 的窗口**：标题 `_titleText=STATIC_TEXT_GAMEOVER_CONTINUE`（「是否继续？」），
       光标是 `Sprite_GameoverBox._selected ∈ {up,down}`，两个选项的符号画在各自的 sprite 上：
       ```
       up   = STATIC_TEXT_CONTINUE_YES   → 读取存档（游戏自己的 continue 路径）← 放行
       down = STATIC_TEXT_CONTINUE_NO    → 返回小镇                        ← 拒绝
       ```
       拒绝整屏的代价是**每次死亡都中止该轮、要人来救**，而 map 8 的陷阱让死亡变成常态：
       ```js
       performTerrainDamage(): terrainDamageCD = 30;      // 每 30 帧重新施加
                              _staggerTime = 60;          // 后摇 60 > 30 → 永远走不完
       挨打无敌帧 = beHitInvincibleTime - 10 = 80         // 不会跟着刷新 → 无敌先耗尽
       if (_delayRevert <= 0 && !isDeath()) revertToLastGroundingPos();   // 死了就不传送
       ```
       实测（角色死在陷阱上的那一帧）：`_invincibleTime=0` 而 `_staggerTime=36`、
       `terrainDamageCD=7`、`hp=0`、`_delayRevert=6` —— **无敌帧确实没撑到后摇之后**，
       且 `!isDeath()` 让传送永不结算。
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
   - **`V.CONFIRM_SCENES` 现在只有一个成员，而且每个成员都必须有「ok 只做一件事」的证明**：
     `Scene_ItemObtain`（拿到道具的弹窗）——游戏自己的绑定就是证明（`nya_game.js`）：
     ```js
     Scene_ItemObtain.prototype.createObtain = function () {
         this._obtainItem = new Sprite_ItemObtain({...});
         this._obtainItem.pressOk = this.popScene.bind(this);   // ok = 关闭，无高亮条目、无提交
     };
     ```
     实测的代价：不列它时实机日志是
     `aborting random-policy rollout: scene 'Scene_ItemObtain' is not Scene_Map gameplay`
     ——**agent 每次捡到东西都会当场停住**（观察到「获取道具时似乎会暂停 agent」）。
     机制本身：`press_ok()` 只按 keymap 的确定键，有 `max_confirm_presses` 上限。
     **真机实测（走完整条链）**：`before: scene=Scene_ItemObtain confirm=True` →
     `press_ok -> True (35 ms)` → `after: scene=Scene_Map inGameplay=True`（一次 ok 即清）。
     **同一个坑还暴露了另一个真缺陷**：`press_ok` 原先靠 `_pump_frames(1)` 让「按下」跨一帧，
     而 **realtime（唯一用于实机自博弈的模式）里没有帧泵，它是 `return 0` 的空操作**——
     于是 down/up 在同一毫秒发出、**游戏一帧都没看到按键**（日志里三次 ok 挤在 5 ms 内，
     `Scene_ItemObtain` 三次都清不掉）。现在改为 `_hold_one_frame()`：有泵就泵一帧，
     没泵就**按真实时间睡一帧**。**凡是「跨一帧」的假设，在 realtime 下必须用真实时间实现。**
     **凡 ok 会在条目中选择的场景一律不得进入**（不管画面看起来多无害）。
     **反例（务必记住）**：`Scene_Transport` 曾被放进这个名单，理由是「它是游戏的过场
     画面、需要按一次确定」。它其实是**选择传送点的菜单**——ok 在那里就是选传送目标，
     于是 agent 选了它根本没选的落点、把角色扔到伤害陷阱上。这正是这道门槛存在的意义。
     **判据是「ok 会不会提交选择」，不是「画面看起来无害」**。菜单类
     （`Scene_Title`/`Scene_Menu`/`Scene_File`/`Scene_Load`/`Scene_Save`/`Scene_Transport`…）
     永远不得进白名单；`tests/test_confirm_scenes.py` 直接在 node 里加载真实的
     `agent.js` 读这份名单（不是读一份副本），并逐个钉死这些菜单不在里面。
   - 实机启动前 CLI 会先查一次场景，不在游戏内且不在白名单里就拒绝启动（退出码 3）；
   - **暂停要「保持住」，而且它跨不过客户端断开**（2026-09-26 实测）。停 ticker 只够一瞬间：
     引擎自己会调 `Graphics._app.start()` 把它拉回来（实测约 2 秒后恢复）。`agent.js` 靠给
     `ticker.start` 打影子压制它，而影子只在**帧泵安装**时生效——realtime 不装泵，所以
     原本没有任何压制。现在加了独立的 `V.paused` 标记 + `__ash.pauseGame()`（影子同时尊重
     它），`close(resume=False)` 走这条并**核实它保持住了**（延迟后再读一次，只看一眼抓不到
     自我恢复）。**但暂停仍跨不过我们断开 CDP**：重连后影子所在 ticker 实例往往已被重建
     （实测 `guarded=false`、帧数继续涨）。因此**训练期间要跑
     `scripts/keep_paused.py`**（守护进程，每 2 秒重新施加暂停，实测 25 秒帧数冻结）。
     另注意：**另一个 CDP 客户端连上来会让游戏恢复**（实测），守护进程会在 2 秒内按回去。
   - **探针与采集脚本也必须留在暂停态**（2026-09-26）。`close()` 默认
     `resume=False` 是对的，但 `scripts/{collect_selfplay,probe_jump_height,probe_jump_hold,probe_telemetry_semantics}.py`
     一度都传了 `resume=True`——于是**每次探针/采集结束后角色被留在世界里挨打**（实测 hp 150→0、
     并留下一张 GAME OVER 界面）。只有 `ash record`（人类在开）才该恢复运行，且它的代码里有
     注释说明理由。
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
  与输入无关。
  **`majority_share` 必须带上「是哪个类」与 noop 占比**（`majority_class` /
  `majority_class_name` / `noop_share`）：同一个 0.6 的集中度，如果那个类是 `right`
  就是「在学跑步」，如果是 `noop` 就是「在学站着不动」——**结论相反，而光看比例分不出来**。
  实测（2026-09-25，观察到「agent 静止不动的时间更长了」）：
  ```
  0r2lVc1uKa0 (裁过的速通)  1999 帧 | noop 48.6%
  BV19s4y1y7un(裁过的速通)  1999 帧 | noop 46.8%
  house-002   (自己的录制)  1999 帧 | noop  2.7%
  ```
  ⇒ **π 的监督信号近一半是「不动」，所以它学成尽量不动**；而 `max_pseudo_majority=0.9`
  拦不住它——那是个**偏斜**分布，不是常数分布。
  **但这 50% 本身是 IDM 在分布外的默认答案**（核对过速通视频：按键基本没停过 →
  真标签应接近 0% noop）。判据：IDM 在**自己那条 CDP 抓帧管线**的数据上很准
  （人工回放逐类召回 92~96%、整体 91.6%；自己的录制上 noop 只 **2.2%**），
  一到**视频抽帧**上就说 ~50% noop。根因与 K 匹配不上**同源**：视频帧经过缩放/重编码/色差。
  - **帧间隔不是原因**（实测步长 1/2/3 的 noop 占比 50.9/51.1/51.8%，几乎不变）；
    裁剪是**降低** noop 的（未裁 65.4% → 裁过 43.4%），所以裁剪不是元凶。
  - **试过的修法：视频管线数据增强 → 实测失败，代码已删除**（2026-09-25）。
    增强后 noop 从 43.4% **升到 91.9%**，而且**在同分布的录制上从 2.2% 升到 59.6%**——
    那不是域适应，是把模型弄坏了：JPEG+缩放往返+色彩偏移一起上，把标签赖以判断的
    细微运动差异一起毁掉，模型退化成「拿不准就答多数类」。
    **不要再往这个方向试**（把训练输入退化到目标域并不等于域适应，这条已经付过一次代价）；
    真要改 IDM 的域泛化，得换别的思路（例如在目标域上有真值、或改模型而非改输入）。
  - **一条曾被写下、后被推翻的结论（保留以示警告）**：曾据此写「可靠的 π 监督信号
    只能来自自己录的 CDP 数据（2.2% noop），所以要靠多录」。**这条已被测否**：
    那 2.2% 是拿「录制应当几乎全是动作」这个**未经核实的假设**当对照的；把 `house-002`
    的**真标签**算出来（真 noop **60.2%**）后，同一个 IDM 的准确率只有 **5.7%**。
    IDM 真正会的是**同场景记忆**（同场景内随机划分 91.6%），**跨场景等于瞎猜**
    （训 3 个屋子 → 留出 `human-001`：**45.5%**，多数类基线 50.4%；换冻结 DINOv2 + MLP 头
    仍然 45.1%）。
    ⇒ **不要再用「和某个假设的真值比」来评价 IDM，一律用跨场景留出 + 多数类基线。**
    ⇒ 也不要再提「多录就能解决」：那既不是本项目的目标（项目要的是**自进化**），
    也补不上跨场景泛化。完整的证据链见 `docs/status.md`。实测过的真 IDM 是 0.008 vs 0.13（比值 ~16）；假后端那轮是 0.11~0.15 vs
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

## 现状（先读 `docs/status.md`）

**速通视频按键标签路线（2026-09-27）尚未收口：同视频内可学，跨视频未迁移。**
交接文档 `docs/keycast-handover.md`，**其第 0 节是必读的修正**（先前的结论被限定过）。
**数据布局（2026-09-26 起）**：一支视频一个文件夹 `data/videos/<id>/`（video.mp4 + meta.json +
labels.npz + 可选帧库），代码入口 `src/ash/data/video_pack.py`（`load_meta`/`artifact`/`migrate`，
`artifact()` 对未迁移的视频回退到旧路径）。
`meta.json` 必须含 **`game_area`**（游戏画面矩形，两角点，**必须实测不许猜**）与 `video`（**pack 内真实文件名**）；
`python -m ash.data.video_pack` 打印各 pack 状态与缺口，`validate_meta()` 校验矩形越界/缺失。
测试 `tests/test_video_pack.py`（12 项）。
- **标签干净可用**：4 支视频、3 支可用、37,903 tick、与语料逐帧对齐、方向语义经独立信号验证；
  键帽字母是**播主自己的绑定**，映射必须逐视频确认。
- **同视频内可学**：输入取「**当前 + 未来 8 帧 @30 fps**」时 macro-F1 **0.594~0.730**、动作键 **0.68~0.86**
  （因果单帧对只有 0.197 / 0.108）。**按键的证据是它的后果，所以窗口必须向前看**；
  灰度优于彩色，网格越细越好（32² > 16²），未来窗峰值在 0.27 s。
- **跨视频未迁移**：0.00~0.12。**四种干预全部失败**——多视频训练、增强、管线归一化（裁游戏区）、
  域适应（CORAL/MMD/DANN）。弱正信号：动作键均值可达 0.30~0.33；**方向键反而最难迁移**
  （它依赖全局滚动尺度，随分辨率/缩放变化）。
- **不要再试**：因果表述下的 14 条排除（`docs/keycast-handover.md` 第 4 节）＋上述跨视频 4 条。
- **剩下的变量是视频数量/多样性**（3 支太少）——这是**采集成本**，不是算法问题。

**已批准的例外（2026-09-26）**：B 站速通视频**自带按键显示**，
可以从中读出人类按键序列当训练标签。这**属于人工数据**（原话：「按键显示算人工数据，
不过当前路线阻力重重，只能适当取舍」）。⇒ 用这条路训练出来的模型**不是「无人工数据」**，
报告时必须写明；详细实测见 `docs/status.md` 二之九。
**目标（2026-09-26）**：**纯视觉速通模型**——推理时只吃像素；
**训练期不得使用任何人类数据**；**可以参考视频**。
⇒ 真人录像——旧 VPT 演示 `data/idm-human/`、真人游玩录制 `house-001/2/3`——**都不是本目标的训练数据**，
而且**它们都不在仓库里**（`data/` 被 gitignore，需要时自备）。
`--idm-replay` 也因此不在可用路径上。关于 IDM 能力的那些实测（跨房间、macro-recall、
可辨识性）**测的是架构能力，不是目标系统的可行性**——引用时要说清这一点。
⇒ **引擎状态允许当训练期老师**（已明确允许）：`obs.state` 里的 `physics.px/py/vx/vy`
（玩家自己的位置与速度）、`player.screenX/screenY`、`hp`、`scene`、`mapId`、`switches/variables`
都可用于**筛选/标注/蒸馏**，但**交付的策略只能看像素**。
`src/ash/data/effect.py` 负责这件事：**`player.x/y`、`realX/realY` 永不变化**（本游戏是自由像素
平台游戏），所以它们**绝不可**当作移动信号——那是已经付出过三次误判代价的坑。

**管道全通、真机逐项验证过；阻塞在学习信号那一段。**
**2026-09-26 决策：接受这条能力边界**——停止学习信号方向的实验，交付「护栏 + 记忆 K + 检索」框架。
**2026-09-27 把这条边界收窄了一次，引用时必须区分两半**：

- **被推翻的那一半**：「按键 → 画面变化」不是学不到，而是**表述选错了时间轴**。
  输入改成「**当前 + 未来 8 帧 @30 fps**」后，**同视频内** macro-F1 **0.594~0.730**、
  动作键 **0.68~0.86**（因果单帧对只有 0.197 / 0.108）⇒ **同视频内的学习信号是存在的**。
- **仍然成立的那一半**：**跨视频迁移**在四种干预下全部失败（0.00~0.12），
  瓶颈指向**域的数量/多样性**（3 支视频太少），属采集成本而非算法问题。

⇒ 本项目的交付描述应是「**框架可用 + 同视频内可学 + 跨管线未迁移**」：
**不要写成「自我精进已实现」**（跨管线那一步没有成立），也**不要写成「学不到」**（已被推翻）。
要在这条路上另开实验仍须先登记并确认；决策记录与两条交接路径见 `docs/status.md` 第五节。
其余几条最容易误解的事实：

1. **IDM 在任何测过的配置下都没有学到「按键 → 画面变化」这个映射**（2026-09-26 重测，
   根因说法已第一次修正）：不是「跨房间不泛化」——**同房间、同时间段的诚实切分也学不到**。
   判据用**「运动帧上」准确率**（只看真标签非 noop 的留出帧；只答 noop 的模型在这里是 0%），
   它在所有配置里都是 0~35%，而 **macro-recall 恒在 0.09~0.25**（按类数的随机水平就在
   0.11~0.25）⇒ 实际等于没学会。
   三个必须记住的数字来源问题：
   - 旧的 **45.5%** 是**配置不同**的运行；同条件重测生产头是 **53.1%**（基线 50.4%）。
   - 旧的 **91.6%（同场景）** 来自**随机划分**，相邻近重复帧泄漏到两侧；诚实的时间序切分
     之后只在基线附近（−5.0 ~ +2.5 个百分点）。
   - **跨房间 7 组平均低于基线 8.0 个百分点**（最高 +4.6，最低 −23.2）。
   **训练集的真实构成**：`house-001/002/003` 是**同一房间的三次会话**（帧间余弦 0.995~0.998），
   所以那些「训 3 个屋子 → 留出 human-001」的测试**训练集只有 1 个房间**。
   **扩覆盖救不了**（固定 4000 对时，房间数 1→6 让总体准确率 33.6%→72.7%，但那只升到基线，
   同时运动帧准确率 69.4%→2.2%）：多加房间买到的是「更会押多数类」。
   完整的表与探针入口见 `docs/status.md` 第三节之二。
2. **任务本身有信号**（动作帧的画面变化是 noop 帧的 3~4 倍），重叠 74~78% → 天花板约 70~80%，
   所以「不可学」不是解释，「模型没泛化」才是。
3. **别重复已否证的假设**（帧间隔、颜色偏移、取景比例、视频管线增强、换编码器）——
   `docs/status.md` 第二节逐条记着数字。
4. **「第三种架构也失败」这条已撤回**（2026-09-26）：差异分支那次（49.6%、loss 0.0000）
   是被 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 打成 NaN 的运行，
   日志里第 3~7 个 epoch 都写着「非有限 loss，停止本轮」。所以「把训练集背到 0 证明严重
   过拟合」**不成立**，**不要引用那个 0.0000**。⇒ 现在只剩两条架构证据（45.5% / 45.1%），
   它们只说明「不比多数类基线好」，**不足以支撑「换架构一定没用」**。
5. **另一个可复用的正面事实**：运动信息必须**以图像形式**进卷积。
   `[ea, eb, eb−ea]`（全局平均池化后相减）留出 **49.0%**；把 `|a−b|` 直接送进 trunk
   留出 **85.7%**（像素差阈值 79.2%）。**但「动了」≠「按了哪个键」**，跨场景的动作映射
   仍然要标签。

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
  **抓来的速通视频当不了 K 的锚点（实测）**：与实机帧余弦只有 0.81~0.84，按每视频固定
  区域裁掉叠加层后 0.87~0.91，**裁完重新拟合 K 再拿同一实机帧查询仍然进不去保留簇**——
  根因是采集管线不同（缩放/重编码/叠加层），不是视频里没那个地方。
  ⇒ **要 K 触发的区域必须自己录**（`ash record`，实测 0.96~1.0），且**同区域要 3 段
  独立会话**（`min_distinct_trajectories=3`：单轨迹的簇会被丢，map 8 覆盖到 0.9577
  却仍不触发就是这个原因）。
  **拟合结果必须经 `cluster_of()` 查询，不许直接 `approximate_predict(kdm._clusterer, …)`**：
  clusterer 活在 PCA 空间里，喂原始 384 维向量会抛
  `ValueError: New points dimension does not match fit data!`（bootstrap 踩过，现已改回
  `cluster_of()`，`tests/test_loop.py` 钉死）。
  **K 的拟合结果缓存到 `data/corpus-kdm.pkl`（运行时生成，不在仓库里）**（`--kdm-cache` 可覆盖）：拟合是确定性的，
  但 6 视频语料要 113 秒 CPU，每次启动都付一遍毫无意义。缓存带指纹
  （语料文件表 + 嵌入分辨率 + 超参 + hdbscan/sklearn 版本），不匹配就重建；文件损坏/
  版本不符一律当作「没有缓存」，**缓存只许是加速手段，不许成为真相来源**
  （`tests/test_kdm.py` 钉死：往返一致性、指纹失效、损坏文件、embedder 不入盘）。
- **跳跃高度由按住时长控制（实测）**：250 ms 的固定按住只拿到最大高度的 **81%**
  （172/212 px），二段跳（最高点松开再按）**在当前编码里无法表达**。这是动作空间的
  真实限制，不要把它归因成「探索不足」——详见 `docs/game-architecture.md`。
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

250 项全过才算改动成立。
**注意（2026-09-27）**：**真人录像从不入库**（`data/` 被 gitignore，24447 帧的那份演示要自备）。
没有它时，`tests/test_demo_mapping.py` 的两个实文件测试会**按自身设计跳过**
（`pytest.skip("recorded demonstrations are not present")`）
⇒ **克隆后的基线是 248 通过 + 2 跳过**；自备那份录像即恢复 250 通过。

测试文件各钉死一类静默失败：
- `tests/test_loop.py`：三个动作维度（BUTTONS / ActionSpace / 策略类别数）必须同源、
  轨迹的 `act` 是「每个转移一个标签」（长度 T-1，不是 T）、策略训练目标是窗口内每个
  位置、bootstrap 必须按 D^R 裁剪语料（否则检索是空操作）、bootstrap 的关键时刻必须
  走 `cluster_of()`（PCA 空间）而不是拿原始向量喂 clusterer。
- `tests/test_safety.py`：实机输入护栏三层（场景探针失败即拒 / 后端硬闸不派发按键 /
  runner 中止该轮）+ 退出保持暂停。**对话选项的三种情形分别钉死**：普通选项**放行**、
  难度选项**拒绝**、选项读不到**拒绝**（fail closed）。
  另外用 node 加载**真实的 `agent.js`** 执行 `guardedChoice`（不是读一份副本）：
  三条真实原文（含带 `\C[18]` 的那条）必须被判为受保护、`['是','否']` 与 `['继续','跳过']`
  必须放行、且 `stripControlCodes` 真的把颜色码剥掉。
  **难度自动应答**（`--difficulty`）单独钉死：未配置预设时**一个键都不许发**；预置文本/序号
  能正确移动（`0→1` 一次 down、`0→2` 两次、`2→0` 用 up）并确认；**不是受保护选择题时拒绝**
  （这条保证它不会退化成「替 agent 答任意对话」的后门）；预置匹配不到任何选项时**不发键**；
  **按完选项仍在时不算成功**（核实而非假设）；高亮项读不到时拒绝。
  runner 侧另钉死两条：解析成功则**继续该轮**（不中止），解析失败则**中止且一步都不走**。
  **菜单条目守卫**另钉死：可操作场景 + 安全条目 → **放行**；命中拒绝名单 → **拒绝**；
  条目读不到 → **拒绝**；非可操作菜单 → 仍只允许 cancel。并在 node 里加载**真实 agent.js**
  逐项验证最后一栏五项（只有「返回城镇」放行）、物品分类放行、以及 GAME OVER 的
  `CONTINUE_YES` 放行 / `CONTINUE_NO` 拒绝。
  **两个阶段的进度必须分开报**（`maps` 只统计策略阶段、`random_maps` 统计随机阶段）：
  反例（实测）——第一次多轮实跑三轮都显示 `maps [[4]]`，而角色**其实已经在随机阶段走到了
  map 5**；用来判断「它出去了吗」的那个指标，恰恰看不见它出去的那一段。
- `tests/test_confirm_scenes.py`：`CONFIRM_SCENES` **只许含被证明过的场景**（当前仅
  `Scene_ItemObtain`，其 ok 被游戏绑定到 `popScene`），且传送菜单等菜单类**不在其中**；
  另钉死若干「看起来像弹窗但没被证明」的场景名不得混进来。
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
  **不相交**（`CONFIRM_SCENES` 现在恰好含 `Scene_ItemObtain`）、**只按 cancel**（记录实际派发的按键）、一到
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
- `tests/test_corpus_loader.py`：D^I 由 `--corpus` 与 `--recordings` 两个目录构成，
  且**录制文件也算语料**（K 在 D^I 上拟合，而抓来的语料是速通录像，不覆盖初始小屋/NPC 房间；
  实测 `key_moments` 每轮都是 0）。钉死：`observations` 与 `frames` 两种字段名都能读、
  目录缺失就跳过、`.npy` 优先 mmap、且**缓存指纹必须同时覆盖 `.npy` 与录制目录**——
  反例（实测）：语料打包成 `.npy` 并删掉原 `.npz` 后，只 glob `*.npz` 的指纹变成空，
  之后语料再怎么改都会被静默忽略、复用一份过期索引。
- `tests/test_record.py`：`ash record` 录出来的文件必须能被 `pretrain-idm`/`--idm-replay`
  直接读（这是录制器的**全部意义**：IDM 只用自博弈转移会塌回类别先验 val 4.64，
  靠 24447 帧人类数据才降到 0.29）。
  钉死：npz 往返后 `control_names` 仍是 17 物理键、掩码逐位不变、映射能解出来；
  **录制器绝不派发按键**（假 conn 会对任何 `Input.*`/`dispatchKeyEvent` 直接失败——
  派发的键会被记成「人类的选择」，正是要学的东西被污染）；键码位序与
  `LEGACY_CONTROLS` 同序；**静止画面上的按键只降权不删除**；
  「截图期间按键变化」的帧数必须被计数而不是隐去。
  另外用 node 加载真实的 `KEY_TRACKER_JS` 并驱动 keydown/keyup/blur，
  钉死掩码位序（Z=bit4、up=bit0、blur 清空）。
  **录制必须边录边落盘**（`session_dir` 溢出目录 + `flush_every`）：反例（实测）——
  第一版把所有帧留在内存、结束时只写一次文件，而后台作业**是被硬杀的**（harness 没把
  SIGTERM 送到 Python 进程，日志里既没有 `signal` 也没有 `saved`），于是**9713 帧
  （16 分钟）的操作一无所有**。现在每 `flush_every` 帧 `flush+fsync` 一次，
  硬杀最多丢那么多帧；`ash assemble <session_dir> --out <npz>` 可抢救一个进程已死的
  会话（真机验证：硬杀后仍恢复出 150 帧）。溢出目录非空时**拒绝开始**（两段会话绝不许
  拼成一条掩码流）。另一个被抓到的真 bug：帧上限与 `run()` 返回值原先读 `len(self.frames)`，
  而溢出模式下帧不留内存，于是上限永不触发、CLI 会报「什么都没录到」——现在用独立的
  `self.samples` 计数。
- `tests/test_effect_report.py`：**引擎状态（训练期老师）只能声称它读得到的东西**。
  钉死：`physics.px/py` 是权威位移信号；**`x/y`、`realX/realY` 永远不得当移动信号**
  （本游戏里它们冻结，读了会把每个转移都判成「没动」，正是已犯过三次的错）；
  无 `physics` 时退回玩家自己的速度（fake 后端走这条），再退回 `screenX/screenY`；
  **读不到状态 = unknown，不是 still**；`own_still`（人没动）与 `nothing_changed`
  （人和画面都没动）必须分开计数。
  另钉死回归：**`runner` 必须把每步的 `state` 与帧一起带出来**——`_frame()` 曾把它丢掉，
  而 `step()` 返回的是 `StepResult`，状态在 `.obs.state` 而不是 `.state`；
  读了错层级的话每个转移都会被算成「不可读」，**看起来像一次正常运行的静默失败**。
- `tests/test_hold_input.py`：**按住跨步**（`--hold-actions`）。钉死：重复同一动作**不重新按**
  也不松开（这就是长按）、切换动作**先松开再按**（这就是二段跳的「松开再按」）、
  `release()` 幂等且**不受安全闸限制**（松开是安全方向；在菜单里拒绝松开会把键永久卡住）、
  而**任何「非按住」的派发都先抬起已按住的键**（覆盖 `press_ok`/`escape_menu`/`apply_action`）、
  `close()` 在暂停前先松开（否则人类一恢复角色就走下悬崖）、默认路径仍是一步一松。
  反例（写出来又被测试抓到的真 bug）：`press()` 收的是**掩码**，传**动作下标**会派发错键
  （下标 3 = 掩码 3 = 上+下）。
- `tests/test_gameover_recovery.py`：GAME OVER 的窄模式恢复。钉死：**第一次派发的键必须是「上」**
  （因此 ok 永远不会落在被拒绝的「返回小镇」上）、**必须重新读场景确认真的离开**才算成功、
  光标没动时**一次 ok 都不按**、读不到光标/不在该场景一律**不发任何键**、
  `gameover_reason()` 只在这张屏上放行。
  反例（实测）：一次 2500 步采集因为光标停在拒绝项上被**中止在 1314 步**；而旧的
  `guard_scene()` **没有任何调用者**、且选的正是被拒绝的那一项。
- `tests/test_report_incremental.py`：每轮必须在 bootstrap **开始前**就落盘。反例（实测）：
  报告只在 `run()` 返回后写一次，于是死在 bootstrap 里的 run 把整轮推理数据抹掉了——
  连「这轮看到过没有关键时刻」都无从得知。


