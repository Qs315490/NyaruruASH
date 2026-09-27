# 注意事项与踩坑实录

只写已实测的结论。新坑随时追加。

## 本次重写过程中新踩的坑

1. **HDBSCAN 的 cosine 度量不可用**：sklearn BallTree 报
   `Unrecognized metric 'cosine'`。嵌入已 L2 归一化，euclidean 距离与
   余弦距离单调等价，直接用 `metric="euclidean"`。
2. **ImpalaCNN 要求 (B, T, H, W, C) 五维**：直接喂 (B,H,W,C) 会在
   `b, t = x.shape[:-3]` 解包失败。调用方必须显式带 time 维。
3. **interleave 的 flatten 位置**：`stack(dim=2)` 后必须 `reshape(b, ws*2, d)`
   （在维度 1 上重组），`flatten(2)` 会得到 (B, ws, 2D) 使 cat 报错。
4. **空历史 padding 越界**：窗口左端不足、`pad == ws` 时
   `hist[pad]` 越界。历史为空时必须留零填充（bootstrap 的 w=0 时刻），
   只有 `0 < pad < ws` 才能复制最早帧。
5. **`.gitignore` 意外为目录**、包缺 `__init__.py`：复制代码时只拷了模块
   文件。`memory/__init__.py` 还引用已删除的 milestones → ImportError。
6. **uv 缓存默认只读**：必须
   `export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python`。
7. **bash / read 工具的文件系统视图可能不一致**（本会话实测：/tmp 下文件
   bash 看得到、read 看不到，或反之）。跨工具传文件时放到会话工作区内。
8. **实机自博弈会在菜单里「按」进玩家存档（最严重的一次）**。MZ 的 ok/cancel
   键与游戏动作同键：`jump` = Z = 确定、`attack` = X = 取消。策略的动作空间是按
   玩法动作建的，但在标题画面上一次「跳跃」就等于选中高亮项——有存档时高亮项就是
   「继续」，于是未训练的实机自博弈直接读进了玩家的存档。
   修法（三层，缺一不可，见 `tests/test_safety.py`）：
   - `agent.js` 的 `V.safety()` 报当前场景，**读不到就算不安全**（失败即拒）；
   - `cdp_backend.apply_action/step_frame` 派发按键前调 `_check_safe()`，非
     `Scene_Map` 抛 `UnsafeSceneError`——这是硬闸，任何调用方都绕不过；
   - `runner.run()` 每轮先问 `unsafe_reason()`，不安全就**中止该轮**并在报告里记
     `aborted`，绝不「先按了再说」；`ash run --backend cdp` 启动时同样先查一次，
     不在游戏内直接拒绝启动（退出码 3）。
   教训：**「先跑起来看看」在实机上不成立**。凡是有副作用的环境（别人的存档、
   不可逆状态），护栏必须先于第一次运行存在。
9. **跑完游戏停在「暂停」是刻意的，不是坏了**。`close(resume=False)` 会
   `ticker.stop()`，为的是不让角色在无人操作时被敌人打死；CLI 每轮结束都 `close()`，
   于是进程退出后游戏 **frameCount 不再前进**（实测 `58455 -> 58455`，且
   `requestAnimationFrame` 并未被劫持——所以不要往 rAF 方向排查）。
   **不要把它「修」成自动恢复**，那正好破坏被刻意保护的东西。真正的缺陷是它**太安静**：
   现在 CLI 退出时明确打印「游戏被留在暂停态」+ 恢复方法。
   恢复：页面里执行 `__ash.pump.resume()`，或 `env.close(resume=True)`，或重启游戏。
   排查提示：判断游戏是否还在跑，**不要**信 `__ash.pump.installed`——重新注入 agent 后
   新实例该标志是 false，旧 pump 的劫持却可能仍在；可靠做法是隔 1.5s 读两次
   `Graphics.frameCount` 看是否推进，救急可用
   `window.requestAnimationFrame = window.__ashNativeRAF`。

10. **时间尺度写死两处 → 静默毁掉整条流水线（120× 错配）**。agent 每步推进 1 游戏帧
    （1/60 s），语料却按 2 s 抽帧。IDM 学的是「解释 1/60 s 变化的那一个动作」，却被
    拿来标注相隔 2 s 的帧 —— 它 99.8% 只输出同一个类，π 靠预测这个常数类把
    `policy_val` 压到 1e-6，日志上没有任何异常。
    修法：`config/game.yaml:control_interval_s`（论文 Appendix G 用 0.25 s）作为唯一
    来源，派生 `control_frame_skip=15`（agent）与 `corpus_fps=4`（语料），CLI 在不一致
    时拒绝启动（退出码 4）。`tests/test_time_scale.py` 钉死。
11. **伪标签是常数时训 π 比不训更糟**。实测 IDM 的逐帧 logits 方差 0.008 vs 类间偏置
    平方 0.13 → **argmax 由偏置决定、与输入无关**；随机初始化的 IDM 已 97% 输出同一类。
    在常数目标上训 π 只会把它教成常量策略，且损失趋零「像收敛」。
    修法：`bootstrap.pseudo_label_stats` 先量标签分布，某类 > `max_pseudo_majority`
    (0.9) 就跳过更新并报错（`tests/test_pseudo_labels.py`）。
    教训：**「损失很低」不是「学会了」**；要检查标签本身有没有信息。
12. **论文里写了但代码里没实现的承诺**：`bootstrap.py` 的模块文档写着 IDM 会用
    「随机策略采样补充动力学覆盖」，但实现里根本没有随机 rollout。这类「文档已承诺、
    代码没做」的差异不会报错，只会让结果变差。现已实现
    （`InferenceRunner.random_rollout`，同样过安全护栏）。

13. **护栏太绝对 = 把游戏永久停在过场画面**。规定「非 Scene_Map 一律不按键」之后，
    游戏进了自定义过场 `Scene_Transport`——它**不会自己结束**（手动 `update()` ×400 场景
    不变）、**必须按一次确定**（有 `pressOk` 方法），于是被永久停在那里。实测：
    `pump.resume()` 后 `frameCount 41121 → 41121` 不动，泵推进 120 帧场景也不变。
    教训：**「一律禁止」不是安全，是把没想清楚的场景外包给运气**。正确做法是分类：
    - `Scene_Map` → 全动作空间；
    - **确认类白名单**（`agent.js: V.CONFIRM_SCENES`，目前只有 `Scene_Transport`）→
      `press_ok()` 只按 keymap 的确定键，绝不按随机动作，`max_confirm_presses` 有上限；
    - 其余（标题/菜单/读档/存档/未知）→ 绝对禁止，菜单里的确定就是「选中菜单项」。
    白名单只有一处定义，菜单类永远不得加入（`tests/test_safety.py` 钉死）。

14. **绝不要把游戏留在「有伤害来源」的地方无人运行**（这次真的把角色玩死了）。
    角色卡在伤害柱里持续掉血时，我为了验证「陷阱应该传送」把 ticker 交还引擎并让它跑，
    结果血扣到 0、进 `Scene_Gameover`。**观察到「位置 10 秒不变」时就该立刻停住**，
    而不是继续让它跑着「观察」。默认的 `close(resume=False)`（保持暂停）才是安全的，
    `resume=True` 只应在**人马上接手**时用。
    - 顺带查清：这个陷阱不会自动传送；角色的 `_pRealState=6/10` 是**受伤态**，
      连续命中会不断刷新 `_invincibleTime`，所以没有可操作窗口。玩家给的破法是
      「按 2 次 C 冲过去」或「受伤后起跳」，说明**移动输入本身是正常的**。
    - `Scene_Gameover` 的恢复（本游戏自定义）：`pressBoxOk()` 读 `_selectBox.selectedButton()`，
      `"up"` → 读档，`"down"` → `backToTown()`。直接调 `backToTown()` 若死亡动画已结束会
      在 `removeChild` 上抛错，要**带 null 判断**重做那四步（removeChild / destroy /
      clearPictures / `Utils.backToTown()`）才会成功。实测恢复后回到城镇地图 6。

15. **`Scene_Map` 不等于「可以随便按」：等待选项的对话框是决策点**。难度是靠一座
    **难度雕像**的对话框选的，而策略的跳跃键（Z）就是确定键——自由输入会替你选中高亮项。
    原来的护栏只看「场景是不是 Scene_Map」，所以这个场景**照样会被盲选**。
    修法：`V.safety()` 新增 `awaitingChoice`（`$gameMessage._choiceCallback` 存在，
    或 `choices()` 非空且 `isBusy()`），为真时 `unsafe_reason()` 直接返回理由 → 该轮中止。
    普通对话（无选项）不受影响，仍是正常 gameplay。
    结论：**决策点要按「ok 会不会替你提交选择」来判断，不是按场景判断**。本游戏已知的
    决策点至少两个：传送点选择菜单（`Scene_Transport`）、难度雕像对话框。两者都必须由
    **策略**或人来决定，安全层永不代选。

16. **重启游戏的正确做法，以及一个匹配错目标的实例**。
    步骤（沙箱内看不到游戏进程，要用提权）：
    1. 找到**游戏本体**的 PID（`nw.exe` 且带 `--remote-debugging-port` 且**不含 `--type=`**
       且**不含 `proton`**），连同它的 `steam.exe` 包装进程一起，用 `kill <pid>` 逐个发 TERM；
    2. 确认 `ps` 里再没有 `nw.exe`、且 `curl :9222/json/version` 已断；
    3. `rm -rf .nw-profile/SingletonLock`；
    4. `./scripts/launch_nw_proton.sh`（后台，需提权），再等 CDP 起。
    坑：模式 `/nw\.exe/ && /--remote-debugging-port/` **会同时匹配 Proton 启动器**——它的
    命令行里就是 `proton run Z:.../nw.exe Z:... --remote-debugging-port=9222 ...`。我第一次
    就因此打到了错误的 PID，游戏本体没停。**必须额外排除 `proton` 和 `--type=`**。
    验证重启是否干净：`typeof window.__ash === "undefined"`（无旧注入残留）+
    `SceneManager._scene` 有值 + `Graphics.frameCount` 在推进。

17. **帧泵会改变游戏行为——这不是理论，是同场景 A/B 实测**。
    角色站在伤害陷阱上，只换驱动方式：
    - **真实 ticker**：`_pRealState` 在 `3→6→1→0→9→4→7` 之间正常流转；站了约 25 秒后
      **陷阱把它传送走**（`map 8 (0,8)` → `map 14 (0,18)`，正是它上次落脚的地方）。
    - **帧泵**：`_pRealState` **1200 帧全程钉在 6**（受伤态），`inv` 被反复刷新，
      **20 秒游戏时间过去陷阱始终不结算、角色无法移动**。
    机制上说得通：帧泵用 `ticker.update(V.pump.time)` 推进一个**只在自己手上走的合成时钟**，
    墙钟/真实 ticker 语义随之失真，依赖它们的动画完成回调与状态退出条件就跑不完。
    后果（这是最重要的一点）：**帧泵驱动下的实机轨迹是「一个动不了的角色」**，
    `key_moments=[0]`、IDM 只学到类别偏置，都在这条链的下游。
    修法：自博弈用 `drive="realtime"`（不装帧泵、按真实时间按持键），帧泵只留给确定性回放。
    `tests/test_drive_mode.py` 钉死。

18. **不要用 `$gamePlayer.x/y` 判断角色有没有动——这个游戏不用它。**
    **我因此连续误判了三次，还写过一条错误的「空地图」结论（已删）。**
    实测（30 秒连续采样，玩家自己用键盘走动）：

    | 字段 | 30 秒内 |
    | --- | --- |
    | 角色精灵的 `sprite.x/y`、`$gamePlayer.screenX()/screenY()` | **持续变化**（715→718→719→711→645→573→531→459→…→594）|
    | `$gamePlayer.x/y`、`_realX/_realY`、`$gameMap._displayX/_displayY` | **全程不变** |

    这是**动作平台游戏**：角色用自由像素坐标移动，位置记在**精灵**上，MZ 的格子坐标
    `x/y` 只是一个固定锚点，**永远不会随移动更新**（`$gamePlayer.moveByInput` 也被改成
    空函数）。所以「读 x/y 看角色动没动」是一条**永远返回「没动」的死路**。
    **正确做法：`$gamePlayer.screenX()/screenY()`，或 `_spriteset` 里那个
    `sprite._character === $gamePlayer` 的精灵的 `x/y`。**
    另一条**不是**证据的现象：`$dataMap.tilesetId === 0` 且全图图块为 0 在这里是**正常**的
    ——这个游戏的地图用**插件图层/图片**绘制（note 里就有 `<layer name:Map004-shadow,...>`），
    不走 MZ 图块数据。我把它误当成「空地图/损坏」，浪费了好几轮。

    排查顺序（先排除环境，再怀疑输入）：
    1. `Graphics.app.ticker.started` —— false 表示游戏被暂停（我的 `close()` 刻意会停它）；
    2. `$gameMap._interpreter.isRunning()` / `$gameMessage.isBusy()` —— 事件/对话进行中时
       `canMove()` 按设计返回 false；
    3. **用 `screenX()/screenY()` 观测角色是否移动**（不要用 `x/y`）；
    4. 只有以上都正常，才去怀疑输入通路。

## 历史踩坑实录（详见 docs/game-architecture.md）

- 快照/回滚 5 个致命缺陷（canonicalNumbering、单段路径、函数占位符、
  整数键错位、删除循环误删方法）。
- 策略接线 4 个静默失败（logit 下标、state 维度、图像尺寸、proposal 覆盖 policy）。
- 录像与状态错位一帧、动作历史泄漏（先取向量后记动作）。
- 搜索 rollout 不得写入训练数据（52% 帧自相矛盾的教训）。
- 256 分辨率 + batch 32 OOM：根因是游戏与训练共享 12G 显存。
  用 `--batch-size 16 --accum-steps 2`。

## 运行环境铁律

- **不要设 `HSA_OVERRIDE_GFX_VERSION`**。本机 RX 7700 XT（`gcnArchName=gfx1101`，
  12G），装的 wheel 是 `device-gfx1101`，原生检测就解析成 gfx1101。实测
  `11.0.0` / `11.0.2` 必报 `hipErrorInvalidKernelFile`（运行时对外声称 gfx1100，
  而 wheel 只带 gfx1101 内核），`11.0.1` 可用但纯属多余。
- `.uv-cache` 不可删：torch 精确 ROCm 版本已从索引下架，缓存是重装唯一来源。
- 重启游戏先 `rm -rf .nw-profile/SingletonLock`；绝不要 `pkill -f 'nw.exe'`。
- 没视频的数据无法训练（旧 pack_runs 直接 ValueError）。
