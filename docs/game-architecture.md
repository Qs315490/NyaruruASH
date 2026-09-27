# 游戏侧架构：《咸鱼喵喵》如何被 CDP 驱动

本文档只记录**已实测**的事实。猜测一律标注。

## 运行时

- RPG Maker MZ 1.6.0，跑在 Proton + nwjs SDK 0.64.1 win-x64 下。
- CDP 端口 `:9222`（launch 时 `--remote-debugging-port`）。
- 主循环是 **PixiJS ticker**（`Graphics.app.ticker`），不是 rAF。
  注入的帧泵必须劫持 ticker，`requestAnimationFrame` 劫持无效。
- 重启游戏前必须 `rm -rf .nw-profile/SingletonLock`。
- **绝不要**执行 `pkill -f 'nw.exe'` 之类的宽匹配（会误杀其他进程）。
- 启动脚本：`scripts/launch_nw_proton.sh`；`NYARURU_GAME_DIR`/`NYARURU_NWSDK_DIR`/`NYARURU_PROTON`/`NYARURU_CDP_PORT` 环境变量覆盖默认值。
  （待迁移入库）依赖 `.nwjs-sdk/nwjs-sdk-v0.64.1-win-x64/nw.exe`。

## 注入 agent（`src/ash/memory/js_source.py`）

注入后暴露 `window.__ash`，三个能力：

1. **帧泵**：劫持 PixiJS ticker，`__ash.pump.pump(n)` 精确推进 n 个逻辑帧，
   游戏在我们不给帧时真的停下来——这是回滚搜索的前提。
2. **状态图快照**：`__ash.snapshot()` / `__ash.restore(json)`。
   序列化 `$gameSystem/$gameSwitches/$gameVariables/$gameMap/$gamePlayer/$gameParty`
   等全局对象图，处理循环引用与共享子对象。
3. **确定性**：`__ash.rng.enable(seed)` 替换 Math.random。

## 快照/回滚的 5 个已知致命缺陷（历史踩坑实录）

1. `canonicalNumbering()` 预注册所有可达对象 → `encode` 第一分支全是裸引用，
   271 字节 vs 应有 227,532 字节。修法：canonicalNumbering 改空操作，
   ref 稳定性由 encode 按排序键遍历保证。
2. `restore()` 单段路径从 `window[parts[0]]` 出发再取最后一段 →
   写到不存在属性上，根对象从未被还原。修法：从 window 走到父节点。
3. 函数占位符覆盖对象自有属性上的真函数（`Input.update`）。
   修法：目标已是函数时保留原函数。
4. 编码器/解码器键序不一致（字符串排序 vs 整数键数值排序）→
   `{__r:N}` 全部错位，只在整数键存在时触发（44 事件的 map 必崩）。
   修法：`canonicalKeys()` 三处统一——整数键数值升序在前，其余字符串排序。
5. `applyInPlace` 结尾删除循环删掉运行时注入的方法
   （加密 `main.bin` 注入的 `eventIsStarting` 等）。
   修法：删除循环跳过函数。

迁移状态：`env/cdp_backend.py` 与 `memory/js_source.py` 已按上述修法改写，
**回归测试已补**（`tests/test_restore_node.py`、`test_restore_methods.py`，离线 Node 复现，含变异检查）。

## 反调试

零售版游戏带反调试，本包使用
nwjs SDK 版规避。若换零售版需重新评估。

## 已知未解问题

- `position_scale` 反常（历史记录，机制未明）。
- 键位映射见 `config/keys.yaml`；游戏配置见 `config/game.yaml`。

## 跳跃：高度由按住时长控制（实测 2026-09-26）

机制：**长按跳跃键越高**；二段跳 = 长按到最高点 → 松开 → 再长按。
实测（真机 map 8，用 `physics.py` 读最高点，每个时长一次）：

| 按住帧数 | ≈毫秒 | 高度 |
|---|---|---|
| 2 | 33 | 40 px |
| 4 | 66 | 66 px |
| 8 | 133 | 111 px |
| **15** | **250** | **172 px** ← `control_frame_skip` 的固定值 |
| **25** | **417** | **212 px** ← 最大 |
| 40 | 666 | 197 px |

**这条直接限制了当前的动作编码**：`step_realtime` 每次动作「按下 → 按住整整一个控制间隔
（250 ms）→ 松开」，**按住时长不可控**。后果有两条，都已实测：
- **跳跃高度被截断在 81%**（172 / 212）；
- **二段跳的时机无法表达**（需要「最高点松开、再按」，而松开时刻由动作边界决定）。

⇒ 这同时给「角色离不开初始小屋 / 卡在某个区域」提供了一个**比「探索不足」更可能的解释：
它跳不上去**。（此前的结论是「随机策略自己浪费」，现在要改成「策略浪费 + 跳跃能力被限」两条并存。）

**已实现（方案 B，2026-09-26）**：**按住跨步**——动作里相同的按钮集合连续出现时，
env **不松开**；切换动作才松开（`press()`/`release()`/`step_holding()`，`--hold-actions` 开关）。
于是「长按到最高点 → 松开 → 再按」= `jump → noop → jump`，**二段跳成为策略可表达的动作序列**，
而不是我写死的宏。默认仍是一步一松（安全契约），跨步按住只在使用方显式开启时生效，
且**在换动作 / `press_ok` / `escape_menu` / `close()` / 退出路径上都会被强制松开**。

真机验证（`scripts/probe_jump_hold.py`，同一落点内比较）：

| 模式 | 相对起跳高度 |
|---|---|
| `jump`（250 ms） | 203 px |
| `jump→noop→jump`（松开再按） | **246 px** ← 二段跳生效 |
| `jump→noop→noop→jump`（按太晚） | 203 px（第二跳没触发） |
| `jump×2`（按住两步） | 180 px |
| `jump×3`（按住三步） | 0 px ← **未解释**，疑为该落点上方遮挡，不作证据 |

注意：**绝对高度随落点变化**（同一招式在另一处测得 172 px），只有同一次运行内的相对值可比。

补充的机制：**水下地图跳跃=上升、跳跃加强，且二段跳可无限连用即持续上升**。
⇒ 同一动作在不同区域的动力学不同，**「动作 → 效果」不是函数**，这是 IDM/LAM 都必须面对的上下文依赖。

