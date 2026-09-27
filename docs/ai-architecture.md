# ASH AI 架构：实现与论文的对照

论文：**ASH: Agents that Self-Hone via Embodied Learning**
(arXiv 2605.14211v3)。本文件说明每个模块对应论文的哪部分、规模缩放决策、
以及数据流。

> **本文描述的是设计与接线**（怎么接、为什么这么缩放）。**哪些部分已经在真机上成立、
> 哪些被实测否证**，见 [`status.md`](status.md)——那里是唯一的证据链与决策记录。
> 特别是：**π 的学习信号（IDM 伪标签）实测不可用**，所以 Alg 4 的后半段在实务上
> 没有成立。

## 自精进循环（Algorithm 1）

```
        ┌────────────────────────────────────────────┐
        │ Step 1 推理：N 个 agent 在共享策略 π 下游玩， │
        │  K 检测新关键时刻 → 追加长期记忆；            │
        │  任一 agent Δ 步无新关键时刻 = 卡住           │
        └──────────────┬─────────────────────────────┘
                       ▼
        ┌────────────────────────────────────────────┐
        │ Step 2 检索：DINOv2 嵌入轨迹，贪心一对一窗口  │
        │  匹配从互联网语料 D^I 取 top-k → D^R         │
        └──────────────┬─────────────────────────────┘
                       ▼
        ┌────────────────────────────────────────────┐
        │ Step 3 自举：K ← D^R 重聚类；                │
        │  IDM ← 自博弈轨迹（真标签）；                 │
        │  π ← D^R + IDM 伪标签 + K 记忆构造            │
        └──────────────┬─────────────────────────────┘
                       └────→ 回到 Step 1
```

## 模块对照

| 论文组件 | 本实现 | 文件 |
| --- | --- | --- |
| 策略 π（双记忆） | IMPALA-CNN tokenizer + 6 层 causal transformer | `models/ash_policy.py` |
| 图像 tokenizer φ | ImpalaCNN（随策略训练；论文用冻结 SigLIP，缩放决策见下） | `models/impala_cnn.py` |
| IDM（双向注意力） | 双帧嵌入 + 拼接差分 + **动作空间分类头（20 类 CE）** | `models/idm.py` |
| 关键时刻分类器 K | HDBSCAN（`approximate_predict`，先 PCA 到 64 维）+ 多轨迹过滤 + 拟合结果缓存 | `memory/kdm.py` |
| 嵌入（冻结） | DINOv2 ViT-S/14，L2 归一化 | `memory/embeddings.py` |
| 检索（Algorithm 3） | 贪心一对一窗口匹配，O(w²) 循环的向量化 | `retrieval/matching.py` |
| 推理（Algorithm 2） | 双记忆 + 卡死计时器 + 逐 4 帧嵌入 | `loop/runner.py` |
| 自举（Algorithm 4） | K → IDM → π 顺序更新，10% holdout | `loop/bootstrap.py` |
| 编排（Algorithm 1） | infer → retrieve → bootstrap | `loop/orchestrator.py` |

## 规模缩放决策（有意偏差）

论文假设数据中心预算（SigLIP 常驻 + 28 层 transformer + 大语料）。
本机是单张 12G AMD 卡且**游戏进程与训练共享显存**：

- tokenizer 用项目现成 IMPALA-CNN，随策略联合训练（论文中 φ 也是被 π
  的梯度经过的，只有 DINOv2 冻结不变——所以这个偏差只是把 SigLIP 换小）。
- transformer 6 层、hidden 256、8 头（论文 28 层）。hidden 256 下 128×128
  输入、batch 8 实测量级在显存余量内；再大需先测。
- 并行 agent 默认 1（单游戏进程）。N>1 需要 N 个 CDP 会话，暂不支持。

改这些偏差前先登记并确认（AGENTS.md 铁律）。

## 数据流与格式

- 观测：`(T, H, W, C)` uint8 RGB。语料 npz 键 `frames`。
- 动作：multi-hot 掩码，14 个按钮（`actions/space.py: BUTTONS`，键位见 `config/keys.yaml`）
  → 运行时动作空间 **20 项**（`ActionSpace.minimal()`）。**IDM 是动作空间上的分类器**
  （20 类），**不是**多标签按键预测器；策略头也是 20 类分布。两个头的宽度都必须等于
  `len(action_space)`（`DEFAULT_NUM_ACTIONS`），历史上 17 / 28 / 13 三个数字各不相同，
  第一次自举就炸成 `addmm: shapes cannot be multiplied (256x17 and 28x256)`。
  **动作空间只能追加、不能插入**：下标是写进轨迹/日志/checkpoint 的标签，
  `tests/test_action_space.py` 钉死。
- 长期记忆 ρ：最近 w_l 个关键时刻观测（每 agent 追加式 memory bank）。
- 短期记忆：最近 w_s 个 (obs, action) 对；窗口左端不足时重复最早帧填充。

## 语料

`data/corpus/*.npz`（或 `scripts/pack_corpus.py` 转出的未压缩 `.npy` mmap，loader 优先用
后者——匿名内存不可回收，而 mmap 的页可以）：速通视频经 yt-dlp 下载、ffmpeg 抽帧
（**抽帧率由 `config/game.yaml:control_interval_s` 派生**，当前 0.25 s → **4 fps**，
与 agent 步长同源；**不要写死第二处**，`tests/test_time_scale.py` 钉死）、
统一缩放到模型输入分辨率打包。检索索引用 DINOv2 嵌入，可预计算缓存
（`--corpus-embeddings`）。**抓来的视频要按 `data/corpus-crops.json` 裁掉叠加层**
（`scripts/detect_game_rect.py --write` 生成；实测余弦 0.8249 → 0.9375）。

**抓来的速通视频当不了 K 的锚点（实测 2026-09-25，`scripts/measure_corpus_crop.py`）**：
K 在语料上聚类，实机帧要落进这些簇才会触发关键时刻。但抓来的帧与实机帧的余弦只有
0.81~0.84；按每个视频的位置固定区域裁掉 LiveSplit 计分板/横幅后升到 **0.87~0.91**
（裁剪确实有用），**可拿裁完的语料重新拟合 K、用同一张实机帧查询，仍然进不去任何
保留簇**。根因是**采集管线不同**（视频被缩放、重编码、带叠加层），**不是「视频里没有
那个地方」**——同一片森林在视频里清晰可见。
⇒ 结论：**需要 K 触发的区域必须自己录**（`ash record`，与实机同一条 CDP 抓帧管线，
实测余弦 **0.96~1.0**），且**同一区域要 3 段独立会话**：`min_distinct_trajectories=3`
会丢弃只跨 1 条轨迹的簇——map 8 被一段录制覆盖到 0.9577 却依然不触发，就是这个原因。

## 训练超参（bootstrap 默认）

lr 3e-5（AdamW）、grad_clip 1.0、epochs 3、batch 8、10% holdout。
256 分辨率训练必须 `--batch-size 16 --accum-steps 2` 等效策略
（游戏占满 12G 显存时的实测余量）。
