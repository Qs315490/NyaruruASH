# ASH AI 架构：实现与论文的对照

论文：**ASH: Agents that Self-Hone via Embodied Learning**
(arXiv 2605.14211v3)。本文件说明每个模块对应论文的哪部分、规模缩放决策、
以及数据流。

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
| IDM（双向注意力） | 双帧嵌入 + 拼接差分 + multi-label 头（VPT 风格简化） | `models/idm.py` |
| 关键时刻分类器 K | HDBSCAN（approximate_predict）+ 多轨迹过滤 | `memory/kdm.py` |
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
- 动作：multi-hot 掩码，17 物理键（`config/keys.yaml`）→ 运行时动作空间
  28 项（`actions/space.py`）。IDM 输出 17 键掩码，策略输出 28 项分布——
  **两者维度不同是设计如此**，接线时必须在键序上对齐（历史上 logit 下标
  错位的教训）。
- 长期记忆 ρ：最近 w_l 个关键时刻观测（每 agent 追加式 memory bank）。
- 短期记忆：最近 w_s 个 (obs, action) 对；窗口左端不足时重复最早帧填充。

## 语料

`data/corpus/*.npz`：速通视频经 yt-dlp 下载、抽帧（建议 20 fps 采样、
统一缩放到模型输入分辨率）打包。检索索引用 DINOv2 以 2 秒间隔嵌入
（论文设定），可预计算成 json 索引（`--corpus-index`）。

## 训练超参（bootstrap 默认）

lr 3e-5（AdamW）、grad_clip 1.0、epochs 3、batch 8、10% holdout。
256 分辨率训练必须 `--batch-size 16 --accum-steps 2` 等效策略
（游戏占满 12G 显存时的实测余量）。
