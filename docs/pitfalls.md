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

## 历史踩坑实录（详见 docs/game-architecture.md）

- 快照/回滚 5 个致命缺陷（canonicalNumbering、单段路径、函数占位符、
  整数键错位、删除循环误删方法）。
- 策略接线 4 个静默失败（logit 下标、state 维度、图像尺寸、proposal 覆盖 policy）。
- 录像与状态错位一帧、动作历史泄漏（先取向量后记动作）。
- 搜索 rollout 不得写入训练数据（52% 帧自相矛盾的教训）。
- 256 分辨率 + batch 32 OOM：根因是游戏与训练共享 12G 显存。
  用 `--batch-size 16 --accum-steps 2`。

## 运行环境铁律

- `HSA_OVERRIDE_GFX_VERSION=11.0.0`（gfx1101）。
- `.uv-cache` 不可删：torch 精确 ROCm 版本已从索引下架，缓存是重装唯一来源。
- 重启游戏先 `rm -rf .nw-profile/SingletonLock`；绝不要 `pkill -f 'nw.exe'`。
- 没视频的数据无法训练（旧 pack_runs 直接 ValueError）。
