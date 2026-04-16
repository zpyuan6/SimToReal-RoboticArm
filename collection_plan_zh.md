# 真实数据采集计划

本文档记录一份更贴近实际执行条件的 RoArm-M3 真实数据采集方案。  
这份方案假设你**基本只能控制桌面目标的位置**，而无法稳定控制灯光、背景等环境因素。

当前计划主要覆盖：

- L1 / `task-id 0`：观察与验证
- L2 / `task-id 1`：预对齐、接近与恢复

## 计划原则

这版计划只依赖一个你可以稳定控制的变量：

- 桌面目标的位置

采集时的基本策略是：

- 预先定义少量固定摆放位置
- 每个批次只摆一次目标
- 同一批次内目标不再移动
- 用多个 repeat 采该位置下的一组数据

这版计划不会要求你：

- 人为控制灯光亮暗
- 频繁更换背景
- 精细制造复杂环境变化

如果这些因素自然变化，可以在采集笔记里记录，但不作为当前版本的主设计变量。

## 当前范围

当前计划覆盖：

- 低风险的 L1 观察类数据
- 中风险的 L2 接近类数据
- 仅基于目标位置变化的数据

当前计划不把 L3 抓取 / 抬升 / 搬运 / 放置作为主采集目标。

## 数据量目标

如果目标只是打通流程、做 smoke test 或 very small few-shot 试验，那么较小的数据量就足够。

但如果目标是同时支持：

- 微调 / 适配
- 离线测试

那么建议把当前计划扩展到一个更稳妥的数据量级。

本计划的推荐目标是：

- L1：5 个位置，每个位置 `10` 次 repeat
- L2：5 个位置，每个位置 `6` 次 repeat

按当前 primitive 数量粗略估算：

- L1：`5 x 10 x 5 = 250` 条 transition 左右
- L2：`5 x 6 x 6 = 180` 条 transition 左右

合计大约：

- `430` 条 transition

这个规模更适合做：

- 第一轮微调 / 适配
- 留出一部分数据做离线测试
- 按位置或按 repeat 划分 train / offline-test

## Primitive 速查

低风险观察类 primitive：

- `2` = `obs_center`
- `0` = `obs_left`
- `1` = `obs_right`
- `3` = `verify_target`

中风险接近类 primitive：

- `4` = `prealign_grasp`
- `5` = `approach_coarse`
- `6` = `approach_fine`
- `7` = `retreat`

## 目标摆放位置定义

先定义一个中心基准位置，然后只围绕它做少量平移：

- `P1`：中心
- `P2`：左偏
- `P3`：右偏
- `P4`：前偏
- `P5`：后偏

推荐移动幅度：

- 左右方向：约 `3-5 cm`
- 前后方向：约 `3-5 cm`

除非物体在移动过程中自然发生一点偏转，否则不需要额外控制物体朝向。

## 采集执行表

| 阶段 | 批次名 | 任务 ID | 物品如何摆放 | 是否需要移动目标 | 如何移动目标 | 采集执行指令 |
|---|---|---:|---|---|---|---|
| 0 | `smoke_verify` | 0 | 将目标放在中心基准位置，确保前臂相机能清楚看到目标。 | 否 | 不移动目标。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/smoke_verify.npz --primitives 2,3 --repeats 1 --task-id 0` |
| 1 | `v1_verify_p1_center` | 0 | 将目标放在中心基准位置。 | 否 | 10 次 repeat 全程保持目标不动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p1_center.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p2_left` | 0 | 在中心基准位置基础上，将目标略微向左摆放，但仍需完全处于视野中。 | 是 | 仅在开始前移动 1 次；相对中心左移约 `3-5 cm`；10 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p2_left.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p3_right` | 0 | 在中心基准位置基础上，将目标略微向右摆放，但仍需完全处于视野中。 | 是 | 仅在开始前移动 1 次；相对中心右移约 `3-5 cm`；10 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p3_right.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p4_front` | 0 | 在中心基准位置基础上，将目标沿前后方向略微向前摆放，仍需保持清晰可见。 | 是 | 仅在开始前移动 1 次；相对中心前移约 `3-5 cm`；10 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p4_front.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p5_back` | 0 | 在中心基准位置基础上，将目标沿前后方向略微向后摆放，仍需保持清晰可见。 | 是 | 仅在开始前移动 1 次；相对中心后移约 `3-5 cm`；10 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p5_back.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 2 | `v2_approach_p1_center` | 1 | 将目标放在中心基准位置，并确保接近路径无遮挡、无碰撞风险。 | 否 | 6 次 repeat 全程保持目标不动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p1_center.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p2_left` | 1 | 将目标摆在中心左侧少量偏移的位置，同时确保机械臂仍可安全接近。 | 是 | 仅在开始前移动 1 次；相对中心左移约 `3-5 cm`；6 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p2_left.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p3_right` | 1 | 将目标摆在中心右侧少量偏移的位置，同时确保机械臂仍可安全接近。 | 是 | 仅在开始前移动 1 次；相对中心右移约 `3-5 cm`；6 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p3_right.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p4_front` | 1 | 将目标摆在中心前侧少量偏移的位置，同时确保机械臂仍可安全接近。 | 是 | 仅在开始前移动 1 次；相对中心前移约 `3-5 cm`；6 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p4_front.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p5_back` | 1 | 将目标摆在中心后侧少量偏移的位置，同时确保机械臂仍可安全接近。 | 是 | 仅在开始前移动 1 次；相对中心后移约 `3-5 cm`；6 次 repeat 过程中不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p5_back.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 3 | `v3_verify_offsets_dense` | 0 | 从前面 5 个摆放位置里选出你觉得最稳定的 3 个位置继续补采。 | 是 | 每个批次开始前摆一次位置；批次内不再移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_verify_offsets_dense.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |

## 推荐执行顺序

为了减少手工操作量，建议按下面顺序执行：

1. `smoke_verify`
2. `v1_verify_p1_center`
3. `v1_verify_p2_left`
4. `v1_verify_p3_right`
5. `v1_verify_p4_front`
6. `v1_verify_p5_back`
7. `v2_approach_p1_center`
8. `v2_approach_p2_left`
9. `v2_approach_p3_right`
10. `v2_approach_p4_front`
11. `v2_approach_p5_back`
12. 可选补采批次：
   - `v3_verify_offsets_dense`

## 推荐数据规模

这份计划可以形成一版更适合支持“微调 + 离线测试”的首批真实数据：

- L1：5 个位置 x 10 次 repeat = 50 次 repeat
- L2：5 个位置 x 6 次 repeat = 30 次 repeat

合计：

- L1 + L2 共 80 次 repeat

按当前 primitive 数量粗略估算：

- L1：约 `250` 条 transition
- L2：约 `180` 条 transition

总计：

- 约 `430` 条 transition

这个规模更适合支撑第一轮：

- 微调 / 适配
- 离线测试
- 行为分析

## 推荐的数据划分方式

如果你的目标是同时做微调和离线测试，建议不要把所有数据都混在一起使用。

推荐的简化划分方式有两种：

1. 按位置划分

- `P1 / P2 / P3 / P4` 用于微调 / 适配
- `P5` 留作离线测试

2. 按 repeat 划分

- 每个位置前面的 repeat 用于微调 / 适配
- 每个位置最后 `2` 次 repeat 留作离线测试

如果你想减少分布偏差，通常更推荐第 2 种。

## 每批采集后的检查项

每完成一个批次，都建议执行一次检查：

1. 检查 `.npz` 文件：

```bash
uv run python -c "import numpy as np; d=np.load('data/real/<batch_name>.npz'); print(d.files); print(d['images'].shape, d['states'].shape, d['primitive_ids'].shape)"
```

2. 检查 session 目录中的内容：

- `meta.json`
- `frames/`
- `preview.mp4`

3. 确认以下几点：

- 目标始终在画面中可见
- 图像清晰度可接受
- primitive 顺序符合预期
- 输出文件名与批次名一致

## 说明

- 正式采集时建议保留原始分辨率图像。
- `.npz` 继续作为后续训练 / 分析的主结构化数据格式。
- `frames/` 和 `preview.mp4` 用于质检、回看和调试。
- 如果采集过程中灯光或背景自然发生变化，只需要在笔记里记录，不必作为当前计划中的控制变量。
- L3 建议等真实抓取链路稳定之后，再单独制定高风险后置采集计划。
