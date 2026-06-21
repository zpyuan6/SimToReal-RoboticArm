# 真实数据采集计划

本文档记录当前正式的真实数据采集方案。  
本方案基于**当前已验证通过的最新代码基线**，适用于：

- L1 真实观察数据采集
- L2 真实接近数据采集
- L3 真实抓取搬运放置数据采集

这版基线已经人工确认通过：

- L1 / L2 / L3 的真实与虚拟动作基本一致
- `validate_actions.py` 的仿真侧会真正执行 primitive，而不是只摆姿态
- L3 可以完成抓、抬、运、放
- 放置前会下放，不再悬空释放
- 抓住后 `11/12` 阶段不会提前松手
- 左右观察角度已经调整到 `±30°`

此前在更早基线下采到的数据，建议统一视为：

- 历史数据
- 调试数据
- 基线未稳定前数据

默认不要直接混入新的正式数据集。

## 当前适用范围

本计划假设你目前能稳定控制的环境变量主要只有：

- 桌面目标的位置

而不能稳定控制：

- 灯光
- 背景
- 其他外观环境因素

这些因素如果自然发生变化，可以记录在笔记中，但不作为当前版本的主动设计变量。

## 采集原则

- 从当前已验证基线重新建立正式数据集
- 每个批次内目标保持不动
- 只在批次之间移动目标
- 保留原始分辨率图像
- `.npz` 继续作为主结构化数据格式
- `frames/`、`meta.json`、`preview.mp4` 用于质检和调试
- L3 不再只做小规模试采，而是纳入正式全量采集

## Primitive 速查

### L1 / task-id 0

- `2` = `obs_center`
- `0` = `obs_left`
- `1` = `obs_right`
- `3` = `verify_target`

推荐序列：

- `2,0,1,2,3`

说明：

- 左右观察角度当前为 `±30°`
- 这组序列用于采集稳定的多视角观察数据

### L2 / task-id 1

- `4` = `prealign_grasp`
- `5` = `approach_coarse`
- `6` = `approach_fine`
- `7` = `retreat`

推荐序列：

- `2,3,4,5,6,7`

说明：

- `7` 仍保留在正式采集里，便于保留恢复动作样本

### L3 / task-id 2

- `8` = `reobserve`
- `9` = `pregrasp_servo`
- `10` = `grasp_execute`
- `11` = `lift_object`
- `12` = `transport_to_dropzone`
- `13` = `place_object`

推荐序列：

- `8,9,10,11,12,13`

说明：

- 当前基线下 `9/10` 已重新对齐，能更接近目标
- `11/12` 会保持闭合，不会提前松手
- `13` 会先下放再释放
- 正式 L3 采集默认使用 `configs/deployment_l3.yaml`，给每个 primitive 更长的等待时间

## 目标摆放位置

先定义一个中心基准位置，然后围绕这个位置做少量平移：

- `P1`：中心
- `P2`：左偏
- `P3`：右偏
- `P4`：前偏
- `P5`：后偏

推荐移动幅度：

- 左右方向：`3-5 cm`
- 前后方向：`3-5 cm`

除非物体在移动时自然发生偏转，否则不需要额外控制目标朝向。

## 数据量目标

本版本的数据量目标是同时支持：

- 微调 / 适配
- 离线测试
- L3 正式分析与回放检查

推荐的第一轮正式数据规模：

- L1：5 个位置 x 10 次 repeat
- L2：5 个位置 x 6 次 repeat
- L3：5 个位置 x 4 次 repeat

粗略 transition 数量：

- L1：`5 x 10 x 5 = 250`
- L2：`5 x 6 x 6 = 180`
- L3：`5 x 4 x 6 = 120`

总计约：

- `550` 条 transition

## 采集执行表

| 阶段 | 批次名 | 任务 ID | 物品如何摆放 | 是否需要移动目标 | 如何移动目标 | 采集执行指令 |
|---|---|---:|---|---|---|---|
| 0 | `smoke_verify` | 0 | 将目标放在中心基准位置，确保前臂相机能稳定看到目标。 | 否 | 不移动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/smoke_verify.npz --primitives 2,3 --repeats 1 --task-id 0` |
| 1 | `v1_verify_p1_center` | 0 | 将目标放在中心基准位置。 | 否 | 10 次 repeat 全程保持不动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p1_center.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p2_left` | 0 | 将目标放在中心左侧少量偏移的位置，并保持完全可见。 | 是 | 开始前仅移动 1 次，相对中心左移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p2_left.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p3_right` | 0 | 将目标放在中心右侧少量偏移的位置，并保持完全可见。 | 是 | 开始前仅移动 1 次，相对中心右移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p3_right.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p4_front` | 0 | 将目标放在中心前侧少量偏移的位置，并保持清晰可见。 | 是 | 开始前仅移动 1 次，相对中心前移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p4_front.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p5_back` | 0 | 将目标放在中心后侧少量偏移的位置，并保持清晰可见。 | 是 | 开始前仅移动 1 次，相对中心后移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p5_back.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 2 | `v2_approach_p1_center` | 1 | 将目标放在中心基准位置，并保证接近路径无遮挡。 | 否 | 6 次 repeat 全程保持不动。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p1_center.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p2_left` | 1 | 将目标放在中心左侧少量偏移的位置，并保证可安全接近。 | 是 | 开始前仅移动 1 次，相对中心左移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p2_left.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p3_right` | 1 | 将目标放在中心右侧少量偏移的位置，并保证可安全接近。 | 是 | 开始前仅移动 1 次，相对中心右移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p3_right.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p4_front` | 1 | 将目标放在中心前侧少量偏移的位置，并保证可安全接近。 | 是 | 开始前仅移动 1 次，相对中心前移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p4_front.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p5_back` | 1 | 将目标放在中心后侧少量偏移的位置，并保证可安全接近。 | 是 | 开始前仅移动 1 次，相对中心后移 `3-5 cm`。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p5_back.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 3 | `v3_pick_place_p1_center` | 2 | 将目标放在中心基准位置，并保证抓取到放置的完整路径安全。 | 否 | 4 次 repeat 全程保持该位置；若物体被移动，需在每次 repeat 后复位。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --output data/real/v3_pick_place_p1_center.npz --primitives 8,9,10,11,12,13 --repeats 4 --task-id 2` |
| 3 | `v3_pick_place_p2_left` | 2 | 将目标放在中心左侧少量偏移的位置，并保证完整路径安全。 | 是 | 开始前仅移动 1 次，相对中心左移 `3-5 cm`；每次 repeat 后都要复位目标到同一起始姿态。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --output data/real/v3_pick_place_p2_left.npz --primitives 8,9,10,11,12,13 --repeats 4 --task-id 2` |
| 3 | `v3_pick_place_p3_right` | 2 | 将目标放在中心右侧少量偏移的位置，并保证完整路径安全。 | 是 | 开始前仅移动 1 次，相对中心右移 `3-5 cm`；每次 repeat 后都要复位目标到同一起始姿态。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --output data/real/v3_pick_place_p3_right.npz --primitives 8,9,10,11,12,13 --repeats 4 --task-id 2` |
| 3 | `v3_pick_place_p4_front` | 2 | 将目标放在中心前侧少量偏移的位置，并保证抓取和放置路径无遮挡。 | 是 | 开始前仅移动 1 次，相对中心前移 `3-5 cm`；每次 repeat 后都要复位目标到同一起始姿态。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --output data/real/v3_pick_place_p4_front.npz --primitives 8,9,10,11,12,13 --repeats 4 --task-id 2` |
| 3 | `v3_pick_place_p5_back` | 2 | 将目标放在中心后侧少量偏移的位置，并保证抓取和放置路径无遮挡。 | 是 | 开始前仅移动 1 次，相对中心后移 `3-5 cm`；每次 repeat 后都要复位目标到同一起始姿态。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --output data/real/v3_pick_place_p5_back.npz --primitives 8,9,10,11,12,13 --repeats 4 --task-id 2` |

## 推荐执行顺序

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
12. `v3_pick_place_p1_center`
13. `v3_pick_place_p2_left`
14. `v3_pick_place_p3_right`
15. `v3_pick_place_p4_front`
16. `v3_pick_place_p5_back`

## 推荐数据划分

推荐两种简单划分方式：

### 按位置划分

- L1 / L2 使用 `P1 / P2 / P3 / P4` 做微调 / 适配
- L1 / L2 保留 `P5` 做离线测试
- L3 可以保留 `P5` 作为完整 hold-out 位置

### 按 repeat 划分

- 前面的 repeat 用于微调 / 适配
- 每个位置最后 `2` 个 repeat 留给离线测试
- L3 每个位置最后 `1` 个 repeat 留给离线测试

如果你想减小训练集和测试集的分布差异，一般更推荐按 repeat 划分。

## 每批采集后的检查项

每完成一个批次，都建议执行：

1. 检查 `.npz`：

```bash
uv run python -c "import numpy as np; d=np.load('data/real/<batch_name>.npz'); print(d.files); print(d['images'].shape, d['states'].shape, d['primitive_ids'].shape)"
```

2. 检查 session 目录：

- `meta.json`
- `frames/`
- `preview.mp4`

3. 确认：

- 目标始终在画面中
- 图像质量可接受
- primitive 顺序正确
- 输出文件名与批次名一致

## L3 额外说明

- L3 现在按正式全量方案采集，不再默认只做小规模试采。
- L3 批次默认使用 `configs/deployment_l3.yaml`，不要和 L1/L2 共用同一份部署节奏配置。
- `configs/deployment_l3.yaml` 当前默认 `runtime.primitive_sleep_s: 4.0`，用于给抓取、抬升、搬运和放置留出足够完成时间。
- L3 采集在每个 repeat 开始前会先执行硬件复位（`safety.reset_before_episode: true`），因此第一步 primitive 是从标准起始位开始，而不是从上一次 repeat 的末态直接继续。
- 每个 repeat 后都要检查目标是否还在统一起始位置，如果被夹走、滚动或偏转，需手动复位。
- 蓝色放置区在同一批次内保持固定，不建议边采边改位置。
- 若某个 L3 批次连续出现异常，应先重新跑 `validate_actions.py` 对照验证，再继续采集。
