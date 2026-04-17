# 真实数据采集计划

本文档记录当前正式的真实数据采集方案。  
本方案默认使用已经验证稳定的基线版本：

- `stable-validation-v1`

因此，**正式数据集应当从当前稳定节点重新开始采集**。  
此前在基线尚未稳定前采到的数据，建议统一视为：

- 历史数据
- 调试数据
- 标定前数据

默认不要直接混入新的正式数据集。

## 当前适用范围

本计划假设你目前能稳定控制的环境变量主要只有：

- 桌面目标的位置

而不能稳定控制：

- 灯光
- 背景
- 其他外观环境因素

这些因素如果在采集过程中自然发生变化，可以记录在笔记中，但不作为当前版本的主动设计变量。

## 稳定基线

本计划基于已经人工确认通过的稳定节点：

- `stable-validation-v1`

在这个稳定节点上，下面这些内容已经验证通过：

- L1 动作验证
- L2 动作验证
- L3 分步动作验证
- 真实 / 虚拟初始姿态对齐
- 夹爪开合方向与幅度
- 虚拟环境桌面对齐

## 采集原则

- 从当前稳定节点开始，重新建立正式数据集
- 每个批次内目标保持不动
- 只在批次之间移动目标
- 保留原始分辨率图像
- `.npz` 继续作为主结构化数据格式
- `frames/`、`meta.json`、`preview.mp4` 用于质检和调试

## Primitive 速查

### L1 / task-id 0

- `2` = `obs_center`
- `0` = `obs_left`
- `1` = `obs_right`
- `3` = `verify_target`

推荐序列：

- `2,0,1,2,3`

### L2 / task-id 1

- `4` = `prealign_grasp`
- `5` = `approach_coarse`
- `6` = `approach_fine`
- `7` = `retreat`

推荐序列：

- `2,3,4,5,6,7`

### L3 / task-id 2

- `8` = `reobserve`
- `9` = `pregrasp_servo`
- `10` = `grasp_execute`
- `11` = `lift_object`
- `12` = `transport_to_dropzone`
- `13` = `place_object`

推荐序列：

- `8,9,10,11,12,13`

## 目标摆放位置

先定义一个中心基准位置，然后只围绕这个位置做少量平移：

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

推荐的第一轮正式数据规模：

- L1：5 个位置 x 10 次 repeat
- L2：5 个位置 x 6 次 repeat
- L3：3 个位置 x 3 次 repeat

粗略 transition 数量：

- L1：`5 x 10 x 5 = 250`
- L2：`5 x 6 x 6 = 180`
- L3：`3 x 3 x 6 = 54`

总计约：

- `484` 条 transition

这是一版比较务实、又足够支持后续实验的正式数据规模。

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
| 3 | `v3_pick_place_p1_center` | 2 | 将目标放在中心基准位置，并保证抓取到放置的完整路径安全。 | 否 | 3 次 repeat 全程保持该位置；若物体被移动，需在每次 repeat 后复位。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_pick_place_p1_center.npz --primitives 8,9,10,11,12,13 --repeats 3 --task-id 2` |
| 3 | `v3_pick_place_p2_left` | 2 | 将目标放在中心左侧少量偏移的位置，并保证完整路径安全。 | 是 | 开始前仅移动 1 次，相对中心左移 `3-5 cm`；必要时每次 repeat 后复位目标。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_pick_place_p2_left.npz --primitives 8,9,10,11,12,13 --repeats 3 --task-id 2` |
| 3 | `v3_pick_place_p3_right` | 2 | 将目标放在中心右侧少量偏移的位置，并保证完整路径安全。 | 是 | 开始前仅移动 1 次，相对中心右移 `3-5 cm`；必要时每次 repeat 后复位目标。 | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_pick_place_p3_right.npz --primitives 8,9,10,11,12,13 --repeats 3 --task-id 2` |

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

## 推荐数据划分

推荐两种简单划分方式：

### 按位置划分

- L1 / L2 使用 `P1 / P2 / P3 / P4` 做微调 / 适配
- L1 / L2 保留 `P5` 做离线测试
- L3 可以完整保留一个位置，例如 `P3` 作为离线测试

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

- 如果目标在抓取 / 放置后发生了偏移，需要在每次 repeat 后手动复位到起始位置。
- 放置区在同一批次内保持固定。
- 采集 L3 时继续使用 `stable-validation-v1` 这版稳定基线。
- 如果某个 L3 批次出现明显异常，应先暂停采集并重新验证 primitive，再继续。
