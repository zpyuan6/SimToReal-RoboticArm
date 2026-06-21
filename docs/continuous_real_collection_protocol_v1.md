# 连续真实数据采集执行协议 v1

日期：2026-06-17

本文档按真实数据采集时的执行顺序编写。采集时从上到下执行即可。

这套协议用于当前冻结版 ACT/Diffusion 连续控制策略的 sim-to-real adapter。它替代旧的 primitive-transition 采集路径。

当前采集入口也可从 `docs/REAL_DATA_COLLECTION_CURRENT.md` 查看。

旧 primitive-transition 采集文件已经移到 `archive/deprecated_real_collection/`，不要从归档目录中运行它们采集当前数据。

这些归档文件会生成 `data/real_v2` 下的 primitive-transition 数据，字段包括 `primitive_ids`、`stage_ids`、`states`、`next_states`，不能直接用于当前 ACT/Diffusion continuous adapter。

## 0. 先理解这套采集在做什么

采集方式：

1. 操作者按当前步骤要求摆放目标物体。
2. 采集脚本控制真实机械臂执行一组 waypoint。
3. 脚本在每个小动作前后记录相机图像、关节状态和动作。
4. 每条 episode 结束后，操作者现场判断 `keep` 或 `redo`。

动作来源：

- 高层 waypoint 是预定义的，例如 `obs_center`、`approach`、`pregrasp`、`lift`、`transport`。
- 两个 waypoint 之间的连续小步由脚本按关节空间线性插值计算。
- `grasp_close` 只关闭夹爪。
- `retreat` 是在当前关节状态上加一个固定退出偏移。

当前 v1 不做在线视觉识别，不根据真实图像实时求解目标位置，也不是让 ACT/Diffusion 在线控制机器人。目标位置的可控性来自操作者按协议摆放目标，以及这些真实 waypoint 已经提前验证过。

## 1. 固定真实场景

在开始任何数据采集前，先固定以下内容：

- 机械臂底座位置。
- 相机安装位置。
- 桌面位置。
- 蓝色放置区域位置。
- 目标物体类型，当前为 `wire_ear_cork`。

整个 v1 采集期间不要移动这些内容。如果相机、机械臂底座、桌面或蓝色区域移动过，应停止当前 v1 采集，重新建立新的版本化采集计划。

## 2. 安全预检

场景：

- 桌面清空或只保留不会碰撞的物体。
- 机械臂处于安全 reset 区域。
- 急停手段可用。
- 相机画面可见且清晰。

执行指令：

```bash
.venv/bin/python scripts/validate_actions.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --task level3_pick_place --primitives 2,3,5,6,7,8,9
```

执行过程中会发生什么：

- 脚本会按 primitive 编号测试基础动作。
- 你需要观察关节方向、夹爪开合方向、相机画面和安全距离。

本步如何验证：

- 关节实际运动方向与预期一致。
- 夹爪打开和关闭方向正确。
- reset 位姿安全。
- 相机没有严重模糊、黑屏、曝光异常。
- 任何方向不确定时，不进入后续采集。

## 3. Dry-run 检查采集脚本

场景：

- 不需要连接真实相机或机械臂。
- 这一步只检查采集计划、插值、保存文件和合并逻辑。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session calib_l1_center --dry-run --auto-start --auto-accept --repeats 1
```

执行过程中会发生什么：

- 脚本使用虚拟相机和虚拟机械臂。
- 生成一个假的 `calib_l1_center` session。
- 保存 `session_dataset_joint_target.npz` 和 `session_dataset_joint_delta.npz`。

本步如何验证：

- 命令正常结束。
- 输出中出现 `episodes_collected` 和 `transitions_collected`。
- 如果 dry-run 输出在真实数据目录下，正式采集前必须删除或移动，避免混入真实数据。

## 4. 真实采集通用规则

所有真实 session 都遵循同一个 episode 流程：

1. 按当前步骤摆放目标。
2. 运行该步骤的采集命令。
3. 每条 episode 开始前，脚本提示按 Enter 执行。
4. 机械臂 reset 到 home。
5. 脚本执行当前 sequence 的 waypoint。
6. 每个插值小步都会保存 before 图像、动作、after 图像。
7. episode 结束后，选择 `keep`、`redo` 或 `quit`。

当前计划使用快速 pilot transition 采集规模，约为原论文级计划的 1/5：

```text
configs/continuous_real_collection_plan_v1.yaml -> shared.repeats: 2
configs/continuous_real_collection_plan_v1.yaml -> calib_l3_center.repeats: 3
configs/continuous_real_collection_plan_v1.yaml -> heldout_l3_offset.repeats: 3
```

因此 L1/L2 每个 sequence 采集 2 次 repeat，L3 每个 sequence 采集 3 次 repeat。

每一次 repeat 都应轻微调整目标位置或朝向：

- calibration 的扰动要小，目标仍保持在中心可达区域内。
- held-out 的扰动要覆盖偏左、偏右、偏前和轻微旋转。
- L2/L3 的扰动不能破坏夹爪接近和抓取条件。
- 不要连续采完全相同的目标摆放，否则数据多样性不足。
- 如果某次扰动导致明显失败，选择 `redo`，把目标移回可达但不同于上一条 episode 的位置。

通用 redo 规则：

- 相机掉帧、严重模糊或曝光异常，redo。
- 运动过程中有人手留在画面中，redo。
- 目标离开相机视野，redo。
- 机械臂运动方向明显错误，redo。
- 发生碰撞、刮擦桌面或不安全运动，redo。
- L2 夹爪没有接近目标，redo。
- L3 没有夹住、没有抬起、没有移动到蓝色区域附近或没有释放，redo。

小而干净的数据集优先于大而混杂的数据集。

每个 session 采完后，必须先用审核脚本查看真实保存的数据，再进入下一步：

1. 找到采集命令输出的 `saved_continuous_real_session=...` 目录。
2. 运行 `scripts/inspect_continuous_real_dataset.py`，生成 PNG 审核页和 `index.html`。
3. 打开 `index.html` 或逐张查看 `episode_*/page_*.png`。
4. 确认 before/after 图像、`waypoint_name`、`q_before/q_after` 和动作幅度符合当前步骤预期。
5. 打开 `meta.json`，确认 `episodes_collected` 和 `transitions_collected` 不为 0。
6. 如果发现错误，不要继续采下一步；应删除或隔离该 session，并重新采当前 session。

这一检查不能只依赖现场肉眼观察。现场观察用于决定单条 episode 的 `keep/redo`，而 session 完成后的审核脚本用于确认真正写入磁盘的数据是否正确。

## 5. 第一步：采集 L1 标定集 `calib_l1_center`

场景：

- 任务：`level1_verify`
- 数据角色：calibration
- 摆放类型：`center_band`
- 目标：采集标准中心区域摆放下的观测对准数据。

如何摆放目标：

- 将目标放在 `obs_center` 视角的中间三分之一区域。
- 目标从 `obs_left` 和 `obs_right` 视角也应可见。
- 不要把目标放入蓝色区域。
- 不需要让夹爪接触目标。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session calib_l1_center --operator "<name>"
```

执行过程中会发生什么：

- 脚本会采集三个 sequence：
  - `obs_left -> obs_center`
  - `obs_right -> obs_center`
  - `obs_center`
- 每个 sequence 默认重复 2 次。
- 每条 episode 开始前会 reset 到 home。
- 动作是预定义观测 waypoint 之间的关节插值。
- 脚本保存 L1 的真实图像、关节状态、`joint_target` 和 `joint_delta`。

本步如何验证：

先运行审核脚本，`<session_dir>` 替换为采集命令输出的 `saved_continuous_real_session`：

```bash
.venv/bin/python scripts/inspect_continuous_real_dataset.py --session-dir <session_dir> --output-dir results/real_collection_audit/calib_l1_center
```

- `obs_left -> obs_center` 中，画面应从偏左观察回到中心观察。
- `obs_right -> obs_center` 中，画面应从偏右观察回到中心观察。
- `obs_center` 不应出现异常大运动。
- 目标在整个 L1 过程中保持可见。
- episode 结束后，如果画面和 waypoint 明显不对应，选择 `redo`。

## 6. 第二步：采集 L2 标定集 `calib_l2_center`

场景：

- 任务：`level2_approach`
- 数据角色：calibration
- 摆放类型：`center_band`
- 目标：采集从观察位姿接近目标、到达预抓取位姿的数据。

如何摆放目标：

- 使用和 L1 标定集相同的中心区域摆放。
- 目标应在夹爪外侧。
- 开始时夹爪不能已经接触目标。
- 目标耳朵区域应清楚可见。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session calib_l2_center --operator "<name>"
```

执行过程中会发生什么：

- 脚本会采集三个 sequence：
  - `obs_center -> approach -> pregrasp`
  - `obs_left -> obs_center -> approach -> pregrasp`
  - `obs_right -> obs_center -> approach -> pregrasp`
- 每个 sequence 默认重复 2 次。
- 动作是预定义观察、接近、预抓取 waypoint 之间的关节插值。
- 脚本不会在线计算目标中心，目标需要按协议放在这些 waypoint 可接近的位置。

本步如何验证：

先运行审核脚本：

```bash
.venv/bin/python scripts/inspect_continuous_real_dataset.py --session-dir <session_dir> --output-dir results/real_collection_audit/calib_l2_center
```

- 从 `obs_center` 到 `approach` 时，夹爪应逐步靠近目标。
- 到 `pregrasp` 时，夹爪中心应接近目标耳朵中心附近。
- 夹爪不应明显从目标旁边偏离。
- 夹爪不应在开始时已经压住或碰撞目标。
- 如果 L2 最终位置明显无法形成后续抓取，选择 `redo`。

## 7. 第三步：采集 L3 标定集 `calib_l3_center`

场景：

- 任务：`level3_pick_place`
- 数据角色：calibration
- 摆放类型：`center_band`
- 目标：采集完整抓取、抬起、移动、释放到蓝色区域的数据。

如何摆放目标：

- 使用可达的中心区域目标摆放。
- 目标不能在蓝色区域内。
- 蓝色区域必须固定且可见。
- 目标耳朵应处在夹爪可以闭合夹住的位置。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session calib_l3_center --operator "<name>"
```

执行过程中会发生什么：

- 脚本会采集两个 sequence：
  - `obs_center -> approach -> pregrasp -> grasp_close -> lift -> transport -> place_release -> retreat`
  - `obs_left -> obs_center -> approach -> pregrasp -> grasp_close -> lift -> transport -> place_release -> retreat`
- 每个 sequence 默认重复 3 次。
- `grasp_close` 会在当前位姿直接闭合夹爪。
- `lift` 抬起物体。
- `transport` 移动到蓝色区域上方。
- `place_release` 到释放位姿。
- `retreat` 从释放后位姿退出。

本步如何验证：

先运行审核脚本：

```bash
.venv/bin/python scripts/inspect_continuous_real_dataset.py --session-dir <session_dir> --output-dir results/real_collection_audit/calib_l3_center
```

- `pregrasp` 后夹爪应处在能夹住目标的位置。
- `grasp_close` 后目标应被夹住。
- `lift` 后目标应明显离开桌面。
- `transport` 后目标应在蓝色区域上方附近。
- `place_release` 后目标应落在蓝色区域附近。
- `retreat` 不应撞到目标或桌面。
- 如果闭合在空中、没有抬起、运输时掉落或释放远离蓝色区域，选择 `redo`。

## 8. 第四步：采集 L1 held-out 集 `heldout_l1_offset`

场景：

- 任务：`level1_verify`
- 数据角色：heldout
- 摆放类型：`front_left_front_right`
- 目标：采集不同于标定集摆放位置的 L1 观测数据。

如何摆放目标：

- 不要复用 L1 标定集的精确摆放位置。
- 使用偏左一个物体宽度、偏右一个物体宽度，或向机械臂方向靠近一个物体长度的摆放。
- 目标仍必须完整可见且可达。
- 目标不能在蓝色区域内。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session heldout_l1_offset --operator "<name>"
```

执行过程中会发生什么：

- 脚本执行与 L1 标定集相同的 waypoint 结构：
  - `obs_left -> obs_center`
  - `obs_right -> obs_center`
  - `obs_center`
- 区别是目标摆放使用 held-out 偏移位置。

本步如何验证：

先运行审核脚本：

```bash
.venv/bin/python scripts/inspect_continuous_real_dataset.py --session-dir <session_dir> --output-dir results/real_collection_audit/heldout_l1_offset
```

- 目标位置确实不同于标定集。
- 目标在 `obs_left`、`obs_right`、`obs_center` 中都可见。
- 运动仍然安全。
- 如果目标被遮挡或离开画面，选择 `redo`。

## 9. 第五步：采集 L2 held-out 集 `heldout_l2_offset`

场景：

- 任务：`level2_approach`
- 数据角色：heldout
- 摆放类型：`front_left_front_right`
- 目标：采集偏移摆放下的 L2 接近目标数据。

如何摆放目标：

- 使用 held-out L1 的偏移摆放逻辑。
- 在不同 repeat 中交替使用偏左、偏右和偏前摆放。
- 目标必须仍能被 `approach -> pregrasp` 接近。
- 开始时夹爪不能已经接触目标。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session heldout_l2_offset --operator "<name>"
```

执行过程中会发生什么：

- 脚本执行与 L2 标定集相同的 waypoint 结构：
  - `obs_center -> approach -> pregrasp`
  - `obs_left -> obs_center -> approach -> pregrasp`
  - `obs_right -> obs_center -> approach -> pregrasp`
- 区别是目标摆放使用 held-out 偏移位置。

本步如何验证：

先运行审核脚本：

```bash
.venv/bin/python scripts/inspect_continuous_real_dataset.py --session-dir <session_dir> --output-dir results/real_collection_audit/heldout_l2_offset
```

- 夹爪仍能接近目标耳朵中心附近。
- 偏移不能大到让预定义 waypoint 完全无法接近。
- 如果最终夹爪明显远离目标，选择 `redo`，并重新摆放到可达的 held-out 偏移位置。

## 10. 第六步：采集 L3 held-out 集 `heldout_l3_offset`

场景：

- 任务：`level3_pick_place`
- 数据角色：heldout
- 摆放类型：`front_left_front_right`
- 目标：采集偏移摆放下的完整 pick-and-place 数据。

如何摆放目标：

- 使用 calibration L3 中没有出现过的偏前、偏左或偏右摆放。
- 蓝色区域保持固定。
- 目标必须仍能被夹爪可靠夹住。
- 不要为了 held-out 而把目标放到明显不可达的位置。

执行指令：

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session heldout_l3_offset --operator "<name>"
```

执行过程中会发生什么：

- 脚本执行与 L3 标定集相同的 waypoint 结构：
  - `obs_center -> approach -> pregrasp -> grasp_close -> lift -> transport -> place_release -> retreat`
  - `obs_left -> obs_center -> approach -> pregrasp -> grasp_close -> lift -> transport -> place_release -> retreat`
- 区别是目标摆放使用 held-out 偏移位置。

本步如何验证：

先运行审核脚本：

```bash
.venv/bin/python scripts/inspect_continuous_real_dataset.py --session-dir <session_dir> --output-dir results/real_collection_audit/heldout_l3_offset
```

- 目标摆放不同于 calibration L3。
- `grasp_close` 后目标应被夹住。
- `lift` 后目标应离开桌面。
- 运输过程中不应掉落。
- 释放后目标应落在蓝色区域附近。
- 出现明显抓取失败或不安全碰撞，选择 `redo`。

## 11. 合并采集数据

场景：

- 六个 session 都采集完成。
- 已确认没有 dry-run session 混在 `data/real_continuous_v1/sessions` 中。

执行指令：

```bash
.venv/bin/python scripts/merge_continuous_real_sessions.py --root data/real_continuous_v1/sessions --output-dir data/real_continuous_v1/merged
```

执行过程中会发生什么：

- 合并脚本按 `split_role` 分为 calibration 和 heldout。
- 按动作表示分为 joint target 和 joint delta。
- 输出四个合并文件。

预期输出：

- `data/real_continuous_v1/merged/calibration_joint_target.npz`
- `data/real_continuous_v1/merged/calibration_joint_delta.npz`
- `data/real_continuous_v1/merged/heldout_joint_target.npz`
- `data/real_continuous_v1/merged/heldout_joint_delta.npz`

使用方式：

- ACT 使用 `*_joint_target.npz`。
- Diffusion 使用 `*_joint_delta.npz`。

## 12. 合并后结构检查

执行指令：

```bash
.venv/bin/python -c "import numpy as np; p='data/real_continuous_v1/merged/calibration_joint_delta.npz'; z=np.load(p, allow_pickle=True); print({k:z[k].shape for k in ['images','proprio','actions','tasks','episode_ids','step_ids','next_images','next_proprio']}); print('max_abs_delta=', abs(z['action_joint_delta']).max(axis=0)); print('tasks=', sorted(set(z['tasks'].tolist())))"
```

预期结果：

- `images` 和 `next_images` 形状为 `N x 224 x 224 x 3`。
- `proprio` 和 `next_proprio` 形状为 `N x 16`。
- `actions` 形状为 `N x 6`。
- `N > 0`。
- `tasks` 包含对应任务 id。
- `action_joint_delta` 不应出现异常巨大跳变。

如果检查失败：

- 先确认是否混入 dry-run 数据。
- 再确认六个 session 是否都采集完成。
- 检查是否把 ACT 和 Diffusion 的动作表示用反。
- 检查采集期间相机或机械臂底座是否移动过。

## 13. 每个 session 输出内容

每个 session 会保存：

- `session_dataset_joint_target.npz`
- `session_dataset_joint_delta.npz`
- `session_dataset.npz`
- `meta.json`
- `preview.mp4`
- `frames/`

每个 `.npz` 包含：

- `images`：动作前图像。
- `next_images`：动作后图像。
- `proprio`：16 维状态，`qpos6 + qvel6 + task_one_hot3 + progress1`。
- `next_proprio`：动作后 16 维状态。
- `actions`：当前文件对应动作。
- `q_before`：动作前 6 维关节位姿。
- `q_after`：动作后 6 维关节位姿。
- `action_joint_target`：绝对关节目标，即 `q_after`。
- `action_joint_delta`：关节增量，即 `q_after - q_before`。
- `tasks`
- `episode_ids`
- `step_ids`
- `task_text`
- `waypoint_name`

`session_dataset_joint_target.npz` 和 `session_dataset_joint_delta.npz` 来自同一次真实物理运行。它们的图像和状态相同，只有 `actions` 字段不同。

## 14. 论文级真实实验建议数据量

真实环境实验需要两类数据，不能混用：

1. real calibration/heldout transition 数据：用于训练 adapter、做离线 transition 检查。
2. real policy rollout 数据：用于最终论文成功率结论。

当前 `configs/continuous_real_collection_plan_v1.yaml` 默认设置为快速 pilot transition 采集规模，约为论文级完整计划的 1/5：

- calibration transition：每个任务 6 episodes。
- heldout transition：每个任务 6 episodes。
- 总计：36 episodes。

按当前 sequence 设计，对应设置为：

| session | repeats | expected episodes | expected transitions |
|---|---:|---:|---:|
| `calib_l1_center` | 2 | 6 | 28 |
| `calib_l2_center` | 2 | 6 | 58 |
| `calib_l3_center` | 3 | 6 | 132 |
| `heldout_l1_offset` | 2 | 6 | 28 |
| `heldout_l2_offset` | 2 | 6 | 58 |
| `heldout_l3_offset` | 3 | 6 | 132 |

总计：

- calibration：18 episodes，218 transitions。
- heldout：18 episodes，218 transitions。
- 全部 transition 数据：36 episodes，436 transitions。
- 每条真实 transition 同时保存 `joint_target` 和 `joint_delta` 两种动作表示。

这些数据可以支持：

- ACT 的真实 calibration adapter 训练，因为 ACT 使用 `joint_target`。
- Diffusion 的真实 calibration adapter 训练，因为 Diffusion 使用 `joint_delta`。
- calibration split 上的适配训练和诊断。
- heldout split 上的离线 transition 检查。
- 真实采集质量审核，包括图像、waypoint、关节状态和动作幅度。

这些数据仍然不能单独支持：

- 最终真实闭环成功率。
- 多方法真实环境成功率对比，例如 `no_adaptation`、`few_shot_finetuning`、`static_adapter`、`probe_feature_alignment`、`ours` 的真实成功率表。
- 论文级统计结论。

原因是当前数据是脚本化采集的 calibration/heldout transition，不是各个模型和方法在真实环境中自主执行后的 rollout 结果。它能训练和检查 adapter，但不能替代真实策略执行评估。

### 14.1 推荐的最终真实 rollout 规模

最终论文结论应来自真实闭环 rollout。建议最低规模为：

- 每个 `model-method-task` 30 次真实 rollout。
- 如果结果差距小于 15 个百分点，或需要更强统计可信度，将相关对比追加到 50 次真实 rollout。

当前主要模型和方法如果按完整对比计算：

- models：ACT、Diffusion，共 2 个。
- methods：`no_adaptation`、`few_shot_finetuning`、`static_adapter`、`probe_feature_alignment`、`ours`，共 5 个。
- tasks：L1、L2、L3，共 3 个。

则真实 rollout 数量为：

| 每个 model-method-task 次数 | 总真实 rollout 数 |
|---:|---:|
| 10 | 300 |
| 30 | 900 |
| 50 | 1500 |

建议采用：

- pilot：10 次/格，共 300 次，用于检查真实执行流程、失败类型和安全性。
- 正式结果：30 次/格，共 900 次，用于论文主表。
- 追加验证：只对 `ours` 与最强 baseline 的接近结果追加到 50 次/格。

如果后续把 VLA 也纳入同一张真实环境主表，模型数变为 3 个，对应总真实 rollout 为：

| 每个 model-method-task 次数 | 总真实 rollout 数 |
|---:|---:|
| 10 | 450 |
| 30 | 1350 |
| 50 | 2250 |

### 14.2 rollout 摆放分布

每个 `model-method-task` 的 30 次 rollout 不应全是同一个位置。建议分成 5 个摆放 bin，每个 bin 6 次：

- center：中心可达位置。
- left：相对中心偏左。
- right：相对中心偏右。
- front：相对中心向机械臂方向靠近。
- orientation/mixed：目标朝向轻微变化，仍保持可达。

L3 的所有测试摆放必须保持可抓取、可抬起、可移动到蓝色区域。不要把目标放到预定义动作或策略明显不可达的位置，否则评估的是场景越界，不是方法能力。

### 14.3 最终实验流程

1. 使用本协议采集 real calibration/heldout transition 数据。
2. 用 calibration 数据训练或适配各方法。
3. 用 heldout transition 数据做离线诊断，但不要把它当最终成功率。
4. 冻结方法和超参数。
5. 运行真实 policy rollout evaluation。
6. 记录每个模型、每种方法、每个任务的真实执行成功/失败、审核页、最终物体状态和失败原因。

最终真实测试集不能用于调参。如果根据真实 rollout 结果修改了方法、阈值、数据处理或超参数，应重新划分新的最终测试批次。

### 14.4 最低可接受版本

如果时间或硬件成本受限，最低可接受版本是：

- transition 数据：每个任务 calibration 20 episodes、heldout 20 episodes。
- rollout 数据：只比较 `no_adaptation`、最强 baseline、`ours`，每个 model-method-task 30 次。

对 ACT/Diffusion 两个模型，这个最低版本需要：

- transition：约 120 episodes。
- rollout：2 models x 3 methods x 3 tasks x 30 = 540 次真实 rollout。

低于这个规模，建议只作为 preliminary real-world validation，不建议作为 IEEE Trans 论文的主要真实环境结论。
