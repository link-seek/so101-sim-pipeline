# Ch6：落地 LIBERO 评测实战

> SO101 仿真评测教程 · 第六章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 详细调研：[Discussion #9 — SO101 跑 LIBERO 评测：现状分析与落地方案](https://github.com/link-seek/so101-sim-pipeline/discussions/9)

---

## 1. 为什么这一章讲 LIBERO 落地

前五章我们完成了从基础设施搭建到 PPO/VLA 训练到 Debug 实战到评测方法论的完整闭环。但评测一直停留在**单任务**层面——Grid Sweep 只测一个 pick-and-place 任务的工作空间覆盖。

**LIBERO 是 VLA 领域的标准 benchmark**（2.2k stars），提供 3 个 suite × 10 tasks 的多任务泛化评测。我们的仓库已设计好完整管线（`eval_vla.py` + 8 个 benchmark 配置 + `so101-eval` 镜像），并已于 2026-08-27 首次实战跑出 0%——直接验证了 LIBERO 只有 Franka Panda 机器人、与我们的 SO101 模型不兼容（详见 Ch5 §4.8 实战数据）。

这一章讲**如何把 SO101 机器人添加到 LIBERO 中**，让我们自己的 SO101 SmolVLA 模型能在 LIBERO 多任务框架里跑评测。

> **详细调研结论**（模型-环境不兼容分析、社区调研、标准化路径对比、旧方案废弃原因）见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。本章聚焦**实操步骤**。

---

## 2. 方案总览

### 2.1 核心思路

**把 SO101 作为自定义机器人添加到 LIBERO/RoboSuite 中**，然后用我们现有的 SO101 SmolVLA 模型直接评测。

```
旧方案（已废弃）: 在 LIBERO 数据上训练 Franka SmolVLA → 评测 Franka 模型
新方案:          在 LIBERO 中添加 SO101 机器人 → 评测我们自己的 SO101 SmolVLA
```

**新方案的核心优势**：
- 评测的是 **我们自己的 SO101 模型**（`xieyucheng123/so101-smolvla`），不是新训练的 Franka 模型
- 不需要新训练数据，可以直接 zero-shot 评测
- V100 能跑（LIBERO 用 RoboSuite/MuJoCo，不需要 Isaac Lab）

### 2.2 路线图

```
Phase 1: SO101 机器人集成 (1.5 天)
  ├── 导出 SO101 robot.xml → RoboSuite 格式
  ├── 编写 SO101 gripper (XML + Python)
  ├── 编写 MountedSO101 类 (继承 ManipulatorModel)
  ├── 配置 JOINT_POSITION 控制器
  ├── 适配 BDDL 任务文件（调整区域坐标）
  └── 集成测试
       ↓
Phase 2: Zero-shot 评测 (1 天)
  ├── 运行 LIBERO 3 suites (1,500 episodes, ~8h)
  ├── 运行 LIBERO-PRO 5 suites (2,500 episodes, ~14h)
  └── 汇总结果
       ↓
Phase 3 (可选): 训练后评测 (2 天)
  ├── 生成 SO101 LIBERO 演示数据
  ├── 微调 SmolVLA (~15h)
  └── 重新评测
       ↓
Phase 4: CI 集成 (0.5 天)
  ├── 更新 evaluate.yml workflow
  └── 结果归档到 OBS
```

**总工期**: ~3 天（zero-shot）| **V100 资源**: 22h | **成本**: ~¥660

---

## 3. Phase 1：在 LIBERO 中添加 SO101 机器人

### 3.1 LIBERO 机器人定义机制

LIBERO 基于 [RoboSuite](https://github.com/ARISE-Initiative/robosuite)，机器人通过两个东西定义：

1. **MuJoCo XML 文件** — 运动链、关节、mesh 几何
2. **Python 类** — 继承 `ManipulatorModel`，指定默认配置

**MountedPanda 示例**（`LIBERO/robots/mounted_panda.py`）：

```python
class MountedPanda(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "OSC_POSE"

    @property
    def init_qpos(self):
        return np.array([0, -0.16, 0, -2.44, 0, 2.23, np.pi/4])
```

RoboSuite 已有 12 种机器人：baxter, gr1, iiwa, jaco, kinova3, panda, sawyer, spot, spot_arm, tiago, ur5e, xarm7。**我们要添加第 13 种：SO101**。

### 3.2 步骤 1：SO101 robot.xml

SO101 的 MJCF 已有——`robot_descriptions.so_arm101_mj_description` 提供完整 XML，`dyordan1/so101-mujoco` 也在用。需要适配 RoboSuite 的目录结构：

```
robosuite/robots/so101/
  ├── robot.xml          # 主机器人 XML（引用 arm + gripper）
  ├── meshes/            # SO101 finger STL meshes
  │   ├── finger_1.stl
  │   ├── finger_2.stl
  │   └── ...
  └── __init__.py
```

```python
# 从 robot_descriptions 导出 SO101 MJCF
from robot_descriptions import so_arm101_mj_description

# robot_descriptions 提供的路径
xml_path = so_arm101_mj_description.mjcf_path
# 复制到 robosuite/robots/so101/ 并调整 <compiler> 路径
```

**关键适配点**：
- RoboSuite 期望 `<compiler>` 标签的 `meshdir` 指向 `meshes/` 子目录
- 关节命名需遵循 RoboSuite 约定：`joint0` ~ `joint4`（5 DoF arm）
- `<actuator>` 需映射到关节 + gripper

### 3.3 步骤 2：SO101 夹爪

SO101 使用平行夹爪（两个 finger），比 Panda 的夹爪更简单。需要在 RoboSuite 中定义：

```python
# robosuite/robots/grippers/so101_gripper.py
class SO101Gripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/grippers/so101_gripper.xml"), idn=idn)

    @property
    def dof(self):
        return 1  # 单自由度夹爪

    @property
    def init_qpos(self):
        return np.array([0.0])  # 张开状态
```

```xml
<!-- robosuite/robots/grippers/so101_gripper.xml -->
<mujoco model="so101_gripper">
  <worldbody>
    <body name="right_finger" pos="0 0.02 0">
      <geom name="right_finger_geom" type="box" size="0.01 0.01 0.02"/>
    </body>
    <body name="left_finger" pos="0 -0.02 0">
      <geom name="left_finger_geom" type="box" size="0.01 0.01 0.02"/>
    </body>
  </worldbody>
</mujoco>
```

### 3.4 步骤 3：MountedSO101 类

仿照 MountedPanda，编写 SO101 的机器人类：

```python
# libero/libero/robots/mounted_so101.py
import numpy as np
from robosuite.models.robots.manipulators import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class MountedSO101(ManipulatorModel):
    """SO101 机器人，5 DoF arm + parallel gripper"""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/so101/robot.xml"), idn=idn)

    @property
    def default_gripper(self):
        return "SO101Gripper"

    @property
    def default_controller_config(self):
        return "JOINT_POSITION"  # SO101 用关节位置控制

    @property
    def default_mount(self):
        return "null"  # 固定底座

    @property
    def init_qpos(self):
        # SO101 初始关节位置（5 DoF arm + 1 gripper）
        return np.array([0.0, -0.5, 0.0, -1.2, 0.0, 0.5])

    @property
    def base_xpos_offset(self):
        return {"null": np.array([0, 0, 0])}
```

**与 Panda 的关键区别**：
- 5 DoF arm（Panda 是 7 DoF）
- `JOINT_POSITION` 控制器（Panda 用 `OSC_POSE` 操作空间控制）
- `init_qpos` 是 6 维（5 arm + 1 gripper），Panda 是 8 维（7 arm + 1 gripper）

### 3.5 步骤 4：控制器配置

RoboSuite 已有 `JOINT_POSITION` 控制器类型，只需为 SO101 配置参数：

```python
# robosuite/controllers/config/so101.json
{
    "JOINT_POSITION": {
        "input_max": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.05],
        "input_min": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0],
        "output_max": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.05],
        "output_min": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0],
        "kp": 150,
        "kd": 10,
        "velocity_limits": [-1.0, 1.0]
    }
}
```

### 3.6 步骤 5：适配 BDDL 任务文件

LIBERO 任务用 BDDL 文件定义（声明式），包含 `regions`（区域坐标）、`objects`、`init`、`goal`。SO101 的臂展比 Panda 短（~0.6m vs ~0.86m），需要缩小物体放置范围：

```python
# scripts/adapt_bddl_for_so101.py
import json
from pathlib import Path

def scale_regions(bddl_path, scale=0.7):
    """将 BDDL 区域坐标缩放，适配 SO101 较短的工作空间"""
    bddl = parse_bddl(bddl_path)
    for region in bddl["regions"]:
        # 缩放 x, y 坐标（工作空间半径）
        region["x_range"] = [v * scale for v in region["x_range"]]
        region["y_range"] = [v * scale for v in region["y_range"]]
        # z 坐标保持不变（桌面高度）
    return bddl

# 批量适配所有 LIBERO 任务
for suite in ["libero_spatial", "libero_object", "libero_goal"]:
    for task_id in range(10):
        adapt_bddl_for_so101(f"libero/bddl/{suite}/task{task_id}.bddl")
```

**关键调整**：
- 缩放区域坐标 ×0.7（SO101 臂展 ≈ Panda × 0.7）
- 降低物体放置高度（SO101 基座较矮）
- 某些需要 7 自由度的任务可能不可解——跳过或放宽位姿约束

### 3.7 步骤 6：集成测试

```python
# tests/test_so101_in_libero.py
import libero.libero as libero

def test_so101_spawn():
    """验证 SO101 能在 LIBERO 中加载"""
    env = libero.get_env("libero_spatial", task_id=0, robot="so101")
    obs = env.reset()
    assert obs["robot0_joint_pos"].shape == (6,)  # 5 arm + 1 gripper

def test_so101_reach():
    """验证 SO101 能到达工作空间内的点"""
    env = libero.get_env("libero_spatial", task_id=0, robot="so101")
    env.reset()
    # 发送关节位置指令
    obs, reward, done, info = env.step(
        action=np.array([0.5, -0.3, 0.5, -1.0, 0.3, 0.0, 0.04])
    )
    assert not done  # 不应立即结束

def test_so101_pick():
    """验证 SO101 能抓取物体"""
    env = libero.get_env("libero_spatial", task_id=0, robot="so101")
    obs = env.reset()
    # 执行抓取序列...
    assert info["grasped"]
```

### 3.8 开发工作量汇总

| 步骤 | 内容 | 工作量 |
|------|------|--------|
| 1. SO101 robot.xml | 从 robot_descriptions 导出，适配 RoboSuite | 2h |
| 2. SO101 gripper | XML + Python 类 | 2h |
| 3. MountedSO101 类 | 继承 ManipulatorModel | 1h |
| 4. 控制器配置 | JOINT_POSITION 参数 | 1h |
| 5. BDDL 适配 | 缩放区域坐标 | 3h |
| 6. 集成测试 | spawn + reach + pick | 2h |
| **总计** | | **~11h** |

---

## 4. Phase 2：Zero-shot 评测

### 4.1 直接用现有模型评测

SO101 机器人集成完成后，直接复用现有 `eval_vla.py`：

```bash
# 评测我们自己的 SO101 SmolVLA 模型
python scripts/eval_vla.py \
    --model-config configs/model_servers/smolvla_so101.yaml \
    --benchmarks libero_spatial,libero_object,libero_goal \
    --robot so101  # 新增参数：指定使用 SO101 机器人
```

评测流程（`eval_vla.py` 已实现）：
1. 启动模型推理服务（model server，加载 `xieyucheng123/so101-smolvla`）
2. 逐 benchmark 运行：`vla-eval run --config configs/benchmarks/{name}.yaml`
3. 合并结果：`vla-eval merge → merged.json`
4. 汇总 success_rate

### 4.2 评测规模与耗时

| Benchmark | Suites | Tasks | Episodes | V100 时间 | 说明 |
|-----------|--------|-------|----------|----------|------|
| LIBERO | 3 | 30 | 1,500 | ~8h | 跨任务泛化 |
| LIBERO-PRO | 5 | 50 | 2,500 | ~14h | 鲁棒性 |
| **总计** | **8** | **80** | **4,000** | **~22h** | — |

**单 episode 耗时估算**：
- SmolVLA 推理: ~50ms/step（V100，SmolVLA ~450M 参数）
- 平均 episode: ~300 steps（pick-and-place 任务）
- 每 episode: 300 × 50ms + 环境开销 ≈ 20s

### 4.3 Zero-shot 预期结果

我们的 SO101 SmolVLA 只在 pick-cube 单任务上训练过，zero-shot 到 LIBERO 多任务预期较低：

| Suite | 预期成功率 | 原因 |
|-------|-----------|------|
| libero_spatial | 10-30% | 物体不同（碗/盘子 vs 方块），但空间泛化相对容易 |
| libero_object | 5-20% | 物体形状差异大 |
| libero_goal | 5-15% | 目标泛化最难 |
| libero_pro_* | 0-10% | 鲁棒性评测，zero-shot 几乎不可能 |

**Zero-shot 的价值**：建立 baseline，量化泛化差距。低分不是失败——这正是「单任务训练 → 多任务评测」泛化差距的精确测量。

### 4.4 结果格式

`eval_vla.py` 输出遵循 LeRobot 的 `per_episode` 格式：

```json
{
  "benchmarks": {
    "libero_spatial": {
      "success_rate": 0.18,
      "num_episodes": 500,
      "num_tasks": 10,
      "per_task": [
        {"task": 0, "success_rate": 0.32, "episodes": 50},
        {"task": 1, "success_rate": 0.08, "episodes": 50}
      ]
    }
  }
}
```

---

## 5. Phase 3（可选）：训练后评测

如果 zero-shot 结果太低，可以在 LIBERO SO101 演示数据上微调模型：

### 5.1 生成 SO101 LIBERO 演示数据

```bash
# 用脚本策略在 LIBERO SO101 环境中生成演示
python scripts/generate_libero_so101_demos.py \
    --suite libero_spatial \
    --robot so101 \
    --demos_per_task 50 \
    --output outputs/libero_spatial_so101_demos
```

### 5.2 微调 SmolVLA

```bash
# 在 SO101 LIBERO 演示上微调（不是从头训练）
docker run --gpus all so101-train:latest \
    python -m lerobot.scripts.train \
    --policy.path=xieyucheng123/so101-smolvla \
    --dataset.repo_id=link-seek/libero_spatial_so101_demos \
    --output_dir=outputs/smolvla_libero_finetuned \
    --steps=10000 \
    --batch_size=32
```

### 5.3 训练后评测

```bash
# 用微调后的模型评测
python scripts/eval_vla.py \
    --model-config configs/model_servers/smolvla_libero_finetuned.yaml \
    --benchmarks libero_spatial,libero_object,libero_goal \
    --robot so101
```

### 5.4 训练后预期

| Suite | Zero-shot | 训练后 | 说明 |
|-------|-----------|--------|------|
| libero_spatial | 10-30% | 60-80% | 空间泛化通过训练大幅提升 |
| libero_object | 5-20% | 50-70% | 物体泛化中等 |
| libero_goal | 5-15% | 40-60% | 目标泛化较难 |
| libero_pro_* | 0-10% | 20-40% | 鲁棒性需要更强的方法 |

---

## 6. Phase 4：CI 集成

### 6.1 更新 evaluate.yml

```yaml
# .github/workflows/evaluate.yml — 添加 SO101 LIBERO 评测
evaluate-so101-libero:
  needs: [build-eval-image]
  runs-on: [self-hosted, Linux, X64, V100]
  steps:
    - uses: actions/checkout@v4
    - name: Run SO101 LIBERO evaluation
      run: |
        docker run --gpus all so101-eval:latest \
          python scripts/eval_vla.py \
          --model-config configs/model_servers/smolvla_so101.yaml \
          --benchmarks libero_spatial,libero_object,libero_goal \
          --robot so101
    - name: Archive results
      run: |
        obsutil cp results/ obs://so101-eval/libero/$(date +%Y%m%d)/
```

### 6.2 触发评测

```bash
# 手动触发
gh workflow run evaluate.yml -f benchmark=libero -f robot=so101

# 或在 PR 中自动触发（添加 label）
gh pr edit --add-label eval-libero
```

---

## 7. 时间与成本估算

| 阶段 | 内容 | 耗时 | V100 费用 |
|------|------|------|-----------|
| Phase 1 | SO101 机器人集成 | 1.5 天（开发） | — |
| Phase 2 | Zero-shot 评测 | 22h（运行） | ¥660 |
| Phase 3 | 训练后评测（可选） | +15h 训练 + 22h 评测 | +¥1,110 |
| Phase 4 | CI 集成 | 0.5 天（开发） | — |
| **Zero-shot 路线** | | **~3 天** | **¥660** |
| **训练后路线** | | **~5 天** | **¥1,110** |

---

## 8. 评测阶梯：落地后的完整体系

```
评测阶梯 (全部 V100 可运行):

  Level 0: 回放验证 (replay_demo.py)
    └── 30s, 1 episode, smoke test

  Level 1: PPO 确定性评估 (eval_ppo.py)
    └── 15min, 50 episodes, 单任务

  Level 2: Grid Sweep (eval_mujoco_policy.py)
    └── 30min, 325 episodes, 单任务工作空间覆盖

  Level 3: LIBERO (eval_vla.py)          ← 本章落地
    └── 8h, 1500 episodes, 跨任务泛化

  Level 4: LIBERO-PRO (eval_vla.py)      ← 本章落地
    └── 14h, 2500 episodes, 鲁棒性

  Level 5: SO-101 Bench (Isaac Lab)      ← 需 RTX GPU
    └── SO101 数字孪生, 硬件不支持
```

**LIBERO 评测落地后**，我们拥有从 smoke test 到跨任务泛化到鲁棒性的完整评测阶梯，全部在 V100 上可运行。

---

## 9. 技术风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SO101 工作空间 < Panda，部分 BDDL 任务物体不可达 | 评测失败 | 调整 BDDL 区域坐标，缩小放置范围（×0.7） |
| SO101 5 DoF arm vs Panda 7 DoF，部分任务需 7 自由度 | 某些任务不可解 | 跳过不可解任务或放宽位姿约束 |
| RoboSuite 版本兼容性 | 集成失败 | 在 Docker 镜像中固定版本 |
| Zero-shot 性能极低 | 结果无参考价值 | 预期的——这正是泛化差距的量化，有 baseline 价值 |
| vla-eval harness 需适配 SO101 | 评测脚本报错 | 修改 benchmark config 指定 `robot: so101` |
| V100 显存不足 (16GB) | 评测 OOM | SmolVLA ~450M 参数，16GB 足够 |

---

## 10. 与旧方案的对比

| | 旧方案（已废弃） | 新方案 |
|--|------------------|--------|
| 核心思路 | 在 LIBERO 数据上训练 Franka 模型 | 在 LIBERO 中添加 SO101 机器人 |
| 评测的模型 | Franka SmolVLA（新训练） | **SO101 SmolVLA**（现有模型） |
| 机器人 | Franka Panda | **SO101** |
| 需要训练 | 是（12-18h） | 否（zero-shot） |
| 代码改动 | 数据转换脚本 + 训练管线 | RoboSuite 机器人定义 + BDDL 适配 |
| 总工期 | 6-9 天 | **3 天** |
| 成本 | ¥900-1200 | **¥660** |
| 与项目目标 | ❌ 评测的不是 SO101 模型 | ✅ 评测的是 SO101 模型 |

---

## 踩坑预警

### 坑 1：SO101 工作空间不足

SO101 臂展 ~0.6m，Panda ~0.86m。直接用 LIBERO 原始 BDDL 坐标，物体可能放在 SO101 够不到的地方。

**解决**：缩放区域坐标 ×0.7，并测试每个任务的可达性。

### 坑 2：6 DoF vs 7 DoF

Panda 有 7 自由度（冗余臂），某些 LIBERO 任务利用了冗余自由度来避障。SO101 只有 6 DoF，可能无法完成这些任务。

**解决**：识别并跳过需要 7 DoF 的任务，或放宽避障约束。

### 坑 3：控制器选择

Panda 用 `OSC_POSE`（操作空间控制），SO101 用 `JOINT_POSITION`（关节位置控制）。控制模式不同导致动作空间语义不同。

**解决**：确保 SmolVLA 的动作输出与 `JOINT_POSITION` 控制器匹配（我们的 SO101 模型本来就是关节位置控制，天然兼容）。

### 坑 4：相机视角差异

LIBERO 的 `agentview` 和 `robot0_eye_in_hand` 相机位置是为 Panda 配置的。SO101 的相机位置不同，需要调整。

**解决**：在 SO101 robot.xml 中定义相机，确保覆盖工作空间。

---

## 思考题

1. **为什么不能直接把 SO101 模型放到 LIBERO 的 Franka 环境里跑？**  
   提示：机器人不同——SO101（6 DoF）vs Franka（7 DoF），关节定义、观测空间、动作语义都不匹配。详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

2. **在 LIBERO 中添加 SO101 机器人后，评测的是哪个模型？**  
   提示：是我们现有的 `xieyucheng123/so101-smolvla`，在 SO101 数据上训练的 SO101 模型——不是新训练的 Franka 模型。

3. **Zero-shot 评测结果很低，有意义吗？**  
   提示：有。它量化了「单任务训练 → 多任务泛化」的差距，是后续优化的 baseline。0% 和 20% 的指导价值完全不同。

4. **SO101 6 DoF 和 Panda 7 DoF 对评测有什么影响？**  
   提示：某些 LIBERO 任务利用了 Panda 的冗余自由度避障，SO101 6 DoF 可能无法完成。需要识别并跳过这些任务。

5. **为什么选择在 LIBERO 中添加 SO101，而不是自建 SO101 多任务 benchmark？**  
   提示：LIBERO 提供标准化的任务定义、评测协议、社区基线。自建 benchmark 缺乏可比性，且工作量更大。

---

> **上一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md) | **下一章**：[附录](so101-tutorial-appendix.md)
