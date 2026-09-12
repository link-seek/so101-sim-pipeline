# Ch6：Franka 评测能力盘点 — LIBERO 开箱即用实战

> SO101 仿真评测教程 · 第六章
> 实战讨论：[Discussion #14 — Ch6: Franka 评测能力盘点 — LIBERO 开箱即用实战](https://github.com/link-seek/so101-sim-pipeline/discussions/14)

---

## 1. 为什么从 Franka 开始

先声明身份：**本章不是第六个新手步骤，而是一次校准实验**。Ch4/Ch5 的主角是 SO101（我们的身体、自训的模型、自家 Grid Sweep 网格）；本章三样全换——身体是 Franka、模型是官方 `smolvla_libero`、考场是 LIBERO 标准 suite。测的不是"我们行不行"，而是"我们的评测管线准不准"。

为什么拿 Franka 校准？它是 LIBERO 的默认机器人：

| 条件 | Franka | SO101 |
|------|--------|-------|
| LIBERO 原生支持 | ✅ 12 种 RoboSuite 机器人之一 | ❌ 需要自行集成 |
| 配置修改 | 零修改 | 需要改镜像 |
| 模型兼容性 | OBS 上有大量预训练模型 | 只有我们自己训练的模型 |
| 社区基线 | 完善（LIBERO 论文原版结果） | 无 |

**这一章的目标**：用 Franka 跑通 LIBERO 评测全流程，验证平台能力，建立基线。带着"对照"预期读——后面那个 47% 和 Ch4/Ch5 的不是一回事（见 §4.3 防火墙）。

---

## 2. LIBERO Benchmark 全景（概念见 Ch5 §4）

概念主场在 Ch5 §4（BDDL、episode 生成、三个 suite 的设计理念、PRO 扰动定义——定义以 harness `vla-eval==0.4.0` 源码为准，见 Ch5 §4.6）。本章只保留跑单必需的两行：

- 三个 suite：`libero_spatial`（空间）/ `libero_object`（物体）/ `libero_goal`（目标），各 10 任务；
- PRO 扰动套件命名 `{base}_{perturbation}`（如 `libero_spatial_swap`），`swap`=位置交换、`object`=新物体、`lan`=同义改写、`task`=目标重设计、`env`=环境替换——§3.3 的命令和 §4.2 的 gap 解读依赖这套命名。

---

## 3. 实战：Franka 评测流程

### 3.1 前提条件

- CodeArts 项目 `AI4test` 的访问权限（点 run 用）
- 华为云 ECS（V100 32GB，由流水线自动开关机，不用手动管）
- SWR 镜像可访问

### 3.2 CodeArts 评测流水线（交付形态）

本章的评测跑在 **`franka-eval-pipeline`**（项目 `AI4test`，`bc38ad5c`）上，三阶段与 GHA 验证版同构：

| 阶段 | 做什么 | 说明 |
|------|--------|------|
| `boot-ecs` | hcloud `BatchStartServers` 拉起 V100（`7f39cb83…`），断言 ACTIVE | 约 3 分钟 |
| `evaluate` | `swr-login` → `fetch-scripts` → `run-eval` → `upload-obs` → `verdict`，下沉 V100 自定义执行机 | 见下 |
| `stop-ecs` | 关机兜底 | 失败 run 是否关机待验证（run #9/#11 失败后 ECS 仍 ACTIVE，已人工关机，见 #19） |

触发方式（二选一）：

```bash
# 方法 1：CodeArts 控制台 → franka-eval-pipeline → 运行，填 4 个运行时参数
# 方法 2：hcloud CLI（AK 传参，不落盘）
hcloud CodeArtsPipeline RunPipeline --cli-region=cn-north-4 \
  --project_id=872f029c6e1646d19b0a3a248fbb26f8 \
  --pipeline_id=bc38ad5cbfe746c8b3d542665c387204 \
  --variables.1.name=BENCHMARKS --variables.1.value=libero_spatial \
  --variables.2.name=EPISODES_PER_TASK --variables.2.value=10 \
  --variables.3.name=MODEL_CONFIG --variables.3.value=smolvla_franka.yaml \
  --variables.4.name=RENDER_VIDEO --variables.4.value=true
```

运行时参数（点 run 时填，不改代码）：

| 参数 | 默认 | 什么时候调 |
|------|------|-----------|
| `BENCHMARKS` | `libero_spatial` | 空格分隔多选（`libero_object`/`libero_goal` 同镜像可跑；PRO 见 §3.3） |
| `EPISODES_PER_TASK` | `1` | 冒烟用 1，全量对照用 10（SmolVLA 官方协议） |
| `MODEL_CONFIG` | `smolvla_franka.yaml` | 换模型走这里（如 #18 的 Panda 变体 hardened 模型） |
| `RENDER_VIDEO` | `true` | `false` 关逐集 mp4（省时间/空间） |
| `IMAGE` | （待控制台落地） | 镜像版本参数化：digest 金丝雀验证后切默认，见 #19 |

容器内做三件事（全部可复现，见仓库 `scripts/`）：

1. `vla-eval serve` 启动 `lerobot/smolvla_libero` 模型服务（`configs/model_servers/smolvla_franka.yaml`）；
2. `vla-eval run` 原地执行 benchmark（容器内运行时 harness 跳过自身 docker 拉起）；
3. `vla-eval merge` + 诚实计分：直接读 `LIBEROBenchmark_*_aggregate.json` 统计成功/错误数——harness 即使 100 个 episode 全错也会 exit 0，**只能以 aggregate 为准**（`verdict` 步实现）。

产物归档到 `obs://so101-sim-pipeline/eval/franka_codearts/`：aggregate json + `eval_summary.json` + console log + 逐集 mp4（文件名自带 `_success`/`_fail` 后缀，成功失败无差别录制）。

> 镜像：`swr.cn-north-4.myhuaweicloud.com/link-seek/so101-eval:latest`
> （run #8 实测使用 pre-overlay 镜像；PRO 叠加后 digest `28924535`，见 §3.3）。
> GHA `franka-eval.yml` 是前期技术验证（9-04 run `33829389761`），不对外交付，此处仅作基线对照。

### 3.3 LIBERO-PRO：已支持（叠加进同一镜像）

PRO 不再需要单独镜像。`so101-eval` 已叠加 Zxy-MLlab/LIBERO-PRO 的扰动注册与数据（PR #25/#27/#28/#31，当前 digest `28924535`），`BENCHMARKS` 参数直接透传：

```bash
# PRO 冒烟（10 tasks × 1 ep），CodeArts/GHA 通用参数名
BENCHMARKS=libero_pro_mug EPISODES_PER_TASK=1
```

两处如实记录：

1. **suite 名有个上游坑**：`{base}_{perturbation}`（如 `libero_pro_env`/`libero_spatial_env`）在 fork 里没有 BDDL 目录（`problem_folder=suite` 名，查无此目录）——上游自己也跑不通这个命名。能跑的是 `with_*`/`_ood` 系，本教程用 harness 默认 suite **`libero_spatial_with_mug`**（配置见 `configs/benchmarks/libero_pro_mug.yaml`）。`libero_pro_env.yaml` 等 5 个旧配置保留在仓库，仅作命名参考。
2. **PRO 首跑成绩**（GHA，run `34710767224`，09-12）：`libero_spatial_with_mug × 1ep`，10/10 跑完，**3 成功 0 errors**。无外部基线可对照，这个数的意义是"扰动考场能开考"，robustness gap 等 CodeArts 全量后再算。CodeArts 侧 PRO 重跑待 pipeline 加 `docker pull` 落地（见 #19，否则跑的是缓存旧镜像）。

叠加法本身（一句话）：底包保持官方 X-embodiment 不动，只搬 fork 新增部分——BDDL+init（874 文件，385 个重叠逐字节一致）、suite 注册（标准条目逐项一致）、assets（`cp -rn`，2 个 scene style 例外保留官方以保护 47% 视觉基线）、对象注册（含 fork 版 `base_object.py`，容忍 `RedAlphabetSoup` 重名定义）。细节与踩坑见 #19。

**LIBERO-Plus 不在流水线里**：它要把整个 `libero` 包换成 Sylvest 的 fork（同名互斥，只能另起镜像），且与本章主线（平台可用性验证）无关——Ch5 原话"只出镜，不参演"，此处不展开。

### 3.4 配置文件结构

每个 benchmark 的 YAML 配置：

```yaml
server:
  url: "ws://localhost:8000"
output_dir: "/data/eval/results/libero_spatial"
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    subname: libero_spatial
    episodes_per_task: 10
    params:
      suite: libero_spatial
      seed: 7
      num_steps_wait: 10
```

关键参数：
- `suite`：评测哪个 suite（spatial/object/goal）
- `episodes_per_task`：每个任务跑多少 episode（**10 是 SmolVLA 官方协议**，不是 50）
- `seed`：随机种子（可复现性）
- `num_steps_wait`：等待环境稳定的时间步

> 注意：上游 `vla-eval`（含 0.5.0）没有 `robot:` 参数，类名是 `LIBEROBenchmark`。换机器人不是改 YAML 参数，而是改文件（见 Ch7 §2 实测机制）。

运行时参数调不到的东西在文件里：`seed`（7，复现性命根子，别动）、`num_steps_wait`（10）、`chunk_size`（10）/`max_batch_size`（1，§4.3 补丁 #3，动了会炸）、`video_fps`（20，pipeline 没透出）。另：`eval_vla.py` 把 HF 缓存钉在 `/data/hf-cache`（宿主机挂载、跨 run 持久）——此前每次 run 全量重下模型，600s serve 健康检查成抛硬币（run #9 实证），修后首跑下载一次、之后常驻。

---

## 4. 结果解读

### 4.1 LIBERO 标准指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `overall_success` | 所有任务成功率的均值 | 综合能力 |
| `per_task_success` | 每个任务的成功率 | 任务级表现 |
| `pc_success` | 按 episode 维度的成功率 | 统计精度 |

LIBERO 论文原版结果（参考值）：

| 模型 | libero_spatial | libero_object | libero_goal |
|------|---------------|---------------|-------------|
| GPT-4V + robotic pipeline | 0% | 0% | 0% |
| OpenVLA (7B) | 32% | 28% | 18% |
| π₀ (3B) | 78% | 72% | 65% |

### 4.2 LIBERO-PRO 指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `robustness_gap` | 原始成功率 − 扰动成功率 | 扰动敏感度 |
| `relative_gap` | gap / 原始成功率 | 相对退化比例 |

gap = 0 说明鲁棒，gap 大说明脆弱。一个策略可以 LIBERO 80% 但 LIBERO-PRO gap 40%，意味着泛化能力有但鲁棒性差。

### 4.3 实测基线（本教程真实跑分）

模型 `lerobot/smolvla_libero`，`libero_spatial` 10 tasks × 10 eps = **100 eps**，V100，2026-09-04（Actions run `33829389761`）：

| 指标 | 值 |
|------|-----|
| 成功率 | **47/100（47%）** |
| harness 错误数 | **0**（100 个 episode 全部正常 rollout，无异常） |
| 单 suite 耗时 | ~2h（V100） |

10 个任务全部 10/10 跑完，没有任何 `failure_reason: exception`——说明**评测管线本身是健康的**，47% 是策略的真实表现，不是框架 bug。

### 4.4 全量对照：CodeArts vs GHA（同一协议）

同一模型（`lerobot/smolvla_libero`）、同一协议（`libero_spatial` 10 tasks × 10 eps = 100 eps，seed 7）：

| 渠道 | run | 成功率 | errors | 耗时（V100） |
|------|-----|--------|--------|--------------|
| GHA `franka-eval.yml` | `33829389761`（09-04） | **47/100（47%）** | 0 | ~2h |
| CodeArts `franka-eval-pipeline` | run #8（09-12） | **44/100（44%）** | 0 | ~2h |

差 3pp，在 100eps 噪声带内（~5pp）——**复刻通过**。两边 0 errors，管线健康结论与 §4.3 一致。CodeArts 侧另有冒烟记录：run #7（7/10）、run #10（8/10），同属噪声带。

> 严谨性备注：run #8 用的是 PRO 叠加前的镜像；叠加只加文件（标准 suite 的 map 条目逐项一致、重叠 BDDL逐字节一致），标准行为按构造不变，GHA PRO-mug 烟（§3.3）已证明新镜像 env 构建+rollout 正常；新镜像上的标准全量重跑待 pipeline 加 `docker pull` 后补（见 #19）。

> **防火墙：别和另一对 47/45 混淆**——本教程三个相近数字，分属两把尺子：
>
> | 数字 | 模型 | 尺子 | 出处 |
> |------|------|------|------|
> | 153/325 = 47% | 我们的 SO101 sim twin | Grid Sweep 网格 | Ch4/Ch5 |
> | 145/325 = 45% | 同上，续到 20K | **同一把尺子**（掉 8 个 episode，噪声带内） | Ch4 尾声 |
> | 47/100 = 47% | 官方 `smolvla_libero` | LIBERO spatial 考场，Franka 身体 | **本节** |
>
> 前两个是一对（同模型同网格，讲"加步数没用"）；后一个是孤立的基线（讲"管线是通的"）。47 撞 47 是巧合。

与官方 ~90% 的差距说明（诚实记录，未掩盖）：
- 官方数字的渲染/种子/解码配置与本流水线不完全一致（EGL 离屏渲染、seed 分布、`chunk_size=10`/`max_batch_size=1` 均为本工程选择）；
- 兼容层（见下）恢复了 API 调用，但随机数流与 1.4 时代不完全相同，任务初始分布有偏移；
- 本基线的价值是**可复现的起点分数**，不是 SOTA 复刻。后续工作可逐项对齐（渲染分辨率、温度/采样、seed 协议）。

> **开箱即用是个神话**：即使 Franka 是 LIBERO 默认机器人，2026 年的依赖组合（`vla-eval 0.4.0` + `robosuite 1.5.2` + LIBERO 快照）也跑不通，harness 全系（≤0.5.0）写的是 1.4 时代 API。本教程用**文件级兼容补丁**（不改 harness、不重建镜像）修了三处，全部挂载覆盖、断言防漂：
>
> | # | 现象 | 根因 | 补丁位置 |
> |---|------|------|----------|
> | 1 | `TypeError: 'NoneType' object is not callable`（100/100） | 1.5 把 `MujocoEnv.seed()` 方法改成了构造参数（实例属性 `None`），LIBERO 仍按方法调用 | `scripts/patch_env_wrapper.py` → `ControlEnv.seed` 改写为 `self.env.seed = seed` + 重建 `rng` |
> | 2 | `AttributeError: 'FixedBaseRobot' object has no attribute 'controller'` | 1.5 用 composite/part controllers 重构，删掉了 `robot.controller` | `scripts/patch_robosuite.py` → `FixedBaseRobot.controller` 兼容属性（`ee_pos`/`ee_ori_mat` 读 sim site 真值） |
> | 3 | `Server error: must override predict_batch() to use max_batch_size > 1` | 挂载的 lerobot server 脚本无 batch 实现 | `configs/model_servers/smolvla_franka.yaml`：`max_batch_size: 8 → 1`（同步评测不需要 batch） |
>
> 另有两个工程教训：`sitecustomize` 绝不能写 stdout（glfw 版本检查会 `eval()` 捕获到的 stdout，一行打印杀死全部 episode）；`vla-eval run` 即使全错也 exit 0，流水线必须以 aggregate 计分并在 errors>0 时判失败。

---

## 5. 这告诉我们什么

### 5.1 从 Franka 到 SO101 的路径

Franka 评测验证了平台的**评测管线**是通的——配置现成、镜像已构建、模型有预训练、结果有社区基线对照。SO101 这四样全缺，差的不是管线，是**机器人适配**：SO101 不是 LIBERO 的原生机器人。

Ch7 将展示：RoboSuite 已有 12 种机器人，改配置就能换。Ch8 将展示：SO101 作为自定义机器人，需要改镜像才能集成。

> **本章 47% 的使命到此结束**：它只证明管线准，不证明我们模型行。我们模型行不行（SO101 身体、我们的数据），看 Ch7 §2.4 和 Ch8。

### 5.2 评测规模参考（实测）

| 评测范围 | episodes | V100 实测耗时 |
|----------|----------|---------------|
| `libero_spatial` 10eps/task | 100 | ~2h（含 serve 启动 ~10min） |
| LIBERO 3 suites（同规模外推） | 300 | ~6h |
| 冒烟（1ep/task） | 10 | ~15min |
| CodeArts 冒烟（run #7/#10） | 10 | ~25min（含开关机） |
| CodeArts 全量（run #8） | 100 | ~2h |
| PRO 冒烟 `libero_spatial_with_mug`（GHA） | 10 | ~25min |

> 旧版表格中的 500eps/suite（~8h）是按 `episodes_per_task=50` 估算的；官方 SmolVLA 协议是 10eps/task，上表以实测为准。

---

## 思考题

1. **harness 全错也会 exit 0，为什么流水线必须以 aggregate 计分？**  
   提示：见 §3.2 第 3 步——`vla-eval merge` 后读 `LIBEROBenchmark_*_aggregate.json`，并在 errors>0 时判失败。exit code 信不得。

2. **三处兼容补丁里，哪处最脆、为什么？**  
   提示：见 §4.3 补丁表。想想哪个补丁依赖的是"读 sim 真值"这种实现细节，哪处是纯参数。

3. **为什么是 10eps/task 而不是 50？**  
   提示：10 是 SmolVLA 官方协议（见 §3.4 `episodes_per_task`）。50 跑不起吗？看 §5.2 的耗时表算一笔账。

4. **如果要在 SO101 上评测 LIBERO，需要解决什么问题？**  
   提示：SO101 不是 RoboSuite 原生机器人，需要添加机器人定义（XML + Python 类），适配 BDDL 任务文件，修改评测镜像。详见 Ch8。

---

> **上一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md) | **下一章**：[Ch7 RoboSuite 机器人扩展](so101-tutorial-ch7-robosuite-extension.md)
