# 板材切割优化系统

全栈板材切割优化应用：基于 2D 装箱与可选 OR-Tools 生成切割方案，支持项目管理、Supabase 存储与排版可视化。

## 项目结构

```
Plate_Cutting/
├── frontend/                    # Next.js（App Router）
│   ├── src/app/                 # 页面：project、layout、login
│   ├── src/components/
│   ├── src/config/api.ts        # 后端 API 基址与路径
│   └── package.json
├── backend/
│   ├── api.py                   # FastAPI：/optimize、/optimize/async、健康检查
│   ├── optimization_jobs.py     # 异步优化任务注册表
│   ├── config.py                # Pydantic Settings（.env.local）
│   ├── run.py                   # Uvicorn 入口
│   ├── core/
│   │   ├── models.py            # CuttingConfig、SmallPlate、Cut、Rectangle
│   │   ├── utils.py             # DataConverter 等
│   │   └── metrics/             # 方案指标、选优、日志（与算法解耦）
│   ├── engine/
│   │   ├── optimizers.py        # PlateOptimizer、StockOptimizer
│   │   ├── packers.py           # 库存填缝：MaxRects BAF、Guillotine BSSF+LLAS
│   │   ├── complementary_pairs.py
│   │   ├── row_layout.py        # 行式互补排布
│   │   ├── rectpack_trials.py   # 单张板 rectpack 多序试探
│   │   ├── ortools_packing.py   # OR-Tools：面积分板、单张 2D CP-SAT
│   │   ├── ortools_plate_engines.py
│   │   ├── cutting_algorithms/  # rectpack / OR-Tools 算法注册与配置解析
│   │   └── pipeline/            # 输入归一、顺序装箱、refine、追踪日志
│   ├── services/
│   │   └── cutting_service.py   # optimize_cutting、multistart、单算法入口
│   ├── tests/                   # pytest（API、行式互补、OR-Tools）
│   └── requirements.txt
└── README.md
```

说明：业务优化入口为 **`services/cutting_service.py`** 与 **`api.py`**（仓库内无独立 `main.py` 优化模块）。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js、React、TypeScript、Tailwind CSS、Supabase 客户端 |
| 后端 | FastAPI、Pydantic Settings、slowapi、**rectpack**、可选 **ortools** |
| 数据 | Supabase（认证与项目数据） |

## 快速开始

**环境**：Node.js 18+、Python 3.10+（推荐 3.12）、Git。

### 前端

```bash
cd frontend
npm install
npm run dev
```

开发地址：<http://localhost:3000>

### 后端

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

API 默认：<http://localhost:8000>，文档：<http://localhost:8000/docs>

### 环境变量

**后端** `backend/.env.local`：

```env
HOST=127.0.0.1
PORT=8000
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

**前端** `frontend/.env.local`：

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## 使用流程（产品）

1. 登录后在 **项目列表** 新建或打开项目。
2. 在 **项目详情** 维护大板、订单零件、余料（others）、锯片厚度与是否启用库存优化。
3. 执行 **切板**：前端调用 `POST /optimize`，将返回的 `cutting_plans` 写回项目。
4. 在 **排版页** 查看 SVG 切割布局与利用率。

## 切割算法与架构要点

- **主装箱（大板上的订单件）**：以 `rectpack` 多种算法类为主（如 Guillotine、MaxRects、Skyline）；算法 ID 在 `engine/cutting_algorithms/packing_registry.py` 注册，可通过环境变量 **`CUTTING_ALGORITHMS_ENABLED`** 控制 `auto` 模式下参与比较的子集。
- **可选 OR-Tools**：`ORToolsAssignMaxRects`（面积分板 + 内层 rectpack）、`ORToolsCP2D`（单张板 2D CP-SAT，大件数回退 rectpack）。相关参数见 `config.py`（`ORTOOLS_*`）。
- **互补对与行式排布**：`complementary_pairs` + `row_layout`，由 `PlateOptimizer` 在启用行式互补时选用。
- **库存填缝**：`StockOptimizer` + `packers.py`；策略 ID 由 **`STOCK_ALGORITHMS_ENABLED`** 限制时可解析为 `stock_registry`。
- **指标与选优**：`core/metrics/`（利用率、板数、未完成订单等），`auto` 模式对多算法结果做字典序比较选优。
- **后处理 · refine**：低利用率板可触发多轮 **refine**（`engine/pipeline/refine.py`），仅当整体指标严格变好时采纳。
- **后处理 · 布局整合（Layout Consolidation）**：全局择优后、返回结果前，`engine/pipeline/consolidate.py` 将「独板版型」（只出现一次的排版组合）的订单件汇总重排，尝试减少版型种类数（例：`5×A + 1×B + 1×C → 5×A + 2×B`）。只有版型数严格减少（且板数不增）或板数严格减少时才采纳，最多 3 轮。
- **后处理 · 切割简化（Cut Simplifier）**：整合之后、`engine/pipeline/cut_simplifier.py` 按版型将订单件重排为行式齐头（Guillotine 行），使订单件产生的内部切割线数（唯一内水平线 + 唯一内垂线）严格减少时才采纳；再 `finalize` 补余板。
- **流水线**：`engine/pipeline/` 负责输入归一、顺序逐板装箱、refine、布局整合、切割简化、可选 **`CUTTING_TRACE_LOG_STAGES` / `CUTTING_DEBUG_DUMP_DIR`** 调试输出。

## 测试结构

- **正式测试集**：`backend/tests/` 文件夹下包含了核心功能的自动化测试（单元和集成测试）。运行方式：`cd backend && pytest tests/ -v`
- **临时验证脚本**：项目根目录曾有一些临时排查、校验的测试和分析脚本（如 `test_expert.py`、`test_api.py` 等）。为保持代码库整洁，它们已统一迁移至 `scripts/test_scripts/` 目录下。

## API 说明

### `POST /optimize`（同步，当前前端使用）

请求体（节选，完整模型见 `api.py` 中 `CuttingRequest`）：

```json
{
  "uid": "string",
  "project_id": "string",
  "plates": [{ "length": 2440, "width": 1220, "quantity": 1 }],
  "orders": [{ "id": "o1", "length": 500, "width": 300, "quantity": 2 }],
  "others": [],
  "optimization": true,
  "saw_blade": 4,
  "multistart_runs": 1,
  "multistart_seed": null
}
```

- **`multistart_runs`**：多起点次数（上限由 `MULTISTART_MAX_RUNS` 约束），`1` 即单次。
- 计算在进程池中执行，避免阻塞事件循环；响应仍为完整 **`CuttingResponse`**（`code`、`cutting_plans`、`total_utilization` 等）。

成功示例：

```json
{
  "code": 0,
  "message": "Success",
  "cutting_plans": [],
  "total_utilization": 0.85,
  "pieces_placed": 10,
  "plates_used": 2
}
```

### `POST /optimize/async` + `GET /optimize/jobs/{job_id}`（异步）

- 提交后立即返回 **`job_id`**；客户端轮询任务状态，`completed` 时 **`result`** 为与同步接口相同的 `CuttingResponse` 结构。
- 校验失败可能返回 **HTTP 400**（与同步路径「200 + 非零 code」不同）。
- 任务数量有上限（`OPTIMIZE_JOB_MAX_STORED`），过期任务查询会 404。

前端若迁移异步模式，需在 `api.ts` 增加路径并实现轮询；保持 `POST /optimize` 则无需改动。

## 后端配置摘要（`config.py` / 环境变量）

| 类别 | 变量示例 | 说明 |
|------|-----------|------|
| 服务 | `HOST`、`PORT`、`RELOAD`、`LOG_LEVEL` | 开发与日志 |
| 限流 | `LIMIT_RATE`、`LIMIT_CONCURRENCY` | API 与并发优化槽位 |
| 多起点 | `MULTISTART_MAX_RUNS` | 请求体 `multistart_runs` 上限 |
| 异步任务 | `OPTIMIZE_JOB_MAX_STORED` | 内存中保留的任务条数 |
| 算法开关 | `CUTTING_ALGORITHMS_ENABLED`、`STOCK_ALGORITHMS_ENABLED` | 逗号分隔 ID |
| OR-Tools | `ORTOOLS_*` | 时间限制、件数上限、内层 rectpack ID |
| 调试 | `CUTTING_TRACE_LOG_STAGES`、`CUTTING_DEBUG_DUMP_DIR` | 阶段日志与 JSON dump |

## 测试

```bash
cd backend
PYTHONPATH=. pytest
PYTHONPATH=. pytest tests/test_api.py -q
```

## 数据与约束

- 板材、订单：**长、宽、数量为正整数**；锯片厚度 **> 0**（可为小数）；板材与零件尺寸需 **大于锯片厚度**。
- 项目持久化字段与前端类型以 Supabase 表及 `project/[id]/page.tsx` 为准。

## 故障排除

| 现象 | 建议 |
|------|------|
| 前端连不上 API | 检查 `NEXT_PUBLIC_API_URL` 与后端是否监听 8000 |
| 优化超时 | 调大 `TIMEOUT`、或减少 `multistart_runs`；大任务可考虑异步接口 |
| ortools 相关错误 | 确认 `pip install -r requirements.txt` 含 `ortools` |
| 限流 429 | 调整 `LIMIT_RATE` 或请求频率 |

## 贡献与许可

欢迎 Issue / PR。许可证以仓库内 **LICENSE** 为准（若未提供，请联系维护者）。

---

**文档维护**：若目录或接口有变更，请同步更新本 README 与根目录 **`CLAUDE.md`**（面向 AI/开发者的更细命令与约定）。
