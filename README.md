# 板材切割优化系统

一个专业的板材切割优化Web应用系统，提供智能的切割方案生成和可视化展示。

## 🚀 项目特色

- **智能优化算法**：基于先进算法的板材切割优化
- **可视化展示**：直观的切割方案图形化展示
- **项目管理**：完整的项目生命周期管理
- **实时统计**：详细的利用率和使用情况统计
- **响应式设计**：支持多设备访问的现代化界面

## 📁 项目结构

```
Plate_Cutting/
├── frontend/                 # 前端项目
│   ├── src/
│   │   ├── app/             # Next.js 应用页面
│   │   ├── components/      # React 组件
│   │   ├── config/          # 配置文件
│   │   └── utils/           # 工具函数
│   ├── public/              # 静态资源
│   └── package.json         # 前端依赖
├── backend/                  # 后端项目
│   ├── api.py               # FastAPI 路由与中间件入口
│   ├── config.py            # 配置管理 (Pydantic Settings)
│   ├── run.py               # Uvicorn 服务启动
│   ├── core/                # 基础模块层
│   │   ├── models.py        # 数据类：CuttingConfig, SmallPlate, Cut, Rectangle
│   │   └── utils.py         # 工具函数：DataConverter, 指标计算, 算法比较
│   ├── engine/              # 算法引擎层
│   │   ├── packers.py       # 装箱算法：MaxRects BAF, Guillotine BSSF+LLAS
│   │   └── optimizers.py    # 优化调度：PlateOptimizer, StockOptimizer
│   ├── services/            # 业务服务层
│   │   └── cutting_service.py  # 核心入口：optimize_cutting, run_single_algorithm
│   ├── tests/               # API 测试 (pytest)
│   └── requirements.txt     # Python 依赖
└── README.md               # 项目说明
```

## 🛠️ 技术栈

### 前端
- **Next.js 15** - React 全栈框架
- **TypeScript** - 类型安全的 JavaScript
- **Tailwind CSS** - 实用优先的 CSS 框架
- **Supabase** - 实时数据库和认证服务

### 后端
- **FastAPI** - 现代、快速的 Python Web 框架
- **Python 3.8+** - 编程语言
- **Supabase** - 数据库服务
- **Pydantic** - 数据验证

## 🚀 快速开始

### 环境要求
- Node.js 18+
- Python 3.8+
- Git

### 1. 克隆项目
```bash
git clone <repository-url>
cd Plate_Cutting
```

### 2. 前端设置
```bash
cd frontend
npm install
npm run dev
```
前端服务将在 http://localhost:3000 启动

### 3. 后端设置
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python run.py
```
后端服务将在 http://localhost:8000 启动

### 4. 环境配置

前后端各自使用独立的 `.env.local`（路径不同，变量名也不同）。

**后端**（`backend/.env.local`）：

```env
HOST=127.0.0.1
PORT=8000
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

**前端**（`frontend/.env.local`）：

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## 📖 使用指南

### 项目管理
1. **创建项目**：在项目列表页面点击"新建项目"
2. **编辑信息**：设置项目名称、详情和描述
3. **配置参数**：设置锯片宽度和优化模式

### 数据管理
1. **板材信息**：添加大板材的尺寸和数量
2. **零件信息**：添加需要切割的零件尺寸和数量
3. **常用尺寸**：管理常用的零件尺寸信息

### 切割优化
1. **参数设置**：选择优化模式（正常/优化）
2. **执行优化**：点击"切板"按钮开始优化
3. **查看结果**：在排版页面查看切割方案
4. **统计分析**：查看利用率和使用情况统计

## 🔧 核心功能

### 前端功能
- **项目列表** (`/project`)：项目管理和列表展示
- **项目详情** (`/project/[id]`)：项目信息编辑和数据管理
- **排版展示** (`/layout/[id]`)：切割方案可视化展示

### 后端功能
- **优化算法**：智能的板材切割优化算法
- **API 接口**：完整的 RESTful API 服务
- **数据管理**：项目数据的增删改查
- **实时同步**：与 Supabase 数据库的实时同步

## 🧮 排版切割算法与流程

后端的排版切割引擎是本系统的核心，基于 Python 和 `rectpack` 库进行深度定制与增强，主要包含以下优化流程与算法原理：

### 1. 数据预处理与参数计算
- **考虑锯片物理损耗**：解析前端传入的板材、零件（订单）及余料库存数据。根据用户设定的**锯片厚度 (Saw Blade)**，在测算阶段为每块待排放的零件自动加上锯片缝隙余量，以符合真实的物理切割占位尺寸。
- **智能寻优排序**：系统通过评估不同规格零件的“组合潜力”与“适配难度”，自动测算**互补尺寸**组合（如发现哪些不同尺寸的零件可结合完美打满行与列）。随后通过轮询 (Round Robin)、贪心 (Greedy) 和平衡 (Balanced) 等策略验证评出最佳的放置序列队列。

### 2. 核心排版引擎 (PlateOptimizer)
- **多向评估与自适应摆放**：不仅依托了基础布局算法体系，更增加自动感知是否应**旋转 (Rotation)** 的功能。对零件落脚点周围空间作双向适应性探测，取能获取较大空间留存的方向。
- **行式定制布局法**：对于前置过滤出的互补组合直接激活特定的“行向式排布 (Row-Based)”，强制这些板件对齐拼列，大幅提升原板整体利用率并使最终走刀线更为规整简洁。

### 3. 边角余料极小化填充 (StockOptimizer)
针对在完成常规订单零件排版后剩余的大量缝隙与边角区域，算法进一步执行库存填缝逻辑优化，引入两种定制的填空引擎：
- 基于 **MaxRects BAF (Best Area Fit)** 策略：维护并动态裁切大板上的矩形空隙最大边界，落刀选择带来最小总废除空间（最严丝合缝）的落脚点。
- 基于 **Guillotine BSSF + LLAS** 策略：即一刀切特制版策略。系统采纳最优短边适应法，结合长轴剩余极长分裂法则（Long Leftover Axis Split）——确保排完以后剩下的是长长的一根整条而非几个方块碎片，既便于后期余料建库利用也方便锯片一刀通切。并提供长宽周长面积的交叉排列顺序检测来获取极致填充率。

### 4. 数据后处理与指标统计
- **尺寸回缩归真**：方案落实后，所有的排版坐标将剥除最初加入的“锯片厚度缓冲值”，提取为纯木板长宽 `(width, height)` 及纯净标定起点 `(x1, y1)` 还原发回前端交互渲染。
- **精算指标下发**：汇总利用总耗原大板片数、超高精度利用率 (Utilization rate%) 以及最终未落地失败（如果板不够放）的订单名单，全景交付出高质量下料单。

## 📊 数据模型

### 项目 (Projects)
```typescript
interface Project {
  id: number;
  name: string;
  details: string;
  description: string;
  saw_blade: number;
  plates: Plate[];
  orders: Order[];
  others: Other[];
  cutted: CuttingPlan[];
  created_at: string;
  updated_at: string;
}
```

### 切割方案 (CuttingPlan)
```typescript
interface CuttingPlan {
  rate: number;           // 利用率
  plate: [number, number]; // 板材尺寸
  cutted: CuttedItem[];   // 切割记录
}
```

## 🔌 API 接口

### 优化切割
```http
POST /optimize
Content-Type: application/json

{
  "plates": [...],
  "orders": [...],
  "others": [...],
  "optimization": true,
  "saw_blade": 4
}
```

### 响应格式
```json
{
  "code": 0,
  "message": "success",
  "cutting_plans": [...],
  "total_utilization": 85.5,
  "pieces_placed": 120,
  "plates_used": 5
}
```

## ⚙️ 配置说明

### 后端配置 (`backend/config.py`)
- **服务器配置**：HOST, PORT, DEBUG 等
- **CORS 配置**：跨域请求设置
- **数据库配置**：Supabase 连接信息
- **优化参数**：默认的切割参数

### 前端配置 (`frontend/src/config/`)
- **API 配置**：后端服务地址
- **Supabase 配置**：数据库连接信息

## 🐛 故障排除

### 常见问题
1. **前端无法启动**：检查 Node.js 版本和依赖安装
2. **后端连接失败**：检查 Python 环境和依赖
3. **数据库连接错误**：验证 Supabase 配置信息
4. **优化算法异常**：检查输入数据的有效性

### 调试与日志

```bash
# 前端开发（热更新）
cd frontend && npm run dev

# 后端开发：`run.py` 使用 `config.py` 中的设置；热重载由环境变量 `RELOAD` 控制（默认 true）
cd backend && python run.py
```

如需更详细的 uvicorn 日志，可在 `backend/.env.local` 中设置 `LOG_LEVEL=debug`（参见 `backend/config.py`）。

## 📝 开发说明

### 代码规范
- 使用 TypeScript 进行类型检查
- 遵循 ESLint 代码规范
- 使用 Prettier 进行代码格式化

### 测试
```bash
cd backend
pytest
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

以仓库内声明为准（若未包含 LICENSE 文件，请联系维护者）。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至：[your-email@example.com]

---

**注意**：板材与零件的长度、宽度、数量需为正整数，锯片宽度需为大于零的数值（支持小数），且板材及零件尺寸必须大于锯片宽度。
