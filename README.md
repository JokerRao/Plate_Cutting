# Plate Cutting Optimization System

一个用于优化板材切割的 Web 应用系统，包含前端界面和后端优化算法。

## 项目结构

```
Plate_Cutting/
├── frontend/          # 前端项目目录
│   ├── src/          # 源代码
│   │   ├── app/      # Next.js 应用目录
│   │   ├── components/# React 组件
│   │   └── utils/    # 工具函数
│   └── public/       # 静态资源
└── backend/          # 后端项目目录
    ├── api.py        # API 接口定义
    ├── db.py         # 数据库操作
    ├── main.py       # 核心优化算法
    └── run.py        # 服务启动脚本
```

## 前端功能

### 项目页面 (`/project`)
- 项目列表展示
- 新建项目
- 项目详情查看

### 项目详情页 (`/project/[id]`)
- 项目基本信息编辑
- 板材信息管理
- 零件信息管理
- 常用尺寸管理
- 切板优化设置
- 数据导入导出

### 排版页面 (`/layout/[id]`)
- 切割方案展示
- 板材利用率统计
- 零件使用情况统计
- 分页查看切割方案

## 后端功能

### API 接口 (`api.py`)
- 项目管理接口
- 切割优化接口
- 数据同步接口

### 数据库操作 (`db.py`)
- Supabase 数据库连接
- 项目数据 CRUD
- 切割方案存储

### 核心算法 (`main.py`)
- 板材切割优化算法
- 间隙利用优化
- 切割方案生成

## 技术栈

### 前端
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- Supabase Client

### 后端
- FastAPI
- Python 3.8+
- Supabase
- NumPy

## 开发环境设置

### 前端设置
```bash
cd frontend
npm install
npm run dev
```

### 后端设置
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## 数据模型

### 项目 (Projects)
- id: 项目ID
- name: 项目名称
- details: 项目详情
- description: 项目描述
- saw_blade: 锯片宽度
- plates: 板材信息
- orders: 零件信息
- others: 常用尺寸
- cutted: 切割方案
- created_at: 创建时间
- updated_at: 更新时间

### 切割方案 (CuttingPlan)
- rate: 利用率
- plate: 板材信息
- cutted: 切割记录

## API 接口

### 优化切割
- 端点: `/api/optimize`
- 方法: POST
- 请求体:
  ```json
  {
    "uid": "用户ID",
    "project_id": "项目ID",
    "plates": [],
    "orders": [],
    "others": [],
    "optimization": 1,
    "saw_blade": 0
  }
  ```
- 响应:
  ```json
  {
    "code": 0,
    "message": "success",
    "cutting_plans": [],
    "total_utilization": 0,
    "pieces_placed": 0,
    "plates_used": 0
  }
  ```

## 使用说明

1. 创建新项目
2. 添加板材信息
3. 添加零件信息
4. 设置优化参数
5. 执行切割优化
6. 查看切割方案
7. 导出或保存结果

## 注意事项

- 确保输入的数据为正整数
- 板材和零件的尺寸必须大于锯片宽度
- 优化模式会影响切割方案的生成
- 建议定期保存项目数据
