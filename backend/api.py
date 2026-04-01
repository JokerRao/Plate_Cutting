import asyncio
import functools
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Union

from config import Settings, get_settings
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from optimization_jobs import OptimizationJobRegistry
from services.cutting_service import optimize_cutting_multistart
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# 错误码定义
ERROR_CODES = {
    0: "Success",
    1001: "No valid cutting plans could be generated",
    1002: "Invalid plate dimensions",
    1003: "Invalid order dimensions",
    1004: "Insufficient plates for orders",
    1005: "All pieces too large for available plates",
    1006: "Invalid quantity specified",
    5000: "Internal server error",
    1008: "Optimization job not found",
}

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def setup_logging(settings: Settings):
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=settings.LOG_FORMAT
    )
    return logging.getLogger('plate_cutting_api')


def create_app(settings: Settings):
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        debug=settings.DEBUG,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 确保这些 URL 完全匹配（包括协议和端口）
    origins = [
        "https://platecutting.cedrao.com",  # 确保没有尾部斜杠
        "http://platecutting.cedrao.com",   # 如果可能使用 HTTP
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://192.168.6.95:3000",
        "http://192.168.1.10:3000",
    ]

    # CORS 中间件必须在其他中间件之前添加
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex="^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+)(:[0-9]+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],  # 使用通配符允许所有头
        expose_headers=["*"],
        max_age=3600,
    )

    # 其他中间件
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.state.optimization_jobs = OptimizationJobRegistry(
        max_jobs=settings.OPTIMIZE_JOB_MAX_STORED)

    return app


settings = get_settings()
logger = setup_logging(settings)
app = create_app(settings)


class PlateBase(BaseModel):
    id: Union[str, int]  # 允许字符串或整数类型的 id
    length: int = Field(..., gt=0, description="Length of the plate in mm")
    width: int = Field(..., gt=0, description="Width of the plate in mm")
    description: Optional[str] = None


class Plate(PlateBase):
    quantity: int = Field(..., gt=0, description="Quantity of plates")


class Order(PlateBase):
    quantity: int = Field(..., gt=0, description="Quantity of pieces needed")


class StockPlate(PlateBase):
    client: Optional[str] = None


class CutPiece(BaseModel):
    start_x: float
    start_y: float
    length: float
    width: float
    is_stock: bool
    id: Union[str, int]  # 允许字符串或整数类型的 id


class CuttingPlan(BaseModel):
    rate: float = Field(..., ge=0, le=1, description="Utilization rate")
    plate: List[float] = Field(...,
                               min_length=2,
                               max_length=2,
                               description="Plate dimensions [length, width]")
    cutted: List[CutPiece]


class CuttingRequest(BaseModel):
    plates: List[Plate]
    orders: List[Order]
    others: Optional[List[StockPlate]] = None
    optimization: bool = Field(
        False, description="Whether to optimize stock plate placement")
    saw_blade: Optional[float] = Field(
        None, gt=0, description="Saw blade thickness in mm (supports decimals)")
    multistart_runs: int = Field(
        1,
        ge=1,
        description="多起点次数：打乱订单行顺序重复优化，取指标最优；1 表示单次",
    )
    multistart_seed: Optional[int] = Field(
        None,
        description="多起点随机种子，缺省固定基准以便可复现",
    )


class CuttingResponse(BaseModel):
    code: int = Field(...,
                      description="Response code, 0 for success, other values for errors")
    message: str = Field(...,
                         description="Response message, error description when code is not 0")
    cutting_plans: List[Dict[str, Any]]
    total_utilization: float
    pieces_placed: int
    plates_used: int
    unplaced_pieces: Optional[dict] = Field(
        None, description="Details of pieces that could not be placed")
    warnings: Optional[List[str]] = Field(
        None, description="Warning messages if any")
    optimization_details: Optional[Dict[str, Any]] = Field(
        None, description="Additional optimization details")


class OptimizeAsyncAccepted(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Use GET /optimize/jobs/{job_id} to poll for results"


class OptimizeJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[CuttingResponse] = None
    error: Optional[str] = None


# Semaphore for limiting concurrent optimizations
optimization_semaphore = asyncio.Semaphore(settings.LIMIT_CONCURRENCY)

# Process pool for CPU-bound cutting calculations (avoids blocking the event loop)
_process_pool = ProcessPoolExecutor(max_workers=4)


def validate_dimensions(plates: List[dict],
                        orders: List[dict]) -> tuple[bool,
                                                     Optional[int],
                                                     Optional[str]]:
    """验证板材和订单尺寸的合法性"""
    # 检查板材数量
    total_plates = sum(p.get('quantity', 0) for p in plates)
    if total_plates == 0:
        return False, 1004, "No valid plates specified"

    # 检查订单数量
    total_orders = sum(o.get('quantity', 0) for o in orders)
    if total_orders == 0:
        return False, 1006, "No valid orders specified"

    # 检查板材尺寸
    for plate in plates:
        if plate.get('length', 0) <= 0 or plate.get('width', 0) <= 0:
            return False, 1002, f"Invalid plate dimensions: {
                plate.get('length')}x{
                plate.get('width')}"

    # 检查订单尺寸
    for order in orders:
        if order.get('length', 0) <= 0 or order.get('width', 0) <= 0:
            return False, 1003, f"Invalid order dimensions: {
                order.get('length')}x{
                order.get('width')}"

    # 检查是否所有订单都大于板材
    min_plate_area = min((p['length'] * p['width'] for p in plates), default=0)
    all_orders_too_large = all(
        o['length'] * o['width'] > min_plate_area
        for o in orders
    )
    if all_orders_too_large:
        return False, 1005, "All order pieces are too large for available plates"

    return True, None, None


def build_cutting_response_from_plans(
    cutting_plans: List[Dict[str, Any]],
    orders_dict: List[dict],
    others_dict: List[dict],
    optimization_details: Dict[str, Any],
) -> CuttingResponse:
    """将 optimize_cutting_multistart 返回的原始方案格式化为 CuttingResponse。"""
    if not cutting_plans:
        return CuttingResponse(
            code=1001,
            message=ERROR_CODES[1001],
            cutting_plans=[],
            total_utilization=0,
            pieces_placed=0,
            plates_used=0,
            optimization_details=optimization_details,
        )

    formatted_plans: List[Dict[str, Any]] = []
    total_pieces = 0
    total_utilization = 0.0
    unplaced_pieces: Dict[Any, int] = {}
    warnings: List[str] = []

    for order in orders_dict:
        unplaced_pieces[order['id']] = order['quantity']

    for plan in cutting_plans:
        pieces: List[CutPiece] = []
        for piece in plan['cutted']:
            pieces.append(CutPiece(
                start_x=piece[0],
                start_y=piece[1],
                length=piece[2],
                width=piece[3],
                is_stock=bool(piece[4]),
                id=piece[5]
            ))
            total_pieces += 1
            piece_id = piece[5]
            if piece_id in unplaced_pieces:
                unplaced_pieces[piece_id] = max(
                    0, unplaced_pieces[piece_id] - 1)

        formatted_plans.append({
            "rate": plan['rate'],
            "plate": plan['plate'],
            "cutted": [p.model_dump() for p in pieces]
        })
        total_utilization += plan['rate']

    avg_utilization = total_utilization / \
        len(cutting_plans) if cutting_plans else 0.0

    unplaced_pieces = {k: v for k, v in unplaced_pieces.items() if v > 0}

    if unplaced_pieces:
        warnings.append(f"Could not place all pieces: {unplaced_pieces}")
    if avg_utilization < 0.5:
        warnings.append(f"Low utilization rate: {avg_utilization:.2%}")

    optimization_details = {
        **optimization_details,
        "average_utilization": avg_utilization,
        "total_pieces_placed": total_pieces,
        "unplaced_pieces_count": sum(
            unplaced_pieces.values()) if unplaced_pieces else 0,
    }

    formatted_plans.append({
        "rate": 1.0,
        "plate": [0, 0],
        "cutted": [],
        "metadata": {
            "orders": orders_dict,
            "others": others_dict
        }
    })

    return CuttingResponse(
        code=0,
        message="Success" if not unplaced_pieces else (
            "Partial success - some pieces could not be placed"),
        cutting_plans=formatted_plans,
        total_utilization=avg_utilization,
        pieces_placed=total_pieces,
        plates_used=len(cutting_plans),
        unplaced_pieces=unplaced_pieces if unplaced_pieces else None,
        warnings=warnings if warnings else None,
        optimization_details=optimization_details,
    )


async def run_optimization_job_task(
    job_id: str,
    registry: OptimizationJobRegistry,
    plates_dict: List[dict],
    orders_dict: List[dict],
    others_dict: List[dict],
    optim: int,
    saw_blade: float,
    n_starts: int,
    multistart_seed: Optional[int],
) -> None:
    if not await registry.mark_running(job_id):
        return
    optimization_details: Dict[str, Any] = {
        "saw_blade_width": saw_blade,
        "optimization_enabled": optim,
        "total_plates_available": sum(p['quantity'] for p in plates_dict),
        "total_pieces_requested": sum(o['quantity'] for o in orders_dict),
        "stock_pieces_available": len(others_dict) if others_dict else 0,
        "multistart_runs": n_starts,
        "multistart_seed": multistart_seed,
        "async_job": True,
    }
    try:
        loop = asyncio.get_event_loop()
        worker = functools.partial(
            optimize_cutting_multistart,
            plates_dict,
            orders_dict,
            others_dict,
            optim,
            saw_blade,
            n_starts=n_starts,
            multistart_seed=multistart_seed,
        )
        async with optimization_semaphore:
            cutting_plans = await loop.run_in_executor(
                _process_pool, worker)
        response = build_cutting_response_from_plans(
            cutting_plans,
            orders_dict,
            others_dict,
            optimization_details,
        )
        await registry.mark_completed(job_id, response.model_dump())
    except Exception as e:
        logger.exception("Optimization job %s failed", job_id)
        await registry.mark_failed(job_id, str(e))


@app.post("/optimize", response_model=CuttingResponse)
@limiter.limit(settings.LIMIT_RATE)
async def optimize_plates(
    request: Request,
    cutting_request: CuttingRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Optimize cutting patterns for given plates and orders

    Args:
        request: The HTTP request object
        cutting_request: CuttingRequest object containing plates, orders, and optimization parameters
        settings: Application settings

    Returns:
        CuttingResponse object with optimized cutting plans and statistics
    """
    try:
        print("FRONTEND PAYLOAD:", cutting_request.model_dump(), flush=True)
        import sys
        sys.stdout.flush()
        logger.info("Received cutting optimization request")
        logger.info(f"Frontend payload: {cutting_request.model_dump()}")

        # Convert request models to dictionaries
        plates_dict = [plate.model_dump() for plate in cutting_request.plates]
        orders_dict = [order.model_dump() for order in cutting_request.orders]
        others_dict = [stock.model_dump(
        ) for stock in cutting_request.others] if cutting_request.others else []

        # 验证输入数据
        is_valid, error_code, error_message = validate_dimensions(
            plates_dict, orders_dict)
        if not is_valid:
            return CuttingResponse(
                code=error_code,
                message=error_message,
                cutting_plans=[],
                total_utilization=0,
                pieces_placed=0,
                plates_used=0
            )

        # Use default saw_blade from settings if not provided
        saw_blade = cutting_request.saw_blade or settings.DEFAULT_SAW_BLADE

        n_starts = min(
            cutting_request.multistart_runs,
            settings.MULTISTART_MAX_RUNS,
        )
        optimization_details: Dict[str, Any] = {
            "saw_blade_width": saw_blade,
            "optimization_enabled": cutting_request.optimization,
            "total_plates_available": sum(p['quantity'] for p in plates_dict),
            "total_pieces_requested": sum(o['quantity'] for o in orders_dict),
            "stock_pieces_available": len(others_dict) if others_dict else 0,
            "multistart_runs": n_starts,
            "multistart_seed": cutting_request.multistart_seed,
            "sync_endpoint": True,
        }

        loop = asyncio.get_event_loop()
        worker = functools.partial(
            optimize_cutting_multistart,
            plates_dict,
            orders_dict,
            others_dict,
            int(cutting_request.optimization),
            saw_blade,
            n_starts=n_starts,
            multistart_seed=cutting_request.multistart_seed,
        )
        async with optimization_semaphore:
            cutting_plans = await loop.run_in_executor(
                _process_pool, worker)

        response = build_cutting_response_from_plans(
            cutting_plans,
            orders_dict,
            others_dict,
            optimization_details,
        )

        logger.info(
            "Successfully generated %s cutting plans (multistart=%s)",
            len(cutting_plans) if cutting_plans else 0,
            n_starts,
        )
        return response

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return CuttingResponse(
            code=5000,
            message=f"{ERROR_CODES[5000]}: {str(e)}",
            cutting_plans=[],
            total_utilization=0,
            pieces_placed=0,
            plates_used=0,
            warnings=["An internal error occurred during optimization"]
        )


@app.post("/optimize/async", response_model=OptimizeAsyncAccepted)
@limiter.limit(settings.LIMIT_RATE)
async def optimize_async_submit(
    request: Request,
    cutting_request: CuttingRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    提交异步优化任务，立即返回 job_id；通过 GET /optimize/jobs/{job_id} 轮询结果。
    """
    print("FRONTEND ASYNC PAYLOAD:", cutting_request.model_dump(), flush=True)
    import sys; sys.stdout.flush()
    plates_dict = [plate.model_dump() for plate in cutting_request.plates]
    orders_dict = [order.model_dump() for order in cutting_request.orders]
    others_dict = [
        stock.model_dump() for stock in cutting_request.others
    ] if cutting_request.others else []

    is_valid, error_code, error_message = validate_dimensions(
        plates_dict, orders_dict)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={"code": error_code, "message": error_message},
        )

    saw_blade = cutting_request.saw_blade or settings.DEFAULT_SAW_BLADE
    n_starts = min(
        cutting_request.multistart_runs,
        settings.MULTISTART_MAX_RUNS,
    )

    registry: OptimizationJobRegistry = request.app.state.optimization_jobs
    job_id = await registry.create_pending()
    background_tasks.add_task(
        run_optimization_job_task,
        job_id,
        registry,
        plates_dict,
        orders_dict,
        others_dict,
        int(cutting_request.optimization),
        saw_blade,
        n_starts,
        cutting_request.multistart_seed,
    )
    return OptimizeAsyncAccepted(job_id=job_id)


@app.get("/optimize/jobs/{job_id}", response_model=OptimizeJobStatusResponse)
@limiter.limit("30/second")
async def optimize_job_status(
    request: Request,
    job_id: str,
):
    registry: OptimizationJobRegistry = request.app.state.optimization_jobs
    job = await registry.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"code": 1008, "message": ERROR_CODES[1008]},
        )
    result_model: Optional[CuttingResponse] = None
    if job.result is not None:
        result_model = CuttingResponse.model_validate(job.result)
    return OptimizeJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=result_model,
        error=job.error,
    )


@app.get("/")
async def root():
    """重定向到 API 文档"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "version": settings.API_VERSION}
