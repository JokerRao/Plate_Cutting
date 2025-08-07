from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from config import Settings, get_settings
from main import optimize_cutting
import logging
import asyncio

# 错误码定义
ERROR_CODES = {
    0: "Success",
    1001: "No valid cutting plans could be generated",
    1002: "Invalid plate dimensions",
    1003: "Invalid order dimensions",
    1004: "Insufficient plates for orders",
    1005: "All pieces too large for available plates",
    1006: "Invalid quantity specified",
    5000: "Internal server error"
}

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

def setup_logging(settings: Settings):
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
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

    # 配置允许的源
    origins = [
        "https://platecutting.cedrao.com",
        "http://localhost:3000",  # 本地开发
        "http://localhost:5173",  # Vite 默认端口
        # 添加其他需要的域名
    ]
    
    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # 或者使用 ["*"] 允许所有源（不推荐用于生产）
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有请求头
        expose_headers=["*"],
        max_age=3600,
    )
    
    # 其他中间件
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
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
    start_x: int
    start_y: int
    length: int
    width: int
    is_stock: bool
    id: Union[str, int]  # 允许字符串或整数类型的 id

class CuttingPlan(BaseModel):
    rate: float = Field(..., ge=0, le=1, description="Utilization rate")
    plate: List[int] = Field(..., min_length=2, max_length=2, description="Plate dimensions [length, width]")
    cutted: List[CutPiece]

class CuttingRequest(BaseModel):
    plates: List[Plate]
    orders: List[Order]
    others: Optional[List[StockPlate]] = None
    optimization: bool = Field(False, description="Whether to optimize stock plate placement")
    saw_blade: Optional[int] = Field(None, gt=0, description="Saw blade thickness in mm")

class CuttingResponse(BaseModel):
    code: int = Field(..., description="Response code, 0 for success, other values for errors")
    message: str = Field(..., description="Response message, error description when code is not 0")
    cutting_plans: List[Dict[str, Any]]
    total_utilization: float
    pieces_placed: int
    plates_used: int
    unplaced_pieces: Optional[dict] = Field(None, description="Details of pieces that could not be placed")
    warnings: Optional[List[str]] = Field(None, description="Warning messages if any")
    optimization_details: Optional[Dict[str, Any]] = Field(None, description="Additional optimization details")

# Semaphore for limiting concurrent optimizations
optimization_semaphore = asyncio.Semaphore(settings.LIMIT_CONCURRENCY)

def validate_dimensions(plates: List[dict], orders: List[dict]) -> tuple[bool, Optional[int], Optional[str]]:
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
            return False, 1002, f"Invalid plate dimensions: {plate.get('length')}x{plate.get('width')}"

    # 检查订单尺寸
    for order in orders:
        if order.get('length', 0) <= 0 or order.get('width', 0) <= 0:
            return False, 1003, f"Invalid order dimensions: {order.get('length')}x{order.get('width')}"

    # 检查是否所有订单都大于板材
    min_plate_area = min((p['length'] * p['width'] for p in plates), default=0)
    all_orders_too_large = all(
        o['length'] * o['width'] > min_plate_area
        for o in orders
    )
    if all_orders_too_large:
        return False, 1005, "All order pieces are too large for available plates"

    return True, None, None

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
        logger.info("Received cutting optimization request")
        
        # Convert request models to dictionaries
        plates_dict = [plate.model_dump() for plate in cutting_request.plates]
        orders_dict = [order.model_dump() for order in cutting_request.orders]
        others_dict = [stock.model_dump() for stock in cutting_request.others] if cutting_request.others else []

        # 验证输入数据
        is_valid, error_code, error_message = validate_dimensions(plates_dict, orders_dict)
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

        # 收集优化细节
        optimization_details = {
            "saw_blade_width": saw_blade,
            "optimization_enabled": cutting_request.optimization,
            "total_plates_available": sum(p['quantity'] for p in plates_dict),
            "total_pieces_requested": sum(o['quantity'] for o in orders_dict),
            "stock_pieces_available": len(others_dict) if others_dict else 0
        }

        # Call optimization function
        cutting_plans = optimize_cutting(
            plates=plates_dict,
            orders=orders_dict,
            others=others_dict,
            optim=int(cutting_request.optimization),
            saw_blade=saw_blade
        )

        if not cutting_plans:
            return CuttingResponse(
                code=1001,
                message=ERROR_CODES[1001],
                cutting_plans=[],
                total_utilization=0,
                pieces_placed=0,
                plates_used=0,
                optimization_details=optimization_details
            )

        # Convert cutting plans to response format
        formatted_plans = []
        total_pieces = 0
        total_utilization = 0
        placed_pieces = {}  # Track placed pieces by ID
        unplaced_pieces = {}  # Track unplaced pieces
        warnings = []

        # Initialize unplaced_pieces with all ordered pieces
        for order in orders_dict:
            unplaced_pieces[order['id']] = order['quantity']

        for plan in cutting_plans:
            pieces = []
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
                # Update placed pieces count
                piece_id = piece[5]
                placed_pieces[piece_id] = placed_pieces.get(piece_id, 0) + 1
                if piece_id in unplaced_pieces:
                    unplaced_pieces[piece_id] = max(0, unplaced_pieces[piece_id] - 1)

            # 将 CuttingPlan 对象转换为字典
            formatted_plan = {
                "rate": plan['rate'],
                "plate": plan['plate'],
                "cutted": [piece.model_dump() for piece in pieces]  # 将 CutPiece 对象转换为字典
            }
            formatted_plans.append(formatted_plan)
            total_utilization += plan['rate']

        avg_utilization = total_utilization / len(cutting_plans) if cutting_plans else 0

        # Remove pieces that were fully placed
        unplaced_pieces = {k: v for k, v in unplaced_pieces.items() if v > 0}

        # 添加警告信息
        if unplaced_pieces:
            warnings.append(f"Could not place all pieces: {unplaced_pieces}")
        if avg_utilization < 0.5:
            warnings.append(f"Low utilization rate: {avg_utilization:.2%}")

        # 更新优化细节
        optimization_details.update({
            "average_utilization": avg_utilization,
            "total_pieces_placed": total_pieces,
            "unplaced_pieces_count": sum(unplaced_pieces.values()) if unplaced_pieces else 0
        })

        # 将 orders 和 others 信息添加到 cutting_plans 中
        formatted_plans.append({
            "rate": 1.0,
            "plate": [0, 0],  # 虚拟板材尺寸
            "cutted": [],  # 空列表，因为这是元数据
            "metadata": {
                "orders": orders_dict,
                "others": others_dict
            }
        })

        # 创建响应对象
        response = CuttingResponse(
            code=0,
            message="Success" if not unplaced_pieces else "Partial success - some pieces could not be placed",
            cutting_plans=formatted_plans,  # 使用已经转换为字典的格式
            total_utilization=avg_utilization,
            pieces_placed=total_pieces,
            plates_used=len(cutting_plans),
            unplaced_pieces=unplaced_pieces if unplaced_pieces else None,
            warnings=warnings if warnings else None,
            optimization_details=optimization_details
        )

        logger.info(f"Successfully generated {len(cutting_plans)} cutting plans")
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

@app.get("/")
async def root():
    """重定向到 API 文档"""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "version": settings.API_VERSION} 