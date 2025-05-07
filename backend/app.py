from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from config import Settings, get_settings
from db import sync_cutting_results
from main import optimize_cutting

# Configure logging
logger = logging.getLogger("plate_cutting_api")

def setup_logging(settings: Settings):
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT
    )
    return logger

def create_app(settings: Settings = Depends(get_settings)):
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        debug=settings.DEBUG
    )
    return app

# Initialize settings and app
settings = get_settings()
setup_logging(settings)
app = create_app(settings)

# Models
class PlateBase(BaseModel):
    id: str
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
    id: str

class CuttingPlan(BaseModel):
    rate: float = Field(..., ge=0, le=1, description="Utilization rate")
    plate: List[int] = Field(..., min_items=2, max_items=2, description="Plate dimensions [length, width]")
    cutted: List[CutPiece]

class CuttingRequest(BaseModel):
    uid: str = Field(..., description="User ID")
    project_id: str = Field(..., description="Project ID")
    plates: List[Plate]
    orders: List[Order]
    others: Optional[List[StockPlate]] = None
    optimization: bool = Field(False, description="Whether to optimize stock plate placement")
    plate_count: Optional[int] = None
    saw_blade: int = Field(None, gt=0, description="Saw blade thickness in mm")

class CuttingResponse(BaseModel):
    cutting_plans: List[CuttingPlan]
    total_utilization: float
    pieces_placed: int
    plates_used: int

@app.post("/optimize", response_model=CuttingResponse)
async def optimize_plates(
    request: CuttingRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Optimize cutting patterns for given plates and orders and sync results to Supabase
    """
    try:
        logger.info(f"Received cutting optimization request for project {request.project_id}")
        
        # Convert request models to dictionaries
        plates_dict = [plate.dict() for plate in request.plates]
        orders_dict = [order.dict() for order in request.orders]
        others_dict = [stock.dict() for stock in request.others] if request.others else []

        # Use default saw_blade from settings if not provided
        saw_blade = request.saw_blade or settings.DEFAULT_SAW_BLADE

        # Call optimization function
        cutting_plans = optimize_cutting(
            plates=plates_dict,
            orders=orders_dict,
            others=others_dict,
            optim=int(request.optimization),
            n_plate=request.plate_count,
            saw_blade=saw_blade
        )

        if not cutting_plans:
            raise HTTPException(status_code=400, detail="No valid cutting plans could be generated")

        # Convert cutting plans to response format
        formatted_plans = []
        total_pieces = 0
        total_utilization = 0

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

            formatted_plans.append(CuttingPlan(
                rate=plan['rate'],
                plate=plan['plate'],
                cutted=pieces
            ))
            total_utilization += plan['rate']

        avg_utilization = total_utilization / len(cutting_plans) if cutting_plans else 0

        response = CuttingResponse(
            cutting_plans=formatted_plans,
            total_utilization=avg_utilization,
            pieces_placed=total_pieces,
            plates_used=len(cutting_plans)
        )

        # Sync results to Supabase
        sync_success = await sync_cutting_results(
            uid=request.uid,
            project_id=request.project_id,
            cutting_plans=formatted_plans,
            plates=plates_dict,
            orders=orders_dict,
            others=others_dict
        )

        if not sync_success:
            logger.warning(f"Failed to sync cutting results to Supabase for project {request.project_id}")

        logger.info(f"Successfully generated {len(cutting_plans)} cutting plans for project {request.project_id}")
        return response

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 