import pytest
from core.models import CuttingConfig
from engine.optimizers import StockOptimizer
from core.models import Cut, SmallPlate, Rectangle

def test_stock_optimizer():
    config = CuttingConfig(blade_thickness=4.0)
    optimizer = StockOptimizer(config, "guillotine_bssf_llas")
    
    # Simulate a plate of 2440x1220 with some existing cuts
    existing_cuts = [
        Cut(plate=SmallPlate(1000, 600), x1=0, y1=0, x2=1000, y2=600),
        Cut(plate=SmallPlate(1000, 600), x1=1004, y1=0, x2=2004, y2=600),
    ]
    
    # We have stock plates to fit
    stock_plates = [
        SmallPlate(400, 300, "S1", 1),
        SmallPlate(500, 400, "S2", 1),
    ]
    
    # fill with stock
    cuts = optimizer.fill_with_stock(2440, 1220, existing_cuts, stock_plates, optimize=True)
    
    assert len(cuts) > 0
    print(f"\nPlaced {len(cuts)} stock plates")
