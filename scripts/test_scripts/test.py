import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from core.models import CuttingConfig, SmallPlate
from services.cutting_service import optimize_cutting

plates = [{"length": 2440, "width": 1220, "quantity": 1000}]
orders = [
    {"id": "A", "length": 400, "width": 200, "quantity": 527},
    {"id": "B", "length": 400, "width": 200, "quantity": 415}
]

results = optimize_cutting(
    plates=plates,
    orders=orders,
    others=[],
    optim=1,
    saw_blade=4.0,
    algorithm="auto",
    stock_algorithm="maxrects_baf"
)

print(f"Total plates used: {len(results)}")
for i, r in enumerate(results):
    n_pieces = len([c for c in r.get("cuts", []) if not c.get("is_stock")])
    if "cuts" not in r and "orders" in r:
        n_pieces = len(r["orders"])
        
    print(f"Plate {i+1}: utilization = {r.get('rate', 0)*100:.2f}%, pieces = {n_pieces}")
