import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from core.models import CuttingConfig, SmallPlate
from services.cutting_service import run_single_algorithm
from engine.cutting_algorithms.packing_registry import resolve_packing_class

plates = [{"length": 2440, "width": 1220, "quantity": 1000}]
orders = [
    {"id": "A", "length": 400, "width": 200, "quantity": 527},
    {"id": "B", "length": 400, "width": 200, "quantity": 415}
]

from config import get_settings
from engine.cutting_algorithms.packing_registry import normalize_enabled_packing_ids, iter_enabled_packing_algorithms
enabled = normalize_enabled_packing_ids(get_settings().CUTTING_ALGORITHMS_ENABLED)

for algo_name, algo_class in iter_enabled_packing_algorithms(enabled):
    print(f"\n--- Testing {algo_name} ---")
    results, metrics = run_single_algorithm(
        plates, orders, [], 1, 5.0, algo_class, "maxrects_baf", True
    )
    print(f"Total plates used: {len(results)}")
    
    # Count pieces per plate
    for i, r in enumerate(results):
        n_pieces = len([c for c in r.get("cuts", []) if not c.get("is_stock")])
        if "cuts" not in r and "orders" in r:
            n_pieces = len(r["orders"])
        if n_pieces < 36:
            print(f"  Plate {i+1}: utilization = {r.get('rate', 0)*100:.2f}%, pieces = {n_pieces}")
