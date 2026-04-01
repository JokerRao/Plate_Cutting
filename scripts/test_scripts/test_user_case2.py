import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from services.cutting_service import optimize_cutting

def test_user_case():
    plates = [{"id": 1, "length": 2440, "width": 1220, "quantity": 100}]
    orders = [
        {"id": 1, "length": 1014, "width": 814, "quantity": 20},
        {"id": 2, "length": 1006, "width": 350, "quantity": 20},
        {"id": 3, "length": 350,  "width": 806, "quantity": 20}
    ]
    
    results = optimize_cutting(
        plates=plates,
        orders=orders,
        others=[],
        optim=0,
        saw_blade=4.0, 
        algorithm="auto",
        stock_algorithm="maxrects_baf"
    )
    
    print(f"\nTotal plates used: {len(results)}")
    
    total_pieces = 0
    for i, r in enumerate(results):
        cuts = r.get("cutted", [])
        real_cuts = [c for c in cuts if c[4] == 0]
        n_pieces = len(real_cuts)
        total_pieces += n_pieces
        rate = r.get("rate", 0) * 100
        sizes_str = ""
        if n_pieces > 0:
            sizes = [f"{c[2]:.0f}x{c[3]:.0f}" for c in real_cuts]
            from collections import Counter
            counts = Counter(sizes)
            sizes_str = ", ".join(f"{count} x ({sz})" for sz, count in counts.items())
        print(f"Plate {i+1}: utilization = {rate:.2f}%, pieces = {n_pieces} ({sizes_str})")
        
    print(f"\nTotal pieces packed: {total_pieces}")

if __name__ == "__main__":
    test_user_case()
