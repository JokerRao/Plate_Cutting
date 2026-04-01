import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

from services.cutting_service import optimize_cutting

def test_no_complementary():
    """Test without complementary pairs to see if pure rectpack does better."""
    plates = [{"id": 1, "length": 2440, "width": 1220, "quantity": 100}]
    orders = [
        {"id": 1, "length": 1014, "width": 814, "quantity": 20},
        {"id": 2, "length": 1006, "width": 350, "quantity": 20},
        {"id": 3, "length": 350,  "width": 806, "quantity": 20}
    ]
    
    # Test each algorithm independently
    for algo in ["MaxRectsBaf", "GuillotineBssfLlas", "SkylineMwfWm"]:
        results = optimize_cutting(
            plates=plates, orders=orders, others=[],
            optim=0, saw_blade=4.0, algorithm=algo,
            enable_row_complementary=False  # disable our complementary logic
        )
        
        n = len(results)
        rates = [r.get("rate", 0)*100 for r in results]
        avg = sum(rates)/n if n else 0
        print(f"{algo}: {n} plates, avg={avg:.2f}%, rates={[f'{r:.1f}' for r in rates[:5]]}...")

if __name__ == "__main__":
    test_no_complementary()
