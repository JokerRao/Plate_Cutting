"""Use OR-Tools SAT to ENUMERATE ALL possible ways to pack 2 big + 4 small_806 on a 2440x1220 board.
Then check if 2 big + 4 small_1006 is also achievable."""

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

from ortools.sat.python import cp_model

def try_pack(big_count, small_w_raw, small_h_raw, small_count, label=""):
    BT = 4
    W = 2440 + BT
    H = 1220 + BT
    
    big_w, big_h = 1014 + BT, 814 + BT   # 1018, 818
    sm_w, sm_h = small_w_raw + BT, small_h_raw + BT
    
    model = cp_model.CpModel()
    
    items = []
    for i in range(big_count):
        items.append(("big", big_w, big_h))
    for i in range(small_count):
        items.append(("small", sm_w, sm_h))
    
    xs, ys, rxs, rys, exs, eys = [], [], [], [], [], []
    ix_list, iy_list = [], []
    
    for i, (name, iw, ih) in enumerate(items):
        r = model.new_bool_var(f"r_{i}")
        rx = model.new_int_var(0, max(W, H), f"rw_{i}")
        ry = model.new_int_var(0, max(W, H), f"rh_{i}")
        model.add(rx == iw).only_enforce_if(r.negated())
        model.add(ry == ih).only_enforce_if(r.negated())
        model.add(rx == ih).only_enforce_if(r)
        model.add(ry == iw).only_enforce_if(r)
        
        x = model.new_int_var(0, W, f"x_{i}")
        y = model.new_int_var(0, H, f"y_{i}")
        ex = model.new_int_var(0, W, f"ex_{i}")
        ey = model.new_int_var(0, H, f"ey_{i}")
        model.add(x + rx == ex)
        model.add(y + ry == ey)
        
        ixv = model.new_interval_var(x, rx, ex, f"ix_{i}")
        iyv = model.new_interval_var(y, ry, ey, f"iy_{i}")
        xs.append(x); ys.append(y); rxs.append(rx); rys.append(ry)
        exs.append(ex); eys.append(ey)
        ix_list.append(ixv); iy_list.append(iyv)
    
    model.add_no_overlap_2d(ix_list, iy_list)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  ✓ FEASIBLE: {label}")
        for i, (name, iw, ih) in enumerate(items):
            print(f"    {name}[{i}]: pos=({solver.value(xs[i])},{solver.value(ys[i])}) "
                  f"size=({solver.value(rxs[i])}x{solver.value(rys[i])})")
        return True
    else:
        print(f"  ✗ INFEASIBLE: {label}")
        return False

print("=== Expert plan feasibility check ===")
print()
print("Board type 1: 2x(1014x814) + 4x(806x350)")
try_pack(2, 806, 350, 4, "2 big + 4 small_806")
print()
print("Board type 2: 2x(1014x814) + 3x(1006x350)")
try_pack(2, 1006, 350, 3, "2 big + 3 small_1006")
print()
print("Board type 3: 5x(1006x350)")
try_pack(0, 1006, 350, 5, "5 small_1006")
print()
print("Board type BONUS: 2x(1014x814) + 4x(1006x350)")
try_pack(2, 1006, 350, 4, "2 big + 4 small_1006")
