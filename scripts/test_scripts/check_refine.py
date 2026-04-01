import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
from ortools.sat.python import cp_model

def try_pack_items(items, label):
    BT = 4
    W = 2440 + BT
    H = 1220 + BT
    
    model = cp_model.CpModel()
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

BT = 4
big = ("big", 1014+BT, 814+BT)
s1006 = ("s1006", 1006+BT, 350+BT)
s806 = ("s806", 806+BT, 350+BT)

print("Refine candidate: plate_7 (2 big + 2 s1006) + plate_12 (2 s806) merged:")
try_pack_items([big, big, s1006, s1006, s806, s806], "2 big + 2 s1006 + 2 s806")

print()
print("Can the refine fit everything onto 1 board? (6 pieces)")
area = 2*1014*814 + 2*1006*350 + 2*806*350
print(f"Total piece area: {area}, Board area: {2440*1220}, ratio: {area/(2440*1220):.2%}")
