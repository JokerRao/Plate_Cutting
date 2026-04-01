from ortools.sat.python import cp_model

def solve():
    model = cp_model.CpModel()
    
    W = 2444
    H = 1224
    
    items = [
        {"name": "Big1", "w": 1018, "h": 818},
        {"name": "Big2", "w": 1018, "h": 818},
        {"name": "Small1", "w": 810, "h": 354},
        {"name": "Small2", "w": 810, "h": 354},
        {"name": "Small3", "w": 810, "h": 354},
        {"name": "Small4", "w": 810, "h": 354},
    ]
    
    x_vars, y_vars, rx_vars, ry_vars, r_vars = [], [], [], [], []
    intervals_x, intervals_y = [], []
    
    for i, item in enumerate(items):
        r = model.NewBoolVar(f"r_{i}")
        w, h = item["w"], item["h"]
        
        rx = model.NewIntVar(0, max(W, H), f"rx_{i}")
        ry = model.NewIntVar(0, max(W, H), f"ry_{i}")
        
        model.Add(rx == w).OnlyEnforceIf(r.Not())
        model.Add(ry == h).OnlyEnforceIf(r.Not())
        
        model.Add(rx == h).OnlyEnforceIf(r)
        model.Add(ry == w).OnlyEnforceIf(r)
        
        x = model.NewIntVar(0, W, f"x_{i}")
        y = model.NewIntVar(0, H, f"y_{i}")
        ex = model.NewIntVar(0, W, f"ex_{i}")
        ey = model.NewIntVar(0, H, f"ey_{i}")
        
        model.Add(x + rx == ex)
        model.Add(y + ry == ey)
        
        ix = model.NewIntervalVar(x, rx, ex, f"ix_{i}")
        iy = model.NewIntervalVar(y, ry, ey, f"iy_{i}")
        
        intervals_x.append(ix)
        intervals_y.append(iy)
        
        x_vars.append(x)
        y_vars.append(y)
        rx_vars.append(rx)
        ry_vars.append(ry)
        r_vars.append(r)
        
    model.AddNoOverlap2D(intervals_x, intervals_y)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("SOLUTION FOUND!")
        for i, item in enumerate(items):
            print(f"{item['name']}: pos=({solver.Value(x_vars[i])}, {solver.Value(y_vars[i])}) "
                  f"size=({solver.Value(rx_vars[i])} x {solver.Value(ry_vars[i])})")
    else:
        print("NO SOLUTION FOUND!")

if __name__ == "__main__":
    solve()
