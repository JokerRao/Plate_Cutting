import logging
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.metrics import calculate_cutting_metrics
from core.models import Cut, CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import PlateOptimizer, StockOptimizer
from engine.pipeline.output import finalize_plate_output
from engine.pipeline.sequential import finalize_metrics_and_refine
from engine.pipeline.templates import clone_plate_template
from engine.pipeline.trace_context import CuttingTraceContext

from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

logger = logging.getLogger("plate_cutting")


def run_common_dim_strip_then_rectpack(
    big_plates: List[SmallPlate],
    plate_templates: List[SmallPlate],
    small_plates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    inner_algo_class: Any,
    stock_algorithm: str,
    optim: int,
    fallback_rectpack: Callable[[], Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    trace: Optional[CuttingTraceContext] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generic 1D strip packer for cases where many pieces share a dimension."""
    
    bt = int(round(config.blade_thickness))
    board_w = int(round(big_plates[0].length))
    board_h = int(round(big_plates[0].width))
    
    # 1. Identify common dimension (pick the most frequent eligible dimension)
    counts: Dict[int, int] = defaultdict(int)
    for sp in small_plates:
        counts[int(round(sp.length)) + bt] += 1
        counts[int(round(sp.width)) + bt] += 1

    total_qty = len(small_plates)
    cand_dim = None
    best_count = 0
    for d, c in counts.items():
        if d <= max(board_h, board_w) + bt and c >= total_qty * 0.35 and c > best_count:
            best_count = c
            cand_dim = d
            
    if cand_dim is None:
        return fallback_rectpack()
        
    logger.info("CommonDimStrip: Found common dimension %d", cand_dim)
    
    # Determine orientation
    # We want cand_dim to be along the Y-axis (height) so strips run horizontally
    if cand_dim <= board_h:
        strip_height = cand_dim
        strip_max_width = board_w
        rotated = False
    else:
        strip_height = cand_dim
        strip_max_width = board_h
        # swap board logic if cand_dim only fits in length
        board_w, board_h = board_h, board_w
        rotated = True
        
    # 2. Separate conforming and non-conforming
    conforming = []
    non_conforming = []
    
    # small_plates are already flattened (one SmallPlate per physical piece) by convert_orders
    pool = list(small_plates)
            
    for p in pool:
        w = int(round(p.length))
        h = int(round(p.width))
        if h + bt == cand_dim:
            conforming.append({"p": p, "w1d": w + bt, "w_raw": w, "h_raw": h})
        elif w + bt == cand_dim:
            conforming.append({"p": p, "w1d": h + bt, "w_raw": h, "h_raw": w})
        else:
            non_conforming.append(p)
            
    # 3. Build macros from non-conforming
    macros = []
    nc_types = list(set((int(round(p.length)), int(round(p.width))) for p in non_conforming))
    
    # Also try single-type self-stacking (e.g. 4×300 stacks to ~1200)
    for w1, h1 in nc_types:
        for nx in range(1, 4):
            for ny in range(2, 6):
                for ow, oh in [(w1, h1), (h1, w1)]:
                    bw = nx * ow + (nx - 1) * bt
                    bh = ny * oh + (ny - 1) * bt
                    if bh <= cand_dim - bt and bw <= strip_max_width - bt:
                        area = nx * ny * w1 * h1
                        util = area / (bw * bh)
                        if util >= 0.85 and bh >= (cand_dim - bt) * 0.95:
                            macros.append({
                                'w1d': bw + bt,
                                'stack_h': bh,
                                'util': util,
                                'req1': ((w1, h1), nx * ny),
                                'req2': None,
                                'layout1': (nx, ny, ow, oh),
                                'layout2': None,
                            })

    for w1, h1 in nc_types:
        for w2, h2 in nc_types:
            if (w1, h1) == (w2, h2):
                continue
            for nx1 in range(1, 4):
                bw1 = nx1 * w1 + (nx1 - 1) * bt
                if bw1 > strip_max_width - bt:
                    break  # wider columns won't fit either
                for ny1 in range(1, 4):
                    bh1 = ny1 * h1 + (ny1 - 1) * bt
                    if bh1 >= cand_dim - bt:
                        break  # no room left for the second block + bt separator
                    for ow2, oh2 in [(w2, h2), (h2, w2)]:
                        for nx2 in range(1, 4):
                            bw2 = nx2 * ow2 + (nx2 - 1) * bt
                            if max(bw1, bw2) > strip_max_width - bt:
                                break  # wider columns won't help
                            for ny2 in range(1, 4):
                                bh2 = ny2 * oh2 + (ny2 - 1) * bt
                                stack_h = bh1 + bh2 + bt
                                if stack_h > cand_dim - bt:
                                    break  # taller stacks won't fit either
                                stack_w = max(bw1, bw2)
                                area = nx1 * ny1 * w1 * h1 + nx2 * ny2 * w2 * h2
                                util = area / (stack_w * stack_h)
                                if util >= 0.85 and stack_h >= (cand_dim - bt) * 0.95:
                                    macros.append({
                                        'w1d': stack_w + bt,
                                        'stack_h': stack_h,
                                        'util': util,
                                        'req1': ((w1, h1), nx1 * ny1),
                                        'req2': ((w2, h2), nx2 * ny2),
                                        'layout1': (nx1, ny1, w1, h1),
                                        'layout2': (nx2, ny2, ow2, oh2),
                                    })
                                        
    # Greedily instantiate macros to maximize non-conforming pieces consumed
    instantiated_macros = []
    if macros and non_conforming:
        model = cp_model.CpModel()
        macro_vars = [model.NewIntVar(0, 200, f'm_{i}') for i in range(len(macros))]
        
        avail = defaultdict(int)
        for p in non_conforming:
            avail[(int(round(p.length)), int(round(p.width)))] += 1
            
        def _macro_demand(m: Dict, key: Tuple[int, int]) -> int:
            q = m['req1'][1] if m['req1'][0] == key else 0
            if m['req2'] is not None and m['req2'][0] == key:
                q += m['req2'][1]
            return q

        def _macro_total(m: Dict) -> int:
            t = m['req1'][1]
            if m['req2'] is not None:
                t += m['req2'][1]
            return t

        # Capacity constraints
        for (w, h), count in avail.items():
            used = []
            for i, m in enumerate(macros):
                q = _macro_demand(m, (w, h))
                if q > 0:
                    used.append(macro_vars[i] * q)
            model.Add(sum(used) <= count)

        # Maximize pieces consumed
        consumed = []
        for i, m in enumerate(macros):
            consumed.append(macro_vars[i] * _macro_total(m))
        model.Maximize(sum(consumed))
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2.0
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i, m in enumerate(macros):
                qty = solver.Value(macro_vars[i])
                for _ in range(qty):
                    pieces1 = []
                    k1, q1 = m['req1']
                    for _ in range(q1):
                        for j, p in enumerate(non_conforming):
                            if p and (int(round(p.length)), int(round(p.width))) == k1:
                                pieces1.append(p)
                                non_conforming[j] = None
                                break
                    pieces2 = []
                    if m['req2'] is not None:
                        k2, q2 = m['req2']
                        for _ in range(q2):
                            for j, p in enumerate(non_conforming):
                                if p and (int(round(p.length)), int(round(p.width))) == k2:
                                    pieces2.append(p)
                                    non_conforming[j] = None
                                    break
                    instantiated_macros.append({
                        'is_macro': True,
                        'w1d': m['w1d'],
                        'layout1': m['layout1'],
                        'layout2': m['layout2'],
                        'p1': pieces1,
                        'p2': pieces2,
                    })
                    
    non_conforming = [p for p in non_conforming if p is not None]
    
    # If we have too many leftover non-conforming pieces, fallback
    if non_conforming:
        logger.info("CommonDimStrip: %d leftover non-conforming pieces, falling back", len(non_conforming))
        return fallback_rectpack()
        
    strips = []
    for c in conforming:
        strips.append({'is_macro': False, 'w1d': c['w1d'], 'w_raw': c['w_raw'], 'h_raw': c['h_raw'], 'p': c['p']})
    strips.extend(instantiated_macros)
    
    # 4. 1D Column Generation + ILP (Price-and-Branch)
    
    # 4a. Group identical items to remove symmetry
    width_to_strips = defaultdict(list)
    for s in strips:
        width_to_strips[s['w1d']].append(s)

    unique_widths = list(width_to_strips.keys())
    demands = [len(width_to_strips[w]) for w in unique_widths]
    num_items = len(unique_widths)

    # 4b. Setup Restricted Master Problem (LP relaxation using GLOP)
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        logger.error("CommonDimStrip: GLOP solver unavailable")
        return fallback_rectpack()
        
    constraints = [solver.Constraint(demands[i], solver.infinity()) for i in range(num_items)]
    objective = solver.Objective()
    objective.SetMinimization()

    # Try to generate all maximal columns if the problem is small
    max_pool_size = 2000
    all_maximal = []
    
    def dfs_maximal(current_pattern, current_width, item_idx):
        if len(all_maximal) >= max_pool_size:
            return
        if item_idx == num_items:
            if current_width <= strip_max_width:
                is_max = True
                for i, w in enumerate(unique_widths):
                    if current_width + w <= strip_max_width:
                        is_max = False
                        break
                if is_max:
                    all_maximal.append(list(current_pattern))
            return
        w = unique_widths[item_idx]
        max_qty = (strip_max_width - current_width) // w
        for q in range(max_qty, -1, -1):
            current_pattern.append(q)
            dfs_maximal(current_pattern, current_width + q * w, item_idx + 1)
            current_pattern.pop()
            
    dfs_maximal([], 0, 0)
    
    if len(all_maximal) < max_pool_size:
        # Fully enumerated, no need for CG
        columns = all_maximal
        MAX_CG_ITER = 0
    else:
        # Too many columns, seed with homogeneous patterns and use CG
        columns = []
        for i in range(num_items):
            pattern = [0] * num_items
            pattern[i] = int(strip_max_width // unique_widths[i])
            if pattern[i] == 0:  # Piece larger than bin width
                logger.error("CommonDimStrip: Piece width %d exceeds bin width %d", unique_widths[i], strip_max_width)
                return fallback_rectpack()
            columns.append(pattern)
        MAX_CG_ITER = 1000

    variables = []
    for i, pattern in enumerate(columns):
        var = solver.NumVar(0, solver.infinity(), f'lambda_init_{i}')
        variables.append(var)
        objective.SetCoefficient(var, 1)
        for j in range(num_items):
            if pattern[j] > 0:
                constraints[j].SetCoefficient(var, pattern[j])

    # 4c. Column Generation Loop
    cg_iter = 0
    while cg_iter < MAX_CG_ITER:
        cg_iter += 1
        solver.Solve()
        duals = [c.DualValue() for c in constraints]
        
        # Solve Pricing Problem (1D Bounded Knapsack) using CP-SAT
        kp_model = cp_model.CpModel()
        kp_vars = [kp_model.NewIntVar(0, demands[i], f'x_{i}') for i in range(num_items)]
        kp_model.Add(sum(kp_vars[i] * unique_widths[i] for i in range(num_items)) <= strip_max_width)
        
        # CP-SAT requires integers; multiply duals by scaling factor
        FACTOR = 1000000 
        kp_model.Maximize(sum(kp_vars[i] * int(round(duals[i] * FACTOR)) for i in range(num_items)))
        
        kp_solver = cp_model.CpSolver()
        status = kp_solver.Solve(kp_model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            val = kp_solver.ObjectiveValue() / FACTOR
            if val > 1.00001:  # Reduced cost is negative (1 - dual_sum < 0)
                new_pattern = [kp_solver.Value(kp_vars[i]) for i in range(num_items)]
                
                # Avoid duplicate columns
                if new_pattern not in columns:
                    columns.append(new_pattern)
                    var = solver.NumVar(0, solver.infinity(), f'lambda_{len(columns)-1}')
                    variables.append(var)
                    objective.SetCoefficient(var, 1)
                    for i in range(num_items):
                        if new_pattern[i] > 0:
                            constraints[i].SetCoefficient(var, new_pattern[i])
                    continue
        break  # Converged: No new favorable patterns found

    # 4d. Column Pool Enrichment (Randomized Knapsack)
    if MAX_CG_ITER > 0:
        enrichment_iters = 100 if num_items <= 15 else 50
        import random
        for _ in range(enrichment_iters):
            kp_model = cp_model.CpModel()
            kp_vars = [kp_model.NewIntVar(0, demands[i], f'x_{i}') for i in range(num_items)]
            kp_model.Add(sum(kp_vars[i] * unique_widths[i] for i in range(num_items)) <= strip_max_width)
            
            obj_expr = []
            for i in range(num_items):
                weight = int(round(unique_widths[i] * random.uniform(0.5, 1.5)))
                obj_expr.append(kp_vars[i] * weight)
            kp_model.Maximize(sum(obj_expr))
            
            kp_solver = cp_model.CpSolver()
            kp_solver.parameters.max_time_in_seconds = 0.1
            status = kp_solver.Solve(kp_model)
            
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                new_pattern = [kp_solver.Value(kp_vars[i]) for i in range(num_items)]
                if new_pattern not in columns:
                    columns.append(new_pattern)
                    var = solver.NumVar(0, solver.infinity(), f'lambda_{len(columns)-1}')
                    variables.append(var)
                    objective.SetCoefficient(var, 1)
                    for i in range(num_items):
                        if new_pattern[i] > 0:
                            constraints[i].SetCoefficient(var, new_pattern[i])

    logger.info("Unique widths: %s, Max Width: %s", unique_widths, strip_max_width)
    logger.info("Demands: %s", demands)
    logger.info("Generated columns: %s", columns)

    # 4d. Integer Finalization (ILP using SCIP with Pattern Minimization)
    ilp_solver = pywraplp.Solver.CreateSolver('SCIP')
    if not ilp_solver:
        logger.error("CommonDimStrip: SCIP solver unavailable")
        return fallback_rectpack()
        
    ilp_solver.SetSolverSpecificParametersAsString('limits/gap = 0.0')
        
    ilp_constraints = [ilp_solver.Constraint(demands[i], ilp_solver.infinity()) for i in range(num_items)]
    ilp_objective = ilp_solver.Objective()
    ilp_objective.SetMinimization()

    ilp_vars = []
    # y_vars track if a pattern is used to slightly penalize distinct pattern count
    ilp_y_vars = [] 
    
    for j, pattern in enumerate(columns):
        var = ilp_solver.IntVar(0, ilp_solver.infinity(), f'int_lambda_{j}')
        y_var = ilp_solver.BoolVar(f'y_{j}')
        
        ilp_vars.append(var)
        ilp_y_vars.append(y_var)
        
        # Objective: 1.0 per board + 0.001 per distinct pattern
        ilp_objective.SetCoefficient(var, 1.0)
        ilp_objective.SetCoefficient(y_var, 0.001) 
        
        # Link y_var to var: var <= M * y_var
        ilp_solver.Add(var <= 1000 * y_var)

        for i in range(num_items):
            if pattern[i] > 0:
                ilp_constraints[i].SetCoefficient(var, pattern[i])

    ilp_solver.set_time_limit(5000)  # 5 seconds
    status_ilp = ilp_solver.Solve()

    if status_ilp not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        logger.info("CommonDimStrip: ILP solver failed, falling back")
        return fallback_rectpack()

    # 4e. Reconstruct layout bins from ILP solution
    bins = []
    for j, var in enumerate(ilp_vars):
        count = int(round(var.solution_value()))
        for _ in range(count):
            current_bin = []
            for i in range(num_items):
                qty = columns[j][i]
                # Pop actual strip objects from our grouped list
                for _ in range(qty):
                    if width_to_strips[unique_widths[i]]:
                        current_bin.append(width_to_strips[unique_widths[i]].pop(0))
            if current_bin:
                bins.append(current_bin)
                
    logger.info("CommonDimStrip: CG+ILP packed into %d bins (CG iterations: %d)", len(bins), cg_iter)

    results = []
    row_template_idx = []
    stock_optimizer = StockOptimizer(config, stock_algorithm)
    plate_engine = PlateOptimizer(config, inner_algo_class)

    strips_per_plate = int(board_h // strip_height)
    if strips_per_plate < 1:
        strips_per_plate = 1
        
    plate_idx = 0
    for i in range(0, len(bins), strips_per_plate):
        if plate_idx >= len(plate_templates):
            logger.warning("CommonDimStrip: Out of plate templates")
            break

        chunk = bins[i:i+strips_per_plate]
        bp = clone_plate_template(plate_templates[plate_idx])
        cuts = []

        for k, b_strips in enumerate(chunk):
            y_offset = k * strip_height
            current_x = 0.0

            for s in b_strips:
                if not s['is_macro']:
                    p = s['p']
                    cx = current_x
                    cy = y_offset + 0.0
                    cw = s['w_raw']
                    ch = float(s['h_raw'])
                    
                    if rotated:
                        cx, cy, cw, ch = cy, cx, ch, cw
                    
                    cuts.append(Cut(
                        plate=p,
                        x1=cx,
                        y1=cy,
                        x2=cx + cw,
                        y2=cy + ch,
                        is_stock=False
                    ))
                    current_x += s['w1d']
                else:
                    nx1, ny1, w1, h1 = s['layout1']
                    bx = current_x
                    by = 0.0
                    p_idx = 0
                    for _y in range(ny1):
                        for _x in range(nx1):
                            p = s['p1'][p_idx]
                            p_idx += 1
                            cx = bx + _x * (w1 + bt)
                            cy = y_offset + by + _y * (h1 + bt)
                            cw, ch = w1, h1
                            if rotated:
                                cx, cy, cw, ch = cy, cx, ch, cw
                            cuts.append(Cut(plate=p, x1=cx, y1=cy, x2=cx+cw, y2=cy+ch, is_stock=False))

                    # Block 2 (top) — only present for dual-type macros
                    if s['layout2'] is not None:
                        nx2, ny2, w2, h2 = s['layout2']
                        bx = current_x
                        by = ny1 * h1 + ny1 * bt
                        p_idx = 0
                        for _y in range(ny2):
                            for _x in range(nx2):
                                p = s['p2'][p_idx]
                                p_idx += 1
                                cx = bx + _x * (w2 + bt)
                                cy = y_offset + by + _y * (h2 + bt)
                                cw, ch = w2, h2
                                if rotated:
                                    cx, cy, cw, ch = cy, cx, ch, cw
                                cuts.append(Cut(plate=p, x1=cx, y1=cy, x2=cx+cw, y2=cy+ch, is_stock=False))

                    current_x += s['w1d']

        row = finalize_plate_output(bp, cuts, stock_plates, stock_optimizer, optim, config, converter)
        results.append(row)
        row_template_idx.append(plate_idx)
        plate_idx += 1

    # Apply the same low-utilization repack pass used by other algorithms
    return finalize_metrics_and_refine(
        results,
        row_template_idx,
        non_conforming,
        plate_templates,
        plate_engine,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
        trace,
    )
