"""
Expert plan:
- 5 boards: 2x big + 4x small_806   -> uses 10 big (of 20), 20 small_806 (of 20)
- 5 boards: 2x big + 3x small_1006  -> uses 10 big (of 20), 15 small_1006 (of 20)
- 1 board:  5x small_1006           -> uses 5 small_1006 (of 20)
Total: 20 big + 20 small_806 + 20 small_1006 = 60 pieces in 11 boards. PERFECT!

Area verification:
Board = 2440x1220 = 2,976,800 mm²
Big = 1014x814 = 825,396 mm²
Small_806 = 806x350 = 282,100 mm²
Small_1006 = 1006x350 = 352,100 mm²

Board 1 (5 of these): 2*825396 + 4*282100 = 1,650,792 + 1,128,400 = 2,779,192
Utilization = 2,779,192 / 2,976,800 = 93.37%

Board 2 (5 of these): 2*825396 + 3*352100 = 1,650,792 + 1,056,300 = 2,707,092
Utilization = 2,707,092 / 2,976,800 = 90.94%

Board 3 (1 of these): 5*352100 = 1,760,500
Utilization = 1,760,500 / 2,976,800 = 59.14%

This matches our current output for plates 1-6 (90.94%) but the expert uses DIFFERENT grouping:
- Expert groups all 20 small_806 on the first 5 boards.
- Our engine interleaves small_1006 with big on all boards, then has leftovers.

The key optimization: treat small_806 as the "companion" for big pieces (5 of per board thanks
to the special geometry), and small_1006 for the remaining boards.

Since the engine selects ONE pattern per call to `sort_orders`, and the pattern only considers
one PAIR (big, small_1006 composite or big, small_806 composite), it cannot select the
globally optimal assignment of piece types to plate templates.

CONCLUSION: we need a GLOBAL plate-type planner that first decides how many of each
plate template to use, then fills them in order. This is fundamentally a bin-packing
with template selection problem.
"""

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

BW, BH = 2440, 1220
BIG = 1014 * 814
S806 = 806 * 350
S1006 = 1006 * 350
BOARD = BW * BH

print("=== Expert Plan Utilization ===")
print(f"Board 1 (2 big + 4 small_806): {(2*BIG + 4*S806)/BOARD:.2%}")
print(f"Board 2 (2 big + 3 small_1006): {(2*BIG + 3*S1006)/BOARD:.2%}")
print(f"Board 3 (5x small_1006): {(5*S1006)/BOARD:.2%}")
print()

# What does our engine output?
# 6x (2 big + 2 small_1006 + 1 small_806 rotated) = 90.94%?
# Let's calculate:
check = 2*BIG + 2*S1006 + S806
print(f"Our engine: 2 big + 2 small_1006 + 1 small_806: {check/BOARD:.2%}")

# Current output is 5x (2 big + 2 small_1006 + 1 small_806) at 90.94%?
# Wait, the output showed:  
# "Plate 1: 90.94%, pieces=5, 2x(1014x814), 2x(1006x350), 1x(350x1006)"
# So it's 2 big + 2 S1006 + 1 S1006-rotated = 2 big + 3 S1006
# 1006x350 and 350x1006 are the SAME piece (just rotated)!
check2 = 2*BIG + 3*S1006 
print(f"Actually: 2 big + 3 small_1006: {check2/BOARD:.2%}")
