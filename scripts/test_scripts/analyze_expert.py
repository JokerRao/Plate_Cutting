"""
The user's expert solution:
- 5 boards: 2x (1014x814) + 4x (806x350) each
- 5 boards: 2x (1014x814) + 3x (1006x350) each  
- 1 board: 5x (1006x350)

Order items (from the screenshot):
1. 1014x814 qty=20
2. 1006x350 qty=20
3. 350x806  qty=20  <-- This is NOT 806x350. It's 350 FIRST, 806 SECOND.

So item 2 is 1006 long x 350 wide.
Item 3 is 350 long x 806 wide.

Are they related? 
Item 2: 1006x350  
Item 3: 350x806

These look very close but are NOT exactly the same:
- Item 2: 1006 x 350
- Item 3: 806 x 350 (rotated is 350x806, which means length=806, width=350)

Wait, the screenshot says:
Row 3: length=350, width=806, qty=20

So Item 3 has length=350, width=806.
If we rotate it: length=806, width=350.

Item 2 has length=1006, width=350.
Rotated Item 3: length=806, width=350.

These differ: 1006 vs 806 DIFFERENT widths along the plate!

The expert's solution mixed these two differently-sized (1006x350 and 350x806=806x350) items.
The board is 2440x1220.

Expert plan:
5 plates: 2x(1014x814) + 4x(806x350)
   -> 2 big pieces + 4 pieces of "350x806 rotated to 806x350"
   
5 plates: 2x(1014x814) + 3x(1006x350)

1 plate: 5x(1006x350)

Let me verify the area math to see if each board is feasible.
"""

bt = 4
BW = 2440 + bt  # 2444
BH = 1220 + bt  # 1224

# Expert board type 1: 2 big + 4 small_806
# Pieces with blade:
big_w, big_h = 1018, 818  # 1014+4, 814+4
small_w, small_h = 810, 354  # 806+4, 350+4

# SAT confirmed layout: 2 stacked columns, each col = big on top, small rotated below
# Column: (max(1018,810)=1018) wide, total height used = 818 + 354 = 1172 <= 1224
# Two columns: 2 * 1018 = 2036 < 2444
# Tail gap: 2444 - 2036 = 408 wide, full height 1224
#   Can we fit small_h (354) x small_w (810) in there? 354 <= 408, 810 <= 1224? YES!
#   Then another small: at x=354, width left=408-354=54. Too small.
# So per board: 2 big + 2 small (cols) + 1 small (tail gap) = 2 big + 3 small

# Expert says 4 small! Let me re-check.
# Actually each column stacks 1 big + 1 small. 
# 2 cols * 1 small = 2 small + 1 tail small = 3 small OF 806x350
# But expert says 4x(806x350). There must be more room.

# Let me check: after 2 columns, remaining strip is 2444-2036=408 wide x 1224 tall.
# Can we fit 2 pieces of small_w x small_h = 354 x 810 in a 408x1224 zone?
# 354 <= 408 (width), 810 <= 1224 (height): YES, one piece fits.
# Stack another: 354+810=1164 <= 1224: YES! Two pieces fit in one column of 354 wide!
# So tail = 2 pieces.
# Total small_806 per board = 2 + 2 = 4. Expert is correct!

print("Board type 1 analysis (2 big + 4x 806x350):")
print(f"  2 columns: {2*1018}mm wide, each col: {max(big_w,small_w)}={max(big_w,small_w)} mm wide")
print(f"  Column height: {big_h} + {small_h} = {big_h+small_h} <= {BH}: {'OK' if big_h+small_h <= BH else 'FAIL'}")
print(f"  Tail gap: {BW - 2*max(big_w,small_w)} mm wide x {BH} mm tall")
tail_w = BW - 2*max(big_w, small_w)
# two small_806 pieces rotated (354 wide, 810 tall) stacked
stacked = 2 * small_h  # 354+354=708? No, stacked vertically: 810+810=1620? Too much.
# Let's place small_806 (810x354) in portrait/landscape modes in the tail:
# Landscape: 810 wide, 354 tall -> 810 > 408, doesn't fit in portrait.
# Portrait: 354 wide, 810 tall -> 354 <= 408, fine. Stack 2 vertically: 2*810=1620>1224.
# Only 1 fits in portrait. That gives only 3 pieces total.

# Perhaps the 4th piece fits in the leftover space above/below the Big pieces?
# Column top: big uses 818 tall. Remainder in that column: 1224-818=406 tall after big (before blade).
# Actually: Big=1018x818, Small=810x354 in same column.
# Column width=1018. Column height=818+354=1172. Remaining height= 1224-1172=52. Too small.
# Adjacent column small side: 1018 columns, but only 810 used for small. Leftover: 1018-810=208 x 354 area.
# 208 is only big enough for... nothing useful.

# Hmm. Let me try a different layout for the tail gap.
# Tail gap = 2444 - 2036 = 408 wide x 1224 tall.
# Small (806x350): rotated is 350x806.
# Place first: 354 wide x 810 tall. Y=0-810.
# Place second: 354 wide x 810 tall. Y=810-1620. No, >1224.
# Only one portrait fits. 3 total.

# Maybe the expert's column width isn't 1018?
# What if column width = max(1018, 810) = 1018, but the big piece is only 1014 wide (1018 with blade).
# So we have 1018+1018=2036, tail=408. One small fits. Total=3. 
# That doesn't match 4.

# Alternative: what if we DON'T use composite stacking but a smarter regular packing?
# Big pieces are "roughly half" the board. 2 big across = 1014*2=2028 + saws = 2036 of 2440.
# Height: 814 of 1220. Remaining strip below (2440 wide, 1220-814=406 tall)
# In that strip, can we fit 806x350? With rotation: 350x806. 350 <= 406, 806 <= 2440. YES!
# How many? floor(2440/806)=3 pieces across, plus one more piece 350x806 (height 806>406, rotated: 806 wide, 350 tall)
# Actually: 2440/810=3.01 => 3 such pieces at 810 wide each, using 2430mm.
# With rotated: 350 tall, 806 wide. 350 <= 406, so 3 fit across, using 3*810=2430 of 2440.
# Wait but 352+4=354 tall. 406-354=52 remaining. Can we fit another row? 354>52, no.
# So 3 small_806 in the bottom strip.
# Total: 2 big + 3 small_806. Still 3, not 4.

# OR: the user means something completely different!
# The user said: "806*350" which might be that this piece is 806 long and 350 wide (same as item 3).
# The user's REAL plan: each of the first 5 plates = 2x (1014x814) + 4x (806x350)

# Let me try: place 2 big horizontally at top, then 4 small in bottom strip.
# Big: 1018 wide, placed at x=0 and x=1018 (with blade). Total: 2036 wide.
# Big height: 818. Bottom strip: 1224-818=406 tall, 2444 wide.
# 806x350 in bottom strip: if height<=406, yes if 354<=406 YES. Width: 810 per piece.
# How many strips of 810 fit in 2444? floor(2444/810) = 3.01 -> 3. Total width = 2430.
# Can we place the 4th piece? Remaining x: 2444-2430=14. Too narrow.

# What about below the big + using remaining right gap from big?
# After 2 big: used 2036 wide. Right gap: 2444-2036=408 wide, full 1224 tall.
# Fit small_806 (810x354) in 408x1224? Only if rotated: 354x810. 354<=408 yes, 810<=1224 yes. ONE piece.
# So from the right gap: 1 piece.
# From bottom strip (2036 wide, 406 tall): floor(2036/810)=2 pieces.
# Total: 2 big + 2 (bottom) + 1 (right) = 2 big + 3 small. Still 3!

# CONCLUSION: The expert's "5 boards with 2+4" may refer to 4 pieces of item 2 (1006x350), 
# NOT item 3 (350x806)! Let me re-read:
# "五张：1014*814*2+806*350*4"  -> 1014x814 x2 pieces, 806x350 x4 pieces
# But item 3 is "length=350, width=806" -> this IS a 350x806 piece.
# When placed: 806 wide, 350 tall (rotated). matches "806x350" description!

# The expert writes "806*350" meaning 806 long, 350 wide.
# The input has item 3 with length=350, width=806 -- these ARE the same pieces (just described differently)

# For 4 pieces of 806x350 (i.e., 4x item3 rotated):
# Piece dimensions (with blade): 810 wide (806+4), 354 tall (350+4)

# Try a grid layout for all 6 pieces (2 big + 4 small):
# Let me search for a valid arrangement.
for big_x2 in [0, 1]:  # try placing 2 big left-to-right
    for small_rows in range(1, 5):
        pass

print(f"\n  Tail gap: {tail_w}mm")
print(f"  One portrait small_806 (354x810): 354 <= {tail_w} = {'YES' if 354<=tail_w else 'NO'}")
print(f"  Two stacked portrait smalls: 2*810={2*810} <= {BH} = {'YES' if 2*810<=BH else 'NO'}")

if __name__ == "__main__":
    pass
