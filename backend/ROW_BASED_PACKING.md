# Row-Based Mixed Packing Implementation

## Overview
Implemented custom row-based packing to solve cases where pieces with the same height can be optimally arranged in mixed rows, achieving near-perfect utilization.

## Problem Solved
**Test Case**: Big plate 2440×1220mm with:
- Type a: 527×400mm (200 pieces)
- Type b: 415×400mm (200 pieces)
- Expected optimal: 9a+6b per plate (97.2% utilization)

**Previous Result**: Only 12a pieces placed (84.98% utilization)
**New Result**: 9a+6b pieces placed (97.19% utilization) ✓

## Implementation Details

### 1. Enhanced Complementary Detection
Modified `find_complementary_pairs()` to:
- Return both utilization gains AND pattern details
- Detect row-based patterns when heights match
- Store pattern type ('row' or 'column') and piece counts

```python
def find_complementary_pairs(self, size_groups, L, W) -> Tuple[Dict, Dict]:
    # Returns: (complementary_dict, pattern_details_dict)
    # Pattern details include: type, count1, count2, rows
```

### 2. Custom Row-Based Packing
Added `pack_orders_row_based()` method:
- Manually places pieces in rows according to detected pattern
- Groups orders by size
- Places pieces row by row: count1 of size1, then count2 of size2
- Achieves exact layout specified by the pattern

```python
def pack_orders_row_based(self, big_plate, orders, size1_key, size2_key,
                          count1_per_row, count2_per_row):
    # Places pieces in rows: 3×a + 2×b per row, 3 rows = 9a+6b
```

### 3. Intelligent Packing Selection
Modified `pack_orders()` to:
- Check if row-based pattern was detected
- Use custom row-based packing when applicable
- Fall back to rectpack for other cases

```python
def pack_orders(self, big_plate, orders):
    if self._detected_pattern and pattern['type'] == 'row':
        return self.pack_orders_row_based(...)  # Custom packing
    else:
        return rectpack_packing(...)  # Standard packing
```

## Algorithm Flow

1. **Detection Phase** (`find_complementary_pairs`):
   - Check if heights match (h1 ≈ h2)
   - Try all combinations: n1 pieces of size1 + n2 pieces of size2 per row
   - Calculate utilization for each combination
   - Store best pattern with details

2. **Sorting Phase** (`_sort_orders_for_optimal_packing`):
   - Store detected pattern in `self._detected_pattern`
   - Interleave pieces for better distribution

3. **Packing Phase** (`pack_orders`):
   - If row-based pattern detected: use custom row-based packing
   - Otherwise: use standard rectpack algorithm

## Test Results

### Test Case: 9a+6b Scenario
```
Big plate: 2440×1220mm
Type a: 527×400mm (with blade: 531×404mm)
Type b: 415×400mm (with blade: 419×404mm)

Result:
- Placed: 9a + 6b (15 pieces total)
- Utilization: 97.19%
- Pattern: 3 rows × (3a + 2b per row)
- Status: ✓ Success
```

### Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Pieces placed | 12a + 0b | 9a + 6b | +3 pieces |
| Utilization | 84.98% | 97.19% | +12.21% |
| Pattern detected | Column-based | Row-based | Correct |

## Key Features

1. **Automatic Detection**: No manual configuration needed
2. **Exact Placement**: Achieves theoretical optimal layout
3. **Backward Compatible**: Falls back to rectpack when row pattern not detected
4. **Efficient**: O(W/min_width) complexity for 2-piece same-height case

## Files Modified

- `backend/main.py`:
  - Enhanced `find_complementary_pairs()` to return pattern details
  - Added `pack_orders_row_based()` for custom row-based packing
  - Modified `_sort_orders_for_optimal_packing()` to store pattern
  - Updated `pack_orders()` to use custom packing when detected

## Files Added

- `backend/test_row_based.py`: Test demonstrating 9a+6b optimization

## Future Enhancements

1. **Column-based custom packing**: Similar approach for column patterns
2. **Multi-row patterns**: Handle more complex arrangements (e.g., 2 rows of a, 1 row of b)
3. **Rotation handling**: Better support for rotated pieces in row layouts
4. **Mixed patterns**: Combine row and column strategies in single plate

## Complexity Analysis

- **Detection**: O(s² × W/min_width) where s = unique sizes
- **Packing**: O(n) for row-based (vs O(n²) for rectpack)
- **Total**: O(s² × W/min_width + n) - very efficient for typical cases
