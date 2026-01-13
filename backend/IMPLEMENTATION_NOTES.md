# Hybrid Complementary Packing Algorithm Implementation

## Overview
Implemented a hybrid approach combining complementary size detection with the existing multi-strategy evaluation to solve the mixed-combination optimization problem.

## Problem Solved
The original algorithm failed to recognize that mixed combinations (e.g., 4a+6c plates) could be more efficient than single-size arrangements (e.g., 8a or 10c plates).

## Solution Architecture

### 1. Complementary Pair Detection (`find_complementary_pairs`)
- **Purpose**: Identifies size pairs that tile together better than individually
- **Algorithm**:
  - For each size pair, tries different column allocations (n1 columns of size1, remaining for size2)
  - Calculates utilization for each allocation
  - Records pairs where mixed utilization exceeds single-size by ≥2%
- **Complexity**: O(s² × L/w_min) where s = number of unique sizes

### 2. Modified Sorting Logic (`_sort_orders_for_optimal_packing`)
- **Step 1**: Group orders by optimal size (considering rotation)
- **Step 2**: Run complementary pair detection
- **Step 3**: If complementary pairs found:
  - Select the pair with highest utilization gain
  - Interleave the two sizes in the order list
  - Append remaining sizes
- **Step 4**: If no complementary pairs found:
  - Fall back to existing multi-strategy evaluation (round-robin, greedy, balanced)

## Key Features

### Automatic Detection
```python
complementary = self.find_complementary_pairs(size_groups, length0, width0)
```
- Automatically identifies optimal size combinations
- No manual configuration required

### Interleaving Strategy
```python
for i in range(max_len):
    if i < len(group1):
        result.append(group1[i])
    if i < len(group2):
        result.append(group2[i])
```
- Alternates between complementary sizes
- Maximizes packing efficiency

### Logging
- Logs when complementary pairs are found with utilization gain
- Logs which strategy is being used (complementary vs multi-strategy)

## Test Results

### Test Case: 8×a plates + 10×c plates
- **Plate dimensions**: 2000×1000mm
- **Size a**: 400×200mm (with blade: 404×204mm)
- **Size c**: 300×150mm (with blade: 304×154mm)

**Results**:
- ✓ Complementary pair detected: 12.53% utilization gain
- ✓ Interleaving pattern: `a c a c a c a c...`
- ✓ All 18 pieces fit on single plate
- ✓ Utilization: 54.50%

## Performance Impact

### Time Complexity
- **Before**: O(n log n) - simple sorting
- **After**: O(s² × L/w + n) - complementary detection + interleaving
- **Typical case**: s < 10, so O(s²) is negligible

### Space Complexity
- O(s²) for complementary pairs dictionary
- O(n) for order grouping
- Total: O(n + s²)

## Integration

### No Breaking Changes
- Existing API remains unchanged
- Falls back to original strategy if no complementary pairs found
- All existing tests pass

### Configuration
- Threshold for complementary detection: 2% utilization gain
- Can be adjusted in `find_complementary_pairs` method

## Future Enhancements

1. **Multi-pair optimization**: Consider more than 2 complementary sizes
2. **Dynamic threshold**: Adjust 2% threshold based on plate size
3. **Rotation optimization**: Better handling of rotated complementary pairs
4. **Caching**: Cache complementary pairs for repeated similar requests

## Files Modified

- `backend/main.py`:
  - Added `find_complementary_pairs()` method (lines 557-599)
  - Modified `_sort_orders_for_optimal_packing()` method (lines 601-907)
  - Fixed integer conversion issues for range() compatibility

## Testing

- All existing tests pass
- New test file: `test_complementary.py` demonstrates the feature
- Run with: `python3 test_complementary.py`
