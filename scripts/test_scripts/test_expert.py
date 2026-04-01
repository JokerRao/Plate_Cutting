def test_expert_pattern():
    W, H = 2444, 1224
    w_a, h_a = 1018, 818
    w_b, h_b = 354, 1010
    
    # Try stacking vertically
    print(f"Stacking A({w_a}x{h_a}) and B({w_b}x{h_b}) vertically")
    sum_h = h_a + h_b
    if sum_h <= H or sum_h > H:
        # Wait, 818 + 1010 = 1828 > 1224. We cannot stack them vertically!
        print(f"sum_h={sum_h} > {H}")
        
    # Wait, in the successful SAT solution:
    # Big1=(1372, 355) 1018x818
    # Big2=(0, 354)  1018x818
    # Small1=(0,0)   1010x354
    # Small2=(1018,0) 354x1010  <-- Rotated!
    # Small3=(1372,1) 1010x354
    
    # Observe the blocks:
    # Block 1 (Left): Small1 (1010x354) + Big2(1018x818).
    # Vertically stacked! 354 + 818 = 1172 <= 1224!!
    # Block width = max(1010, 1018) = 1018!
    
    # Block 2 (Middle): Small2 (354x1010).
    # Width = 354.
    
    # Block 3 (Right): Small3 (1010x354) + Big1(1018x818).
    # Vertically stacked! 354 + 818 = 1172 <= 1224!!
    # Block width = max(1010, 1018) = 1018!

    # Total width: 1018 * 2 + 354 = 2390 <= 2444.
    
    print("\nThe pattern is:")
    print("Block type 1: Stack A (1018x818) and B_rotated (1010x354)")
    print("  Height = 818 + 354 = 1172 <= 1224")
    print("  Width = max(1018, 1010) = 1018")
    print("We fit 2 of these blocks. width used: 2036. Items: 2x A, 2x B_rotated")
    print("Remaining width: 2444 - 2036 = 408")
    print("In remaining width 408 x height 1224:")
    print("  We can fit B (354x1010) vertically! 354 <= 408. 1010 <= 1224.")
    print("  This adds 1x B. Total: 2x A, 3x B!")

if __name__ == '__main__':
    test_expert_pattern()
