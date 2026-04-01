import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

from engine.complementary_pairs import find_complementary_pairs

# Use with-blade dimensions as the engine sees them
# Blade = 4. All dims here have blade added by the pipeline.
L, W = 2444, 1224  # 2440+4, 1220+4

# The three sizes (with blade)
size_groups = {
    (1018, 818): list(range(20)),  # 1014+4, 814+4
    (1010, 354): list(range(20)),  # 1006+4, 350+4
    (354, 810):  list(range(20)),  # 350+4, 806+4
}

comp, details = find_complementary_pairs(size_groups, L, W)

print("Detected complementary pairs:")
for key, gain in sorted(comp.items(), key=lambda x: -x[1]):
    print(f"  {key}: gain={gain:.4f}, pattern={details[key]}")
