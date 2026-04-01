import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

from core.models import CuttingConfig, SmallPlate
from services.cutting_service import _pre_sort_composite_affinity

config = CuttingConfig(blade_thickness=4.0)

# Simulate what small_plates looks like after prepare.py  
# small_plates = order pieces with blade_thickness already factored in (length/width are RAW)
# Actually SmallPlate stores the raw order dimensions. Blade is added during packing.

big = [SmallPlate(length=2440, width=1220, plate_id="p1")]  # raw dims (blade added separately)

smalls = []
for _ in range(20):
    smalls.append(SmallPlate(length=1014, width=814, plate_id="s1"))  # Big orders
for _ in range(20):
    smalls.append(SmallPlate(length=1006, width=350, plate_id="s2"))  # Small_1006
for _ in range(20):
    smalls.append(SmallPlate(length=350, width=806, plate_id="s3"))   # Small_806

result = _pre_sort_composite_affinity(smalls, big, config)
print("Pre-sort grouping:")
from collections import Counter
keys = []
for p in result:
    keys.append((p.length, p.width))
for k, cnt in Counter(keys).items():
    print(f"  {k}: {cnt} pieces")
print(f"\nOrder: {[(p.length, p.width) for p in result[:10]]}...")
