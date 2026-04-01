import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

import rectpack

packer = rectpack.newPacker(
    mode=rectpack.PackingMode.Offline,
    bin_algo=rectpack.PackingBin.Global,
    pack_algo=rectpack.MaxRectsBaf,
    sort_algo=rectpack.SORT_PERI,
    rotation=True
)

packer.add_bin(2440, 1220)
for _ in range(942):
    packer.add_rect(404, 204)

packer.pack()

print(f"Total bins used: {len(packer)}")
for i, b in enumerate(packer):
    print(f"Bin {i}: {len(b)} pieces")
