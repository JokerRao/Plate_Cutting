"""大板模板克隆（避免迭代中污染尺寸）。"""
from core.models import SmallPlate


def clone_plate_template(p: SmallPlate) -> SmallPlate:
    return SmallPlate(
        length=p.length,
        width=p.width,
        plate_id=p.plate_id,
        quantity=p.quantity,
    )
