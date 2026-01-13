#!/usr/bin/env python3
"""
测试行式混合装箱功能
验证 9a+6b 的最优解（大板 2440×1220，a=527×400，b=415×400）
"""

from main import PlateOptimizer, SmallPlate, CuttingConfig

def test_row_based_packing():
    """测试行式混合装箱 - 9a+6b 场景"""
    config = CuttingConfig(blade_thickness=4.0)
    optimizer = PlateOptimizer(config)

    # 大板尺寸：2440×1220
    big_plate = SmallPlate(length=2440, width=1220, plate_id="big1")

    # 订单：
    # - 尺寸a: 527×400 (加锯片后 531×404)
    # - 尺寸b: 415×400 (加锯片后 419×404)
    #
    # 预期最优解：
    # - 每行：3×a (1593mm) + 2×b (838mm) = 2431mm (剩余9mm)
    # - 3行：1212mm (剩余8mm)
    # - 总计：9×a + 6×b，利用率 97.2%

    orders = []
    # 添加200个尺寸a的订单
    for i in range(200):
        orders.append(SmallPlate(length=527, width=400, plate_id=f"a{i}"))

    # 添加200个尺寸b的订单
    for i in range(200):
        orders.append(SmallPlate(length=415, width=400, plate_id=f"b{i}"))

    print("=" * 70)
    print("测试行式混合装箱 - 9a+6b 场景")
    print("=" * 70)
    print(f"\n大板尺寸: {big_plate.length}×{big_plate.width}mm")
    print(f"尺寸a: 527×400mm (加锯片: 531×404mm) - 200件")
    print(f"尺寸b: 415×400mm (加锯片: 419×404mm) - 200件")
    print(f"\n预期最优解: 9a+6b 每板，利用率 97.2%")

    # 执行排序
    sorted_orders = optimizer._sort_orders_for_optimal_packing(orders, big_plate)

    print(f"\n排序后的订单数: {len(sorted_orders)}")

    # 检查是否使用了互补策略
    order_types = []
    for idx, order, rotate in sorted_orders[:30]:  # 只看前30个
        if order.length == 527:
            order_types.append('a')
        elif order.length == 415:
            order_types.append('b')

    print(f"排序后的订单类型序列（前30个）: {' '.join(order_types)}")

    # 实际执行装箱
    cuts, remaining = optimizer.pack_orders(big_plate, orders)

    print(f"\n装箱结果:")
    print(f"- 成功装入: {len(cuts)} 个")
    print(f"- 剩余未装: {len(remaining)} 个")

    # 统计每种类型的数量
    a_count = sum(1 for cut in cuts if cut.plate.length == 527)
    b_count = sum(1 for cut in cuts if cut.plate.length == 415)

    print(f"\n装入详情:")
    print(f"- 尺寸a: {a_count} 个")
    print(f"- 尺寸b: {b_count} 个")

    # 计算利用率
    total_area = sum(cut.plate.length * cut.plate.width for cut in cuts)
    big_plate_area = big_plate.length * big_plate.width
    utilization = total_area / big_plate_area * 100

    print(f"\n利用率: {utilization:.2f}%")

    # 检查是否接近预期的9a+6b
    if abs(a_count - 9) <= 1 and abs(b_count - 6) <= 1:
        print(f"✓ 成功！接近预期的 9a+6b 组合")
    else:
        print(f"✗ 未达到预期的 9a+6b 组合")

    # 检查利用率是否接近97.2%
    if utilization >= 95.0:
        print(f"✓ 利用率优秀 (≥95%)")
    elif utilization >= 90.0:
        print(f"○ 利用率良好 (≥90%)")
    else:
        print(f"✗ 利用率偏低 (<90%)")

    return cuts, remaining, utilization, a_count, b_count

if __name__ == "__main__":
    cuts, remaining, utilization, a_count, b_count = test_row_based_packing()

    print("\n" + "=" * 70)
    print(f"测试完成")
    print(f"实际组合: {a_count}a + {b_count}b")
    print(f"利用率: {utilization:.2f}%")
    print("=" * 70)
