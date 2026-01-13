#!/usr/bin/env python3
"""
测试互补尺寸检测功能
验证混合组合（如4a+6c）优于单一尺寸（8a或10c）的场景
"""

from main import PlateOptimizer, SmallPlate, CuttingConfig

def test_complementary_detection():
    """测试互补尺寸检测"""
    config = CuttingConfig(blade_thickness=4.0)
    optimizer = PlateOptimizer(config)

    # 大板尺寸：2000x1000
    big_plate = SmallPlate(length=2000, width=1000, plate_id="big1")

    # 订单：
    # - 尺寸a: 400x200 (加锯片后 404x204)
    # - 尺寸c: 300x150 (加锯片后 304x154)
    #
    # 单一尺寸：
    # - 8个a: (2000//404) * (1000//204) = 4 * 4 = 16个位置，但只需8个
    # - 10个c: (2000//304) * (1000//154) = 6 * 6 = 36个位置，但只需10个
    #
    # 混合组合：
    # - 4个a + 6个c 可能更优

    orders = []
    # 添加8个尺寸a的订单
    for i in range(8):
        orders.append(SmallPlate(length=400, width=200, plate_id=f"a{i}"))

    # 添加10个尺寸c的订单
    for i in range(10):
        orders.append(SmallPlate(length=300, width=150, plate_id=f"c{i}"))

    # 执行排序
    sorted_orders = optimizer._sort_orders_for_optimal_packing(orders, big_plate)

    print(f"\n总订单数: {len(orders)}")
    print(f"排序后的订单数: {len(sorted_orders)}")

    # 检查是否使用了互补策略
    # 如果使用了互补策略，应该会交错排列a和c
    order_types = []
    for idx, order, rotate in sorted_orders:
        if order.length == 400:
            order_types.append('a')
        elif order.length == 300:
            order_types.append('c')

    print(f"\n排序后的订单类型序列: {' '.join(order_types[:20])}")

    # 检查是否有交错模式
    has_interleaving = False
    for i in range(len(order_types) - 1):
        if order_types[i] != order_types[i+1]:
            has_interleaving = True
            break

    if has_interleaving:
        print("✓ 检测到交错排列，互补策略已启用")
    else:
        print("✗ 未检测到交错排列，可能使用了其他策略")

    # 实际执行装箱
    cuts, remaining = optimizer.pack_orders(big_plate, orders)

    print(f"\n装箱结果:")
    print(f"- 成功装入: {len(cuts)} 个")
    print(f"- 剩余未装: {len(remaining)} 个")

    # 计算利用率
    total_area = sum(cut.plate.length * cut.plate.width for cut in cuts)
    big_plate_area = big_plate.length * big_plate.width
    utilization = total_area / big_plate_area * 100

    print(f"- 利用率: {utilization:.2f}%")

    return cuts, remaining, utilization

if __name__ == "__main__":
    print("=" * 60)
    print("测试互补尺寸检测功能")
    print("=" * 60)

    cuts, remaining, utilization = test_complementary_detection()

    print("\n" + "=" * 60)
    print(f"测试完成 - 利用率: {utilization:.2f}%")
    print("=" * 60)
