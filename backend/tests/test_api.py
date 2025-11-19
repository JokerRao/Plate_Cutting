import pytest
from fastapi.testclient import TestClient
from fastapi import Request
from api import app, CuttingRequest, Plate, Order, StockPlate
import asyncio
import time

client = TestClient(app)

def test_successful_optimization():
    """测试成功的切割优化场景"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1,
                "description": "Standard plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 2,
                "description": "Small piece"
            },
            {
                "id": "order2",
                "length": 800,
                "width": 400,
                "quantity": 1,
                "description": "Medium piece"
            }
        ],
        "optimization": True,
        "saw_blade": 3
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["message"] == "Success"
    assert "cutting_plans" in data
    assert len(data["cutting_plans"]) > 0
    assert data["total_utilization"] > 0
    assert data["pieces_placed"] > 0
    assert data["plates_used"] > 0

def test_invalid_plate_dimensions():
    """测试无效的板材尺寸"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": -100,  # 无效的长度
                "width": 1000,
                "quantity": 1,
                "description": "Invalid plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1,
                "description": "Test piece"
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "detail" in data  # FastAPI validation error format

def test_invalid_order_dimensions():
    """测试无效的订单尺寸"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1,
                "description": "Standard plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 0,  # 无效的长度
                "width": 300,
                "quantity": 1,
                "description": "Invalid order"
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "detail" in data  # FastAPI validation error format

def test_oversized_pieces():
    """测试过大的订单尺寸"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 1000,
                "width": 500,
                "quantity": 1,
                "description": "Small plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 2000,  # 超过板材尺寸
                "width": 1000,
                "quantity": 1,
                "description": "Oversized piece"
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 1005
    assert "All order pieces are too large for available plates" in data["message"]

def test_partial_success():
    """测试部分成功的情况（有未放置的件）"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1,
                "description": "Standard plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 1000,
                "width": 800,
                "quantity": 3,  # 需要放置3个，但板材可能放不下
                "description": "Large piece"
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "Partial success" in data["message"]
    assert "unplaced_pieces" in data
    assert data["warnings"] is not None

def test_optimization_details():
    """测试优化细节信息"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1,
                "description": "Standard plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1,
                "description": "Test piece"
            }
        ],
        "optimization": True,
        "saw_blade": 3
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "optimization_details" in data
    assert "saw_blade_width" in data["optimization_details"]
    assert "optimization_enabled" in data["optimization_details"]
    assert "total_pieces_requested" in data["optimization_details"]

def test_invalid_quantity():
    """测试无效的数量"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 0,  # 无效的数量
                "description": "Invalid quantity"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1,
                "description": "Test piece"
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "detail" in data  # FastAPI validation error format

def test_zero_quantity():
    """测试数量为零的情况"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 0  # 无效的数量
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_missing_required_fields():
    """测试缺少必填字段的情况"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                # 缺少 length 和 width
                "quantity": 1
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1
            }
        ]
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_optimization_with_stock_pieces():
    """测试包含库存件的优化"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1,
                "description": "Standard plate"
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1,
                "description": "New piece"
            }
        ],
        "others": [
            {
                "id": "stock1",
                "length": 400,
                "width": 200,
                "description": "Stock piece",
                "client": "client1"
            }
        ],
        "optimization": True
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "optimization_details" in data
    assert data["optimization_details"]["stock_pieces_available"] == 1

@pytest.mark.asyncio
async def test_concurrent_requests():
    """测试并发请求处理"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1
            }
        ],
        "optimization": True
    }

    async def make_request():
        try:
            response = client.post("/optimize", json=request_data)
            return response.status_code
        except Exception as e:
            return 500

    # 同时发送10个请求
    tasks = [make_request() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # 检查结果
    success_count = sum(1 for status in results if status == 200)
    rate_limited_count = sum(1 for status in results if status == 429)
    
    assert success_count > 0  # 至少有一些请求成功
    assert rate_limited_count > 0  # 至少有一些请求被限制

def test_rate_limiting():
    """测试速率限制"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1
            }
        ],
        "optimization": True
    }

    responses = []
    for _ in range(10):  # 发送更多请求以确保触发限制
        response = client.post("/optimize", json=request_data)
        responses.append(response)

    # 至少有一个请求应该被限制
    assert any(r.status_code == 429 for r in responses)
    # 至少有一个请求应该成功
    assert any(r.status_code == 200 for r in responses)

def test_saw_blade_parameter():
    """测试锯片宽度参数"""
    request_data = {
        "uid": "test_user",
        "project_id": "test_project",
        "plates": [
            {
                "id": "plate1",
                "length": 2000,
                "width": 1000,
                "quantity": 1
            }
        ],
        "orders": [
            {
                "id": "order1",
                "length": 500,
                "width": 300,
                "quantity": 1
            }
        ],
        "optimization": True,
        "saw_blade": 5.5  # 指定锯片宽度（支持小数）
    }

    response = client.post("/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["optimization_details"]["saw_blade_width"] == 5.5

def test_health_check():
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# 在其他测试之间添加延迟
@pytest.fixture(autouse=True)
def delay_between_tests():
    """在每个测试之间添加延迟以避免速率限制影响"""
    yield
    time.sleep(0.5)  # 500ms delay between tests 