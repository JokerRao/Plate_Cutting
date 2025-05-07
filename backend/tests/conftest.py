import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

@pytest.fixture
def test_settings():
    """测试配置"""
    return {
        "uid": "test_user",
        "project_id": "test_project",
        "standard_plate": {
            "id": "plate1",
            "length": 2000,
            "width": 1000,
            "quantity": 1,
            "description": "Standard plate"
        },
        "standard_order": {
            "id": "order1",
            "length": 500,
            "width": 300,
            "quantity": 1,
            "description": "Standard piece"
        }
    } 