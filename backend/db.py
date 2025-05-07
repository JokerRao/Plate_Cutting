from supabase import create_client, Client
from config import get_settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_supabase_client() -> Client:
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

async def sync_cutting_results(
    uid: str,
    project_id: str,
    cutting_plans: List[Dict[str, Any]],
    plates: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    others: List[Dict[str, Any]]
) -> bool:
    """
    将切板结果同步到 Supabase
    
    Args:
        uid: 用户ID
        project_id: 项目ID
        cutting_plans: 切板方案列表（包含 orders 和 others 的元数据）
        plates: 板材列表（未使用）
        orders: 订单列表（未使用）
        others: 其他尺寸列表（未使用）
        
    Returns:
        bool: 同步是否成功
    """
    try:
        supabase = get_supabase_client()
        
        # 准备要更新的数据
        update_data = {
            'cutted': cutting_plans,  # 切板方案（包含所有必要信息）
            'updated_at': 'now()'  # 更新时间
        }

        # 更新数据库
        result = supabase.table('Projects').update(update_data).eq('id', project_id).eq('uid', uid).execute()
        
        # 检查响应状态
        if hasattr(result, 'status_code') and result.status_code >= 400:
            logger.error(f"Failed to sync cutting results: HTTP {result.status_code}")
            return False
            
        logger.info(f"Successfully synced cutting results for project {project_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error syncing cutting results: {str(e)}")
        return False 