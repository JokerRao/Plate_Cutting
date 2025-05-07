'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { supabase } from '@/utils/supabaseClient';

interface CuttedItem {
  start_x: number;
  start_y: number;
  length: number;
  width: number;
  type: 0 | 1;  // 0: 零件, 1: 常用尺寸
  id: string;
}

export default function LayoutPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;
  const pageNum = parseInt(params.page as string);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [notification, setNotification] = useState<{message: string, type: 'warning'} | null>(null);
  const [projectName, setProjectName] = useState('');
  const [layoutData, setLayoutData] = useState<any>(null);
  const [orders, setOrders] = useState<any[]>([]);
  const [others, setOthers] = useState<any[]>([]);
  const [ordersCutted, setOrdersCutted] = useState<any[]>([]);
  const [othersCutted, setOthersCutted] = useState<any[]>([]);
  const [totalPages, setTotalPages] = useState(0);
  const [allCutted, setAllCutted] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);  // 添加加载状态
  const [isTransitioning, setIsTransitioning] = useState(false);  // 添加过渡状态

  // 显示通知的函数
  const showNotification = (message: string) => {
    setNotification({ message, type: 'warning' });
    setTimeout(() => {
      setNotification(null);
    }, 5000); // 5秒后消失
  };

  // 预加载下一页数据
  const preloadNextPage = async (nextPageNum: number) => {
    if (nextPageNum > 0 && nextPageNum <= totalPages) {
      const nextPageData = allCutted[nextPageNum - 1];
      if (nextPageData) {
        setLayoutData(nextPageData);
      }
    }
  };

  // 处理页面切换
  const handlePageChange = async (newPageNum: number) => {
    if (newPageNum < 1 || newPageNum > totalPages || newPageNum === pageNum) return;
    
    setIsTransitioning(true);
    setIsLoading(true);
    
    try {
      // 预加载下一页数据
      await preloadNextPage(newPageNum);
      
      // 使用 router.push 进行页面切换
      router.push(`/layout/${projectId}/${newPageNum}`);
    } catch (error) {
      console.error('Error changing page:', error);
      showNotification('页面切换失败，请重试');
    } finally {
      setIsLoading(false);
      setIsTransitioning(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const { data } = await supabase
          .from('Projects')
          .select('name, cutted')
          .eq('id', projectId)
          .single();
        
        if (data && data.cutted) {
          setProjectName(data.name);
          // 获取最后一个元素（包含元数据）
          const metadata = data.cutted[data.cutted.length - 1]?.metadata || {};
          console.log('Metadata:', metadata); // 添加日志
          setOrders(metadata.orders || []);
          setOthers(metadata.others || []);
          
          // 移除最后一个元素（元数据），只保留切板方案
          const cuttingPlans = data.cutted.slice(0, -1);
          setTotalPages(cuttingPlans.length);
          setAllCutted(cuttingPlans);
          
          if (cuttingPlans[pageNum - 1]) {
            setLayoutData(cuttingPlans[pageNum - 1]);
            console.log('Current page data:', cuttingPlans[pageNum - 1]); // 添加日志
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        showNotification('数据加载失败，请刷新页面重试');
      } finally {
        setIsLoading(false);
      }
    };
    
    if (projectId) fetchData();
  }, [projectId, pageNum]);

  useEffect(() => {
    if (!layoutData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置画布大小
    canvas.width = 1200;
    canvas.height = 800;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 计算缩放比例
    const [plateLength, plateWidth] = layoutData.plate;
    const margin = 60;
    const scale = Math.min(
      (canvas.width - margin * 2) / plateLength,
      (canvas.height - margin * 2) / plateWidth
    );

    // 坐标系变换：原点移到左下角，Y轴向上
    ctx.translate(margin, canvas.height - margin);
    ctx.scale(1, -1);

    // 绘制坐标轴
    ctx.beginPath();
    ctx.strokeStyle = '#000';
    ctx.moveTo(0, 0);
    ctx.lineTo(40, 0);
    ctx.moveTo(0, 0);
    ctx.lineTo(0, 40);
    ctx.stroke();

    // 标注原点
    ctx.scale(1, -1);
    ctx.font = '14px Arial';
    ctx.fillStyle = '#000';
    ctx.fillText('O(0,0)', -20, 20);
    ctx.scale(1, -1);

    // 绘制大板
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, plateLength * scale, plateWidth * scale);
    ctx.strokeStyle = '#000';
    ctx.strokeRect(0, 0, plateLength * scale, plateWidth * scale);

    // 标注大板尺寸
    ctx.scale(1, -1);
    ctx.fillStyle = '#000';
    ctx.font = '14px Arial';
    ctx.fillText(`${plateLength}mm`, plateLength * scale / 2, 20);
    ctx.save();
    ctx.translate(plateLength * scale + 20, -plateWidth * scale / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillText(`${plateWidth}mm`, 0, 0);
    ctx.restore();
    ctx.scale(1, -1);

    // 绘制切割的板件
    if (layoutData.cutted && Array.isArray(layoutData.cutted)) {
      layoutData.cutted.forEach((piece: any) => {
        const { start_x, start_y, length, width, is_stock, id } = piece;
        
        // 绘制板件填充
        ctx.fillStyle = !is_stock ? '#93c5fd' : '#fcd34d';
        ctx.fillRect(
          start_x * scale,
          start_y * scale,
          length * scale,
          width * scale
        );
        
        // 绘制板件边框
        ctx.strokeStyle = '#000';
        ctx.strokeRect(
          start_x * scale,
          start_y * scale,
          length * scale,
          width * scale
        );

        // 标注尺寸和ID
        ctx.scale(1, -1);
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        
        // 长度标注 - 固定距离15px
        const lengthText = `${length}mm`;
        const lengthWidth = ctx.measureText(lengthText).width;
        ctx.fillText(
          lengthText,
          (start_x + length/2) * scale - lengthWidth/2,
          -(start_y * scale) - 15  // 固定在上方15px处
        );
        
        // 宽度标注 - 固定距离15px
        const widthText = `${width}mm`;
        const widthWidth = ctx.measureText(widthText).width;
        ctx.save();
        ctx.translate(
          (start_x * scale) + 15,  // 固定在左侧15px处
          -(start_y + width/2) * scale
        );
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(widthText, -widthWidth/2, 0);
        ctx.restore();

        // ID标注 - 保持在中心
        ctx.font = '14px Arial';
        const idText = !is_stock ? `${id}` : `R${id}`;
        const idWidth = ctx.measureText(idText).width;
        ctx.fillText(
          idText,
          (start_x + length/2) * scale - idWidth/2,
          -(start_y + width/2) * scale
        );
        
        ctx.scale(1, -1);
      });
    }
  }, [layoutData]);

  // 在客户端添加动画样式
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeOut {
        0% { opacity: 1; transform: translateY(0); }
        70% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-100%); }
      }
      .animate-fade-out {
        animation: fadeOut 5s forwards;
      }
    `;
    document.head.appendChild(style);
    
    // 清理函数
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const renderTable = (title: string, data: any[], type: 'orders' | 'others') => {
    console.log(`Rendering ${title} table:`, { data, type }); // 添加日志
    
    // 计算当前页面中每个ID的使用数量
    const pageUsageCount = new Map<string, number>();
    if (layoutData && layoutData.cutted) {
      layoutData.cutted.forEach((piece: any) => {
        const id = piece.id;
        if ((type === 'orders' && !piece.is_stock) || (type === 'others' && piece.is_stock)) {
          pageUsageCount.set(id, (pageUsageCount.get(id) || 0) + 1);
        }
      });
    }
    console.log('Page usage count:', Object.fromEntries(pageUsageCount)); // 添加日志

    // 计算所有页面中的使用数量
    const totalUsageCount = new Map<string, number>();
    if (allCutted) {
      allCutted.forEach(page => {
        if (page.cutted) {
          page.cutted.forEach((piece: any) => {
            const id = piece.id;
            if ((type === 'orders' && !piece.is_stock) || (type === 'others' && piece.is_stock)) {
              totalUsageCount.set(id, (totalUsageCount.get(id) || 0) + 1);
            }
          });
        }
      });
    }
    console.log('Total usage count:', Object.fromEntries(totalUsageCount)); // 添加日志

    // 过滤数据，只显示在当前页面有使用的零件
    const filteredData = data.filter((item) => {
      const pageCount = pageUsageCount.get(item.id) || 0;
      return pageCount > 0;
    });
    console.log('Filtered data:', filteredData); // 添加日志

    if (filteredData.length === 0) return null;

    return (
      <div className="flex-1">
        <h3 className="font-semibold mb-2">{title}</h3>
        <table className="w-full border">
          <thead>
            <tr className="bg-gray-50">
              <th className="border p-2">编号</th>
              <th className="border p-2">长度</th>
              <th className="border p-2">宽度</th>
              <th className="border p-2">总数量</th>
              <th className="border p-2">本页数量</th>
              {type === 'others' && <th className="border p-2">客户</th>}
              <th className="border p-2">描述</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map((item) => (
              <tr key={item.id} className={type === 'orders' ? 'bg-blue-200' : 'bg-yellow-200'}>
                <td className="border p-2">{type === 'others' ? `R${item.id}` : item.id}</td>
                <td className="border p-2">{item.length}</td>
                <td className="border p-2">{item.width}</td>
                <td className="border p-2">{type === 'others' ? totalUsageCount.get(item.id) || 0 : item.quantity}</td>
                <td className="border p-2">{pageUsageCount.get(item.id) || 0}</td>
                {type === 'others' && <td className="border p-2">{item.client}</td>}
                <td className="border p-2">{item.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  if (isLoading && !layoutData) return (
    <div className="flex items-center justify-center h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
    </div>
  );

  return (
    <div className="relative max-w-7xl mx-auto my-8 rounded-2xl shadow-2xl border bg-white flex flex-col h-[92vh]">
      {/* 通知组件 */}
      {notification && (
        <div className="absolute top-0 left-0 right-0 z-50 animate-fade-out">
          <div className="mx-auto max-w-md px-4 py-2 bg-yellow-100 border-l-4 border-yellow-500 rounded-b">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-yellow-700">
                  {notification.message}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* 导航和标题部分保持不变 */}
      <div className="flex items-center px-6 pt-1">
        <div className="flex gap-2 mb-2">
          <button 
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold"
            onClick={() => router.push(`/project/${projectId}`)}
          >
            项目
          </button>
          <button 
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold"
            onClick={() => router.push(`/layout/${projectId}`)}
          >
            切板统计
          </button>
        </div>
      </div>

      <div className="px-6 border-b">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-bold">
            {projectName || '未命名项目'} - 第 {pageNum} 页
          </h1>
          <div className="flex gap-2">
            <button
              className={`px-4 py-1 rounded text-sm transition-all duration-200 ${
                pageNum > 1
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              } ${isTransitioning ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={() => handlePageChange(pageNum - 1)}
              disabled={pageNum <= 1 || isTransitioning}
            >
              {isTransitioning ? '加载中...' : '上一页'}
            </button>
            <button
              className={`px-4 py-1 rounded text-sm transition-all duration-200 ${
                pageNum < totalPages
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              } ${isTransitioning ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={() => handlePageChange(pageNum + 1)}
              disabled={pageNum >= totalPages || isTransitioning}
            >
              {isTransitioning ? '加载中...' : '下一页'}
            </button>
          </div>
        </div>
        <p className="text-xs text-gray-600 -mt-1">
          使用率: {(layoutData?.rate * 100).toFixed(1)}%
        </p>
      </div>

      {/* 内容区：上方画布，下方表格 */}
      <div className="flex-1 px-6 pt-1 pb-4 flex flex-col">
        {/* 上方画布 */}
        <div className="flex-1 flex justify-center items-start mb-4">
          <div className={`transition-opacity duration-300 ${isTransitioning ? 'opacity-50' : 'opacity-100'}`}>
            <canvas
              ref={canvasRef}
              className="rounded-lg border border-gray-200"
              style={{ maxWidth: '100%', height: 'auto' }}
            />
          </div>
        </div>

        {/* 下方表格 */}
        <div className="flex gap-4 h-56">
          {renderTable('零件信息', orders, 'orders')}
          {renderTable('常用尺寸信息', others, 'others')}
        </div>
      </div>
    </div>
  );
} 