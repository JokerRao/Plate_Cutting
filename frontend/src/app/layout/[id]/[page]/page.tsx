'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { supabase } from '@/utils/supabaseClient';

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
          setOrders(metadata.orders || []);
          setOthers(metadata.others || []);
          
          // 移除最后一个元素（元数据），只保留切板方案
          const cuttingPlans = data.cutted.slice(0, -1);
          setTotalPages(cuttingPlans.length);
          setAllCutted(cuttingPlans);
          
          if (cuttingPlans[pageNum - 1]) {
            setLayoutData(cuttingPlans[pageNum - 1]);
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
    // 基础边距，用于防止标注文字被裁切
    const margin = 80;
    
    // 我们希望整个板材加上padding后能居中显示
    // 首先计算可用的绘制区域大小
    const availableWidth = canvas.width - margin * 2;
    const availableHeight = canvas.height - margin * 2;
    
    // 计算缩放比例，使得板材完全填满可用区域
    const scale = Math.min(
      availableWidth / plateLength,
      availableHeight / plateWidth
    );
    
    // 计算板材在画布上的实际像素尺寸
    const pixelPlateLength = plateLength * scale;
    const pixelPlateWidth = plateWidth * scale;
    
    // 计算居中偏移量，使板材图形本身在画布上居中
    const offsetX = (canvas.width - pixelPlateLength) / 2;
    const offsetY = (canvas.height - pixelPlateWidth) / 2;

    // 坐标系变换：原点移到左下角，Y轴向上，并应用居中偏移
    // 我们将原点移动到板材左下角的实际像素位置
    ctx.translate(offsetX, canvas.height - offsetY);
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
    ctx.fillStyle = '#f2f3f5';
    ctx.fillRect(0, 0, plateLength * scale, plateWidth * scale);
    ctx.strokeStyle = '#1d1e20';
    ctx.strokeRect(0, 0, plateLength * scale, plateWidth * scale);

    // 标注大板尺寸
    ctx.scale(1, -1);
    ctx.fillStyle = '#1d1e20';
    ctx.font = '14px system-ui, sans-serif';
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
        ctx.fillStyle = !is_stock ? '#b8d4ff' : '#e5dfd6';
        ctx.fillRect(
          start_x * scale,
          start_y * scale,
          length * scale,
          width * scale
        );
        
        // 绘制板件边框
        ctx.strokeStyle = '#1d1e20';
        ctx.strokeRect(
          start_x * scale,
          start_y * scale,
          length * scale,
          width * scale
        );

        // 标注尺寸和ID
        ctx.scale(1, -1);
        ctx.fillStyle = '#1d1e20';
        ctx.font = '12px system-ui, sans-serif';
        
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
        ctx.font = '14px system-ui, sans-serif';
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
    // 计算当前页面中每个ID的使用数量
    const pageUsageCount = new Map<string, number>();
    if (layoutData && layoutData.cutted) {
      layoutData.cutted.forEach((piece: any) => {
        const id = String(piece.id);
        if ((type === 'orders' && !piece.is_stock) || (type === 'others' && piece.is_stock)) {
          pageUsageCount.set(id, (pageUsageCount.get(id) || 0) + 1);
        }
      });
    }

    // 计算所有页面中的使用数量
    const totalUsageCount = new Map<string, number>();
    if (allCutted) {
      allCutted.forEach(page => {
        if (page.cutted) {
          page.cutted.forEach((piece: any) => {
            const id = String(piece.id);
            if ((type === 'orders' && !piece.is_stock) || (type === 'others' && piece.is_stock)) {
              totalUsageCount.set(id, (totalUsageCount.get(id) || 0) + 1);
            }
          });
        }
      });
    }

    // 过滤数据，只显示在当前页面有使用的零件
    const filteredData = data.filter((item) => {
      const pageCount = pageUsageCount.get(String(item.id)) || 0;
      return pageCount > 0;
    });

    if (filteredData.length === 0) return null;

    return (
      <div className="table-container hover-lift shadow-sm animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
        <div className="table-title flex items-center gap-2">
          {type === 'orders' ? (
            <svg className="w-4 h-4 text-[#9333ea]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
          ) : (
            <svg className="w-4 h-4 text-[#d97706]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>
          )}
          {title}
        </div>
        <div className="table-content">
          <table className="w-full">
            <thead>
              <tr>
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
                <tr
                  key={item.id}
                  className={
                    type === 'orders'
                      ? 'border-l-2 border-l-accent bg-muted/60'
                      : 'border-l-2 border-l-accent-green bg-muted/60'
                  }
                >
                  <td className="border p-2">{type === 'others' ? `R${item.id}` : item.id}</td>
                  <td className="border p-2">{item.length}</td>
                  <td className="border p-2">{item.width}</td>
                  <td className="border p-2">{type === 'others' ? totalUsageCount.get(String(item.id)) || 0 : item.quantity}</td>
                  <td className="border p-2">{pageUsageCount.get(String(item.id)) || 0}</td>
                  {type === 'others' && <td className="border p-2">{item.client}</td>}
                  <td className="border p-2">{item.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  if (isLoading && !layoutData) return (
    <div className="page-gallery flex h-screen items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-hairline border-t-accent" />
    </div>
  );

  return (
    <div className="page-gallery">
      <div className="page-gallery-inner">
      {/* 通知 */}
      {notification && (
        <div className="mb-6 animate-fade-out">
          <div className="border border-hairline border-l-2 border-l-accent bg-muted px-4 py-3" style={{ borderRadius: 2 }}>
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 shrink-0 text-accent" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <p className="text-sm text-ink">{notification.message}</p>
            </div>
          </div>
        </div>
      )}

      {/* 导航 */}
      <div className="mb-6 flex gap-2">
        <button 
          type="button"
          className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border border-transparent text-text-secondary hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
          onClick={() => router.push(`/project/${projectId}`)}
        >
          项目配置
        </button>
        <button 
          type="button"
          className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border border-transparent text-text-secondary hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
          onClick={() => router.push(`/layout/${projectId}`)}
        >
          切板统计
        </button>
        <span className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border border-hairline bg-surface shadow-sm text-foreground">
          板材切割
        </span>
      </div>

      {/* 标题和分页 */}
      <div className="mb-8 flex flex-col gap-6 border-b border-hairline pb-4 md:flex-row md:items-center md:justify-between animate-fade-in-up">
        <h1 className="text-2xl font-semibold tracking-tight text-ink flex items-center gap-3">
          <svg className="w-6 h-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" /></svg>
          {projectName || '未命名项目'} — 第 {pageNum} 页
        </h1>
        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            <button
              type="button"
              className={
                pageNum > 1 && !isTransitioning ? 'btn-gallery-primary flex items-center gap-1.5 shadow-sm px-4 py-2 text-sm' : 'btn-gallery-secondary flex items-center gap-1.5 shadow-sm px-4 py-2 text-sm'
              }
              onClick={() => handlePageChange(pageNum - 1)}
              disabled={pageNum <= 1 || isTransitioning}
            >
              {isTransitioning ? '加载中…' : '上一页'}
            </button>
            <button
              type="button"
              className={
                pageNum < totalPages && !isTransitioning ? 'btn-gallery-primary flex items-center gap-1.5 shadow-sm px-4 py-2 text-sm' : 'btn-gallery-secondary flex items-center gap-1.5 shadow-sm px-4 py-2 text-sm'
              }
              onClick={() => handlePageChange(pageNum + 1)}
              disabled={pageNum >= totalPages || isTransitioning}
            >
              {isTransitioning ? '加载中…' : '下一页'}
            </button>
          </div>
          
          <div className="h-4 w-[1px] bg-border-hairline hidden md:block"></div>
          
          <button 
            type="button" 
            className="btn-gallery-secondary flex items-center gap-1.5 shadow-sm px-4 py-2 text-sm hidden md:flex" 
            onClick={() => router.push('/project')}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
            <span>返回列表</span>
          </button>
        </div>
      </div>
      
      <p className="mb-8 text-sm text-ink-muted">
        使用率: {(layoutData?.rate * 100).toFixed(1)}%
      </p>

      {/* 画布 */}
      <div className="mb-10 flex justify-center hover-lift transition-all animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
        <div className={isTransitioning ? 'opacity-50' : ''}>
          <canvas
            ref={canvasRef}
            className="border-hairline border bg-surface shadow-sm w-full"
            style={{ maxWidth: '100%', height: 'auto', borderRadius: 6 }}
          />
        </div>
      </div>

      {/* 表格 */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {renderTable('零件信息', orders, 'orders')}
        {renderTable('常用尺寸信息', others, 'others')}
      </div>
      </div>
    </div>
  );
} 