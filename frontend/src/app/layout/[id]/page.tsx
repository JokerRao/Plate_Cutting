'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { supabase } from '@/utils/supabaseClient';

/** 页面卡片区分：低饱和中性底，避免彩虹色 */
const COLOR_POOL = [
  { bg: 'bg-surface', text: 'text-ink-muted' },
  { bg: 'bg-muted', text: 'text-ink-muted' },
  { bg: 'bg-[#eef0f2]', text: 'text-ink-muted' },
  { bg: 'bg-surface', text: 'text-ink-muted' },
  { bg: 'bg-muted', text: 'text-ink-muted' },
];

// 添加一个新的函数来获取总体统计信息
const getTotalSummary = (allPages: any[]) => {
  const summary = {
    parts: new Map<string, { length: number, width: number, quantity: number, client: string }>(),
    reusable: new Map<string, { length: number, width: number, quantity: number, client: string }>()
  };

  allPages.forEach(page => {
    page.cutted.forEach((piece: any) => {
      const id = piece.id;
      const map = !piece.is_stock ? summary.parts : summary.reusable;
      
      if (!map.has(id)) {
        map.set(id, { length: piece.length, width: piece.width, quantity: 1, client: '' });
      } else {
        const existing = map.get(id)!;
        existing.quantity += 1;
      }
    });
  });

  return {
    parts: Array.from(summary.parts.entries()).map(([id, data]) => ({ id, ...data })),
    reusable: Array.from(summary.reusable.entries()).map(([id, data]) => ({ id, ...data }))
  };
};

export default function LayoutStatsPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;
  const [projectName, setProjectName] = useState('');
  const [cutted, setCutted] = useState<any[]>([]);
  const [clients, setClients] = useState<{ [key: string]: string }>({});

  // 获取零件和常用尺寸统计
  const getPartsSummary = (parts: any[]) => {
    const partsMap = new Map<string, number>();
    const reusableMap = new Map<string, number>();
    
    parts.forEach((piece) => {
      const size = `${piece.id}: ${piece.length}x${piece.width}`;
      if (!piece.is_stock) {
        // 零件
        partsMap.set(size, (partsMap.get(size) || 0) + 1);
      } else {
        // 常用尺寸
        reusableMap.set(size, (reusableMap.get(size) || 0) + 1);
      }
    });
    
    return {
      parts: Array.from(partsMap.entries()),
      reusable: Array.from(reusableMap.entries())
    };
  };

  // 添加函数来比较两页的信息是否相同
  const arePagesEqual = (page1: any, page2: any) => {
    if (!page1 || !page2) return false;
    
    // 比较板材尺寸
    if (page1.plate[0] !== page2.plate[0] || page1.plate[1] !== page2.plate[1]) {
      return false;
    }

    // 比较切割方案
    if (page1.cutted.length !== page2.cutted.length) {
      return false;
    }

    // 深度比较切割数据
    return JSON.stringify(page1.cutted.sort()) === JSON.stringify(page2.cutted.sort());
  };

  // 获取页面的颜色索引
  const getPageColorIndex = (index: number) => {
    if (index === 0) return 0;
    
    // 如果与上一页相同，使用相同的颜色索引
    if (arePagesEqual(cutted[index], cutted[index - 1])) {
      return getPageColorIndex(index - 1);
    }
    
    // 如果不同，使用新的颜色
    const prevColors = new Set();
    for (let i = 0; i < index; i++) {
      prevColors.add(getPageColorIndex(i));
    }
    
    // 找到未使用的最小颜色索引
    for (let i = 0; i < COLOR_POOL.length; i++) {
      if (!prevColors.has(i)) {
        return i;
      }
    }
    
    // 如果所有颜色都用完了，循环使用
    return prevColors.size % COLOR_POOL.length;
  };

  useEffect(() => {
    const fetchData = async () => {
      const { data } = await supabase
        .from('Projects')
        .select('name, cutted')
        .eq('id', projectId)
        .single();
      
      if (data) {
        setProjectName(data.name);
        // 获取最后一个元素（包含元数据）
        const metadata = data.cutted[data.cutted.length - 1]?.metadata || {};
        // 移除最后一个元素（元数据），只保留切板方案
        const cuttingPlans = data.cutted.slice(0, -1);
        setCutted(cuttingPlans);
        
        // 构建客户信息映射
        const clientMap: { [key: string]: string } = {};
        (metadata.others || []).forEach((item: any) => {
          clientMap[item.id] = item.client || '';
        });
        setClients(clientMap);
      }
    };
    
    if (projectId) fetchData();
  }, [projectId]);

  return (
    <div className="page-gallery flex min-h-screen flex-col">
      <div className="page-gallery-inner flex max-h-[92vh] min-h-0 flex-1 flex-col border border-hairline bg-surface" style={{ borderRadius: 2 }}>
      {/* 导航 */}
      <div className="flex items-center px-6 pt-8">
        <div className="mb-6 flex gap-2">
          <button 
            type="button"
            className="btn-gallery-secondary"
            onClick={() => router.push(`/project/${projectId}`)}
          >
            项目
          </button>
          <span className="btn-gallery-secondary-active">切板统计</span>
        </div>
      </div>

      {/* 项目名称 */}
      <div className="border-b border-hairline px-6 pb-4 animate-fade-in-up">
        <h1 className="text-2xl font-semibold tracking-tight text-ink flex items-center gap-3">
          <svg className="w-6 h-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" /></svg>
          {projectName || '未命名项目'}
        </h1>
      </div>

      {/* 切板统计 */}
      <div className="min-h-0 flex-1 overflow-auto p-6 md:p-8">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {cutted.map((item, index) => {
            const colorIndex = getPageColorIndex(index);
            const { bg, text } = COLOR_POOL[colorIndex];
            
            return (
              <div 
                key={index}
                className={`border-hairline border p-5 shadow-sm hover-lift transition-all animate-fade-in-up ${bg}`}
                style={{ borderRadius: 6, animationDelay: `${index * 0.05}s` }}
              >
                <div className="mb-3 flex items-start justify-between gap-3">
                  <h3 className="text-sm font-medium text-ink">第 {index + 1} 页</h3>
                  <button
                    type="button"
                    className={`btn-gallery-link text-sm ${text}`}
                    onClick={() => router.push(`/layout/${projectId}/${index + 1}`)}
                  >
                    查看详情 →
                  </button>
                </div>
                <div className={`text-sm leading-relaxed ${text}`}>
                  <p>板材尺寸: {item.plate[0]} × {item.plate[1]}</p>
                  <p>已切件数: {item.cutted.length}</p>
                  <p>使用率: {(item.rate * 100).toFixed(1)}%</p>
                  <div className="mt-3 border-t border-hairline pt-3">
                    {/* 零件统计 */}
                    <div className="mb-2">
                      <p className="text-ink font-medium">零件:</p>
                      {getPartsSummary(item.cutted).parts.map(([size, count], i) => (
                        <p key={i} className={`text-xs ${text}`}>
                          {size}x{count}
                        </p>
                      ))}
                    </div>
                    {/* 常用尺寸统计 */}
                    <div>
                      <p className="text-ink font-medium">常用尺寸:</p>
                      {getPartsSummary(item.cutted).reusable.map(([size, count], i) => (
                        <p key={i} className={`text-xs ${text}`}>
                          {size}x{count}
                        </p>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {cutted.length === 0 && (
          <div className="mt-12 text-center text-ink-muted">
            暂无切板数据
          </div>
        )}

        {/* 修改总体统计部分 */}
        {cutted.length > 0 && (
          <div className="mt-14 border-t border-hairline pt-10">
            <h2 className="mb-8 text-xl font-semibold text-ink flex items-center gap-2">
              <svg className="w-5 h-5 text-accent-green" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
              总体统计
            </h2>
            
            {/* 基本统计信息 */}
            <div className="mb-10 grid grid-cols-1 gap-4 text-center sm:grid-cols-3">
              <div className="border-hairline border bg-surface p-6 shadow-sm hover-lift transition-all" style={{ borderRadius: 6 }}>
                <div className="text-sm font-medium text-ink-muted flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 text-[#0284c7]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                  总页数
                </div>
                <div className="mt-3 text-3xl font-bold text-ink">{cutted.length}</div>
              </div>
              <div className="border-hairline border bg-surface p-6 shadow-sm hover-lift transition-all" style={{ borderRadius: 6 }}>
                <div className="text-sm font-medium text-ink-muted flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 text-[#9333ea]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
                  总切件数
                </div>
                <div className="mt-3 text-3xl font-bold text-ink">
                  {cutted.reduce((sum, item) => sum + item.cutted.length, 0)}
                </div>
              </div>
              <div className="border-hairline border bg-surface p-6 shadow-sm hover-lift transition-all" style={{ borderRadius: 6 }}>
                <div className="text-sm font-medium text-ink-muted flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 text-accent-green" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" /></svg>
                  平均使用率
                </div>
                <div className="mt-3 text-3xl font-bold text-ink">
                  {(cutted.reduce((sum, item) => sum + item.rate, 0) / cutted.length * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* 详细统计表格 */}
            <div className="space-y-6">
              {/* 零件统计表格 */}
              <div className="table-container hover-lift shadow-sm">
                <div className="table-title flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#9333ea]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
                  零件统计
                </div>
                <div className="table-content">
                  <table className="min-w-full">
                    <thead>
                      <tr>
                        <th className="border p-2">编号</th>
                        <th className="border p-2">长度</th>
                        <th className="border p-2">宽度</th>
                        <th className="border p-2">数量</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getTotalSummary(cutted).parts.map((item) => (
                        <tr key={item.id}>
                          <td className="border p-2 text-center">{item.id}</td>
                          <td className="border p-2 text-center">{item.length}</td>
                          <td className="border p-2 text-center">{item.width}</td>
                          <td className="border p-2 text-center">{item.quantity}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 常用尺寸统计表格 */}
              <div className="table-container hover-lift shadow-sm">
                <div className="table-title flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#d97706]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>
                  常用尺寸统计
                </div>
                <div className="table-content">
                  <table className="min-w-full">
                    <thead>
                      <tr>
                        <th className="border p-2">编号</th>
                        <th className="border p-2">长度</th>
                        <th className="border p-2">宽度</th>
                        <th className="border p-2">数量</th>
                        <th className="border p-2">客户</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getTotalSummary(cutted).reusable.map((item) => (
                        <tr key={item.id}>
                          <td className="border p-2 text-center">{item.id}</td>
                          <td className="border p-2 text-center">{item.length}</td>
                          <td className="border p-2 text-center">{item.width}</td>
                          <td className="border p-2 text-center">{item.quantity}</td>
                          <td className="border p-2 text-center">{clients[item.id] || '未命名客户'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
