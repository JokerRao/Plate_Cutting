'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { supabase } from '@/utils/supabaseClient';

// 添加颜色池常量
const COLOR_POOL = [
  { bg: 'bg-blue-50', text: 'text-blue-600' },
  { bg: 'bg-green-50', text: 'text-green-600' },
  { bg: 'bg-purple-50', text: 'text-purple-600' },
  { bg: 'bg-pink-50', text: 'text-pink-600' },
  { bg: 'bg-orange-50', text: 'text-orange-600' },
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
    <div className="relative max-w-7xl mx-auto my-8 rounded-2xl shadow-2xl border bg-white flex flex-col h-[92vh]">
      {/* 导航按钮 */}
      <div className="flex items-center px-6 pt-4">
        <div className="flex gap-2 mb-4">
          <button 
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold"
            onClick={() => router.push(`/project/${projectId}`)}
          >
            项目
          </button>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold">
            切板统计
          </button>
        </div>
      </div>

      {/* 项目名称 */}
      <div className="px-6 pb-2 border-b">
        <h1 className="text-xl font-bold">{projectName || '未命名项目'}</h1>
      </div>

      {/* 切板统计 */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="grid grid-cols-3 gap-4">
          {cutted.map((item, index) => {
            const colorIndex = getPageColorIndex(index);
            const { bg, text } = COLOR_POOL[colorIndex];
            
            return (
              <div 
                key={index}
                className={`border rounded-lg p-4 hover:brightness-95 ${bg}`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-semibold">第 {index + 1} 页</h3>
                  <button
                    className={`${text} hover:brightness-90 text-sm`}
                    onClick={() => router.push(`/layout/${projectId}/${index + 1}`)}
                  >
                    查看详情 →
                  </button>
                </div>
                <div className="text-sm text-gray-600">
                  <p>板材尺寸: {item.plate[0]} × {item.plate[1]}</p>
                  <p>已切件数: {item.cutted.length}</p>
                  <p>使用率: {(item.rate * 100).toFixed(1)}%</p>
                  <div className="mt-2 pt-2 border-t">
                    {/* 零件统计 */}
                    <div className="mb-2">
                      <p className="font-medium text-gray-700">零件:</p>
                      {getPartsSummary(item.cutted).parts.map(([size, count], i) => (
                        <p key={i} className={`text-xs ${text}`}>
                          {size}x{count}
                        </p>
                      ))}
                    </div>
                    {/* 常用尺寸统计 */}
                    <div>
                      <p className="font-medium text-gray-700">常用尺寸:</p>
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
          <div className="text-center text-gray-500 mt-8">
            暂无切板数据
          </div>
        )}

        {/* 修改总体统计部分 */}
        {cutted.length > 0 && (
          <div className="mt-8 border-t pt-4">
            <h2 className="text-lg font-semibold mb-4">总体统计</h2>
            
            {/* 基本统计信息 */}
            <div className="grid grid-cols-3 gap-4 text-center mb-6">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-gray-600">总页数</div>
                <div className="text-2xl font-bold">{cutted.length}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-gray-600">总切件数</div>
                <div className="text-2xl font-bold">
                  {cutted.reduce((sum, item) => sum + item.cutted.length, 0)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-gray-600">平均使用率</div>
                <div className="text-2xl font-bold">
                  {(cutted.reduce((sum, item) => sum + item.rate, 0) / cutted.length * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* 详细统计表格 */}
            <div className="space-y-6">
              {/* 零件统计表格 */}
              <div className="table-container">
                <div className="table-title">零件统计</div>
                <div className="table-content">
                  <table className="min-w-full">
                    <thead>
                      <tr className="bg-blue-50">
                        <th className="border p-2">编号</th>
                        <th className="border p-2">长度</th>
                        <th className="border p-2">宽度</th>
                        <th className="border p-2">数量</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getTotalSummary(cutted).parts.map((item) => (
                        <tr key={item.id} className="hover:bg-gray-50">
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
              <div className="table-container">
                <div className="table-title">常用尺寸统计</div>
                <div className="table-content">
                  <table className="min-w-full">
                    <thead>
                      <tr className="bg-yellow-50">
                        <th className="border p-2">编号</th>
                        <th className="border p-2">长度</th>
                        <th className="border p-2">宽度</th>
                        <th className="border p-2">数量</th>
                        <th className="border p-2">客户</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getTotalSummary(cutted).reusable.map((item) => (
                        <tr key={item.id} className="hover:bg-gray-50">
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
  );
}
