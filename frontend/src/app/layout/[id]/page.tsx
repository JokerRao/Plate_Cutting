'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useMemo } from 'react';
import { supabase } from '@/utils/supabaseClient';
import ProjectLayoutNavPills, { ProjectListNavButton } from '@/components/ProjectLayoutNavPills';
import { groupCutPlansInOrder, getGroupAccent, formatIndices } from '@/utils/cutPlanGroup';

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

  const cutGroups = useMemo(() => groupCutPlansInOrder(cutted), [cutted]);

  useEffect(() => {
    const fetchData = async () => {
      const { data } = await supabase
        .from('Projects')
        .select('name, cutted')
        .eq('id', projectId)
        .single();
      
      if (data) {
        setProjectName(data.name || '');
        if (data.cutted && data.cutted.length > 0) {
          const lastItem = data.cutted[data.cutted.length - 1];
          const hasMetadata = lastItem?.metadata != null;
          const metadata = hasMetadata ? (lastItem.metadata || {}) : {};
          const cuttingPlans = hasMetadata ? data.cutted.slice(0, -1) : data.cutted;
          setCutted(cuttingPlans);
          const clientMap: { [key: string]: string } = {};
          (metadata.others || []).forEach((item: any) => {
            clientMap[item.id] = item.client || '';
          });
          setClients(clientMap);
        } else {
          setCutted([]);
          setClients({});
        }
      }
    };
    
    if (projectId) fetchData();
  }, [projectId]);

  return (
    <div className="page-gallery">
      <div className="page-gallery-inner">
      {/* ① 左标题 | 右：当前 + 项目列表 ② 左：共 N 张 | 右：四联导航 */}
      <div className="pt-3 space-y-6 border-b border-hairline pb-4 mb-6 animate-fade-in-up">
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <h1 className="flex min-w-0 flex-1 items-center gap-2 text-xl font-semibold tracking-tight text-ink sm:gap-3 sm:text-2xl">
            <svg className="h-5 w-5 shrink-0 text-accent sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
            </svg>
            <span className="truncate">{projectName || '未命名项目'}</span>
          </h1>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            <span
              className="inline-flex h-8 items-center rounded-md border border-hairline bg-muted/40 px-2.5 text-xs font-medium text-ink-muted"
              title="当前所在界面"
            >
              方案总览
              <span className="ml-1.5 rounded bg-surface/90 px-1.5 py-0.5 text-[10px] font-semibold text-ink">当前</span>
            </span>
            <ProjectListNavButton size="toolbar" />
          </div>
        </div>
        <div className="flex min-h-8 flex-wrap items-center justify-between gap-x-3 gap-y-2 mt-4">
          <div className="flex flex-wrap items-center gap-2">
            {cutted.length > 0 ? (
              <span
                className="inline-flex h-8 shrink-0 items-center whitespace-nowrap rounded-md border border-[rgba(0,122,255,0.22)] bg-[rgba(0,122,255,0.08)] px-2.5 text-xs font-medium tabular-nums text-[var(--accent)]"
                title="当前项目板材方案总张数"
              >
                共 {cutted.length} 张板材方案
              </span>
            ) : (
              <span className="inline-flex h-8 shrink-0 items-center rounded-md border border-hairline bg-muted/60 px-2.5 text-xs font-medium text-ink-muted">
                暂无方案
              </span>
            )}
          </div>
          <ProjectLayoutNavPills
            projectId={projectId}
            active="layout-list"
            className="mb-0"
            size="toolbar"
            show={{ schemeOverview: false, projectList: false }}
            suppressPillCurrentLabel
          />
        </div>
      </div>

      {/* 总体统计（在排版卡片之上） */}
      {cutted.length > 0 && (
        <div className="mb-10 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
          <h2 className="mb-6 flex items-center gap-2 text-xl font-semibold text-ink">
            <svg className="w-5 h-5 text-accent-green" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
            总体统计
          </h2>

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

          <div className="space-y-6">
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

      {/* 排版统计（各板方案卡片） */}
      <div className="animate-fade-in-up border-t border-hairline pt-8" style={{ animationDelay: '0.1s' }}>
        {cutted.length > 0 && (
          <h2 className="mb-4 text-lg font-semibold tracking-tight text-ink">排版统计</h2>
        )}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {cutGroups.map((g, gi) => {
            const item = g.representative;
            const multi = g.indices.length > 1;
            const first = g.indices[0] + 1;
            const ac = getGroupAccent(gi);
            const detailLine = multi ? `text-xs ${ac.text} opacity-90` : 'text-xs text-ink-muted';

            const cardShell = multi
              ? `border-2 ${ac.border} ${ac.bg} shadow-sm ring-2 ring-offset-1 ${ac.ring}`
              : 'border-hairline border bg-surface';

            return (
              <div
                key={g.signature}
                className={`p-5 transition-all animate-fade-in-up hover-lift ${cardShell}`}
                style={{ borderRadius: 6, animationDelay: `${gi * 0.05}s` }}
              >
                <div className="mb-3 flex items-start justify-between gap-3">
                  <h3 className={`text-sm font-semibold ${multi ? ac.text : 'text-ink'}`}>
                    {multi ? `第 ${formatIndices(g.indices)} 张（${g.indices.length} 张同切法）` : `第 ${first} 页`}
                  </h3>
                  <button
                    type="button"
                    className={`btn-gallery-link text-sm ${multi ? ac.text : ''}`}
                    title={multi ? `进入该同切组第一张（第 ${first} 张）` : undefined}
                    onClick={() => router.push(`/layout/${projectId}/${first}`)}
                  >
                    查看详情 →
                  </button>
                </div>
                <div className={`text-sm leading-relaxed ${multi ? ac.text : 'text-ink-muted'}`}>
                  <p>板材尺寸: {item.plate[0]} × {item.plate[1]}</p>
                  <p>已切件数: {item.cutted.length}</p>
                  <p>使用率: {(item.rate * 100).toFixed(1)}%</p>
                  <div className={`mt-3 border-t pt-3 ${multi ? 'border-black/15' : 'border-hairline'}`}>
                    <div className="mb-2">
                      <p className={`font-medium ${multi ? ac.text : 'text-ink'}`}>零件:</p>
                      {getPartsSummary(item.cutted).parts.map(([size, count], i) => (
                        <p key={i} className={detailLine}>
                          {size}x{count}
                        </p>
                      ))}
                    </div>
                    <div>
                      <p className={`font-medium ${multi ? ac.text : 'text-ink'}`}>常用尺寸:</p>
                      {getPartsSummary(item.cutted).reusable.map(([size, count], i) => (
                        <p key={i} className={detailLine}>
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
      </div>
      </div>
    </div>
  );
}
