'use client'

import { useEffect, useState } from 'react'
import { supabase } from '@/utils/supabaseClient'
import { useRouter } from 'next/navigation';
import { IconNavConfig, IconNavHomeLayout, IconNavOverview } from '@/components/ProjectLayoutNavPills';
import { useAppDialog } from '@/components/AppDialog';

interface Project {
  id: number
  name: string
  details: string
  description: string
  updated_at: string
  plates?: { description: string | null; length: number; width: number; quantity: number }[]
  orders?: { description: string | null; length: number; width: number; quantity: number }[]
  others?: { description: string | null; length: number; width: number; client: string | null }[]
}

interface Item {
  id: number
  description: string | null
  length: number
  width: number
  quantity: number
  customer?: string | null
}

/** cutted 最后一项为元数据（若存在），前面为各页排版方案 */
function cuttingPlanPages(cutted: unknown): unknown[] {
  if (!Array.isArray(cutted) || cutted.length === 0) return [];
  const last = cutted[cutted.length - 1];
  const hasMetadata = last != null && typeof last === 'object' && 'metadata' in (last as object);
  return hasMetadata ? cutted.slice(0, -1) : cutted;
}

/** Supabase PostgrestError 等通常不是 Error 实例，直接 instanceof 会得到「未知错误」 */
function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'object' && err !== null) {
    const o = err as { message?: unknown; details?: unknown; hint?: unknown; code?: unknown };
    if (typeof o.message === 'string' && o.message.length > 0) return o.message;
    const parts = [o.details, o.hint, o.code].filter((x) => typeof x === 'string' && (x as string).length > 0) as string[];
    if (parts.length > 0) return parts.join(' · ');
  }
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

export default function ProjectPage() {
  const { alert: dialogAlert, confirm: dialogConfirm } = useAppDialog();
  const [projects, setProjects] = useState<Project[]>([])
  // 使用 expandedProjects 管理哪些行展开
  const [expandedProjects, setExpandedProjects] = useState<Set<number>>(new Set())
  const [isCreating, setIsCreating] = useState(false);
  const [isDeleting, setIsDeleting] = useState<number | null>(null);
  const [userEmail, setUserEmail] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'dateDesc' | 'dateAsc' | 'nameAsc' | 'nameDesc'>('dateDesc');
  const [selectedProjects, setSelectedProjects] = useState<Set<number>>(new Set());

  const router = useRouter();

  useEffect(() => {
    fetchProjects()
  }, [])

  const fetchProjects = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      setProjects([]);
      return;
    }
    
    setUserEmail(user.email || '未知用户');

    const { data: bridgeData, error: bridgeError } = await supabase
      .from('Bridges')
      .select('uid, project_ids')
      .eq('uid', user.id)
      .maybeSingle();

    if (bridgeError || !bridgeData || !bridgeData.project_ids || bridgeData.project_ids.length === 0) {
      setProjects([]);
      return;
    }

    const projectIds = bridgeData.project_ids;

    const { data: projectsData, error: projectsError } = await supabase
      .from('Projects')
      .select('id, name, details, description, updated_at, plates, orders, others')
      .in('id', projectIds);

    if (projectsError || !projectsData) {
      setProjects([]);
      return;
    }

    const sortedProjects = projectIds
      .map((id: number) => projectsData.find((p: any) => p.id === id))
      .filter(Boolean);

    setProjects(sortedProjects);
  };

  const toggleExpand = (projectId: number) => {
    const newExpanded = new Set(expandedProjects);
    if (newExpanded.has(projectId)) {
      newExpanded.delete(projectId);
    } else {
      newExpanded.add(projectId);
    }
    setExpandedProjects(newExpanded);
  };

  const handleDeleteProject = async (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation(); // 阻止展开折叠
    
    if (!(await dialogConfirm('确定要删除此项目吗？', '删除确认'))) return;

    setIsDeleting(projectId);

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        await dialogAlert('请先登录', '提示');
        return;
      }

      const { data: bridgeData } = await supabase
        .from('Bridges')
        .select('project_ids')
        .eq('uid', user.id)
        .single();

      if (bridgeData) {
        const newIds = bridgeData.project_ids.filter((id: number) => id !== projectId);
        await supabase
          .from('Bridges')
          .update({ project_ids: newIds, updated_at: new Date().toISOString() })
          .eq('uid', user.id);
      }

      await supabase
        .from('Projects')
        .delete()
        .eq('id', projectId)
        .eq('uid', user.id);

      setExpandedProjects(prev => {
        const next = new Set(prev);
        next.delete(projectId);
        return next;
      });
      
      await fetchProjects();
    } catch (error) {
      await dialogAlert(errorMessage(error), '删除失败');
    } finally {
      setIsDeleting(null);
    }
  };

  const handleBatchDelete = async () => {
    if (selectedProjects.size === 0) return;
    if (!(await dialogConfirm(`确定要删除选中的 ${selectedProjects.size} 个项目吗？`, '批量删除'))) return;

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        await dialogAlert('请先登录', '提示');
        return;
      }

      const { data: bridgeData } = await supabase
        .from('Bridges')
        .select('project_ids')
        .eq('uid', user.id)
        .single();

      if (bridgeData) {
        const newIds = bridgeData.project_ids.filter((id: number) => !selectedProjects.has(id));
        await supabase
          .from('Bridges')
          .update({ project_ids: newIds, updated_at: new Date().toISOString() })
          .eq('uid', user.id);
      }

      const idsToDelete = Array.from(selectedProjects);
      await supabase
        .from('Projects')
        .delete()
        .in('id', idsToDelete)
        .eq('uid', user.id);

      setExpandedProjects(prev => {
        const next = new Set(prev);
        for (const id of idsToDelete) {
          next.delete(id);
        }
        return next;
      });
      setSelectedProjects(new Set());
      await fetchProjects();
    } catch (error) {
      await dialogAlert(errorMessage(error), '批量删除失败');
    }
  };

  const toggleSelect = (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const next = new Set(selectedProjects);
    if (next.has(projectId)) {
      next.delete(projectId);
    } else {
      next.add(projectId);
    }
    setSelectedProjects(next);
  };

  const filteredAndSortedProjects = projects.filter(p => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return p.name.toLowerCase().includes(q) || 
           (p.description && p.description.toLowerCase().includes(q)) ||
           (p.details && p.details.toLowerCase().includes(q));
  }).sort((a, b) => {
    switch (sortBy) {
      case 'dateDesc': return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      case 'dateAsc': return new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime();
      case 'nameAsc': return a.name.localeCompare(b.name, 'zh-CN');
      case 'nameDesc': return b.name.localeCompare(a.name, 'zh-CN');
      default: return 0;
    }
  });

  const toggleSelectAll = () => {
    if (selectedProjects.size === filteredAndSortedProjects.length && filteredAndSortedProjects.length > 0) {
      setSelectedProjects(new Set());
    } else {
      setSelectedProjects(new Set(filteredAndSortedProjects.map(p => p.id)));
    }
  };

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr);
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const min = String(d.getMinutes()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
  };

  const sectionToneClass = (tone: 'plate' | 'part' | 'stock') => {
    if (tone === 'plate') return '!bg-sky-50 border-l-[3px] !border-l-sky-500'
    if (tone === 'part') return '!bg-violet-50 border-l-[3px] !border-l-violet-500'
    return '!bg-amber-50 border-l-[3px] !border-l-amber-500'
  }

  const sectionTitleBarClass = (tone: 'plate' | 'part' | 'stock') => {
    if (tone === 'plate') return 'bg-sky-100/70'
    if (tone === 'part') return 'bg-violet-100/70'
    return 'bg-amber-100/70'
  }

  const renderTable = (
    items: Item[],
    title: string,
    showQuantity: boolean = true,
    showCustomer: boolean = false,
    colorClass: string = "text-ink",
    sectionTone?: 'plate' | 'part' | 'stock'
  ) => {
    const tone = sectionTone ? sectionToneClass(sectionTone) : ''
    const titleBar = sectionTone ? sectionTitleBarClass(sectionTone) : ''

    if (!items || items.length === 0) {
      return (
        <div className={`table-container shadow-sm ${tone}`}>
          <div className={`table-title ${colorClass} flex items-center gap-2 ${titleBar}`}>{title}</div>
          <div className={`p-4 text-center text-xs text-ink-muted ${sectionTone === 'plate' ? 'bg-sky-50/50' : sectionTone === 'part' ? 'bg-violet-50/50' : sectionTone === 'stock' ? 'bg-amber-50/50' : 'bg-surface'}`}>暂无数据</div>
        </div>
      );
    }

    return (
      <div className={`table-container shadow-sm ${tone}`}>
        <div className={`table-title ${colorClass} flex items-center gap-2 ${titleBar}`}>
          {title}
        </div>
        <div className="table-content">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="border p-2 w-12 text-center">#</th>
                <th className="border p-2">长 × 宽</th>
                {showQuantity && <th className="border p-2 w-16 text-center">数量</th>}
                {showCustomer && <th className="border p-2 truncate max-w-[80px]">客户</th>}
                <th className="border p-2 truncate max-w-[100px]">描述</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => (
                <tr key={item.id || index}>
                  <td className="border p-2 text-center text-ink-muted">{index + 1}</td>
                  <td className="border p-2 text-ink font-mono tracking-tight">{item.length} × {item.width}</td>
                  {showQuantity && <td className="border p-2 text-center">{item.quantity}</td>}
                  {showCustomer && <td className="border p-2 truncate max-w-[80px]">{item.customer || '-'}</td>}
                  <td className="border p-2 truncate max-w-[100px] text-ink-muted">{item.description || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const handleEdit = (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/project/${projectId}`);
  };

  const handleHomeLayout = async (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const { data } = await supabase.from('Projects').select('cutted').eq('id', projectId).single();
    if (cuttingPlanPages(data?.cutted).length > 0) {
      router.push(`/layout/${projectId}/1`);
    } else {
      await dialogAlert('暂无切板方案，请先在项目中执行切板', '提示');
    }
  };

  const handleSchemeOverview = (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/layout/${projectId}`);
  };

  const handleProjectNameNavigate = async (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const { data } = await supabase.from('Projects').select('cutted').eq('id', projectId).single();
    if (cuttingPlanPages(data?.cutted).length > 0) {
      router.push(`/layout/${projectId}/1`);
    } else {
      router.push(`/project/${projectId}`);
    }
  };

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (!error) {
      router.push('/login');
    } else {
      await dialogAlert('退出登录失败: ' + error.message, '退出失败');
    }
  };

  const handleNew = async () => {
    if (isCreating) return;
    setIsCreating(true);

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        await dialogAlert('请先登录', '提示');
        setIsCreating(false);
        return;
      }

      const { data: projectsData } = await supabase.from('Projects').select('name');
      const existingNames = new Set(projectsData?.map(p => p.name) || []);
      
      let i = 1;
      let newName = `new_${i}`;
      while (existingNames.has(newName)) {
        i++;
        newName = `new_${i}`;
      }

      const defaultPlates = [{ id: 1, width: 1220, length: 2440, quantity: 100, description: 'default' }];
      const now = new Date().toISOString();

      const { data: newProject, error: projectError } = await supabase
        .from('Projects')
        .insert([
          {
            name: newName,
            uid: user.id,
            details: '',
            description: '',
            saw_blade: 4,
            plates: defaultPlates,
            orders: [],
            others: [],
            updated_at: now,
          },
        ])
        .select()
        .single();

      if (projectError) throw projectError;
      if (!newProject?.id) {
        throw new Error('已插入但未返回项目 id，请检查 Projects 表的 select 策略与 RLS');
      }

      const { data: bridgeData, error: bridgeError } = await supabase
        .from('Bridges')
        .select('uid, project_ids')
        .eq('uid', user.id)
        .maybeSingle();

      if (bridgeError) throw bridgeError;

      const existingProjectIds = bridgeData?.project_ids || [];
      const newProjectIds = [...existingProjectIds, newProject.id];

      if (!bridgeData) {
        const { error: insertBridgeError } = await supabase
          .from('Bridges')
          .insert({ uid: user.id, project_ids: newProjectIds, updated_at: now });
        if (insertBridgeError) throw insertBridgeError;
      } else {
        const { error: updateBridgeError } = await supabase
          .from('Bridges')
          .update({ project_ids: newProjectIds, updated_at: now })
          .eq('uid', user.id);
        if (updateBridgeError) throw updateBridgeError;
      }

      await fetchProjects();
      router.push(`/project/${newProject.id}`);
    } catch (error) {
      console.error('创建项目失败', error);
      await dialogAlert('创建项目失败: ' + errorMessage(error), '创建失败');
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="page-gallery">
      <div className="page-gallery-inner">
      {/* 顶部导航区 */}
      <div className="relative z-30 mb-8 flex min-w-0 flex-wrap items-center justify-between gap-4 border-b border-hairline pb-4 animate-fade-in-up">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <svg className="w-6 h-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
            </svg>
            <h1 className="text-2xl font-semibold tracking-tight text-ink">
              Plate Cutting
            </h1>
          </div>
        </div>

        <div className="flex min-w-0 flex-wrap items-center justify-end gap-3 sm:gap-4">
          <button 
            type="button"
            className="btn-gallery-primary inline-flex h-8 items-center gap-1.5 px-3 text-xs shadow-sm"
            onClick={handleNew}
            disabled={isCreating}
          >
            {isCreating ? (
              <svg className="h-4 w-4 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            )}
            新建项目
          </button>

          <div className="h-4 w-px shrink-0 bg-border-hairline" aria-hidden />

          <div
            className="has-tooltip has-tooltip-user flex cursor-default items-center gap-0 text-ink-muted min-w-0"
            aria-label={userEmail ? `当前用户 ${userEmail}，悬停查看详情` : '当前用户，悬停查看详情'}
          >
            <svg className="h-5 w-5 shrink-0 rounded-full bg-muted p-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <div className="tooltip-text flex min-w-[12rem] max-w-[min(90vw,18rem)] flex-col items-stretch gap-2">
              <div className="text-[10px] font-medium uppercase tracking-wider text-ink-muted">当前用户</div>
              <div className="break-all text-xs leading-snug text-ink">{userEmail || '未登录'}</div>
            </div>
          </div>

          <button onClick={handleLogout} className="has-tooltip icon-btn icon-btn-red ml-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
            <span className="tooltip-text">退出登录</span>
          </button>
        </div>
      </div>

      {/* 工具栏：左筛选 / 右统计+排序，避免挤在一侧 */}
      <div
        className="relative z-10 mb-4 flex w-full min-w-0 flex-col gap-3 animate-fade-in-up sm:flex-row sm:items-center sm:justify-between sm:gap-4"
        style={{ animationDelay: '0.05s' }}
      >
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <button 
            type="button"
            className="btn-gallery-ghost !p-1 !h-8 w-8 flex shrink-0 items-center justify-center border-hairline"
            onClick={toggleSelectAll}
            title={selectedProjects.size === filteredAndSortedProjects.length && filteredAndSortedProjects.length > 0 ? "取消全选" : "全选"}
          >
            <input 
              type="checkbox" 
              checked={selectedProjects.size === filteredAndSortedProjects.length && filteredAndSortedProjects.length > 0} 
              readOnly 
              className="h-4 w-4 cursor-pointer accent-[#0284c7]"
            />
          </button>

          <div className="relative min-w-0 flex-1 sm:max-w-xs sm:flex-none">
            <svg className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
            <input 
              type="text" 
              placeholder="搜索项目…" 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="field-gallery h-8 w-full !py-0 !pl-8 !pr-3 text-xs leading-8 placeholder:text-[12px] placeholder:text-ink-muted/70 focus:border-accent focus:ring-1 focus:ring-accent"
            />
          </div>

          {selectedProjects.size > 0 && (
            <button 
              type="button"
              onClick={handleBatchDelete}
              className="btn-gallery-danger flex h-8 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md border border-[#fecaca] bg-[#fef2f2] px-2.5 text-xs text-[#dc2626] transition-colors hover:bg-[#fee2e2]"
            >
              <svg className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
              批量删除 ({selectedProjects.size})
            </button>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2 border-t border-hairline pt-3 sm:border-t-0 sm:pt-0">
          <span
            className="badge badge-accent inline-flex shrink-0 items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-semibold tabular-nums"
            title={searchQuery.trim() ? `列表显示 ${filteredAndSortedProjects.length} 项，账号共 ${projects.length} 个项目` : '账号下项目总数'}
          >
            共 {projects.length} 个
          </span>
          <select 
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            aria-label="排序方式"
            className="field-gallery h-8 min-w-[9.5rem] cursor-pointer !py-0 pl-2.5 pr-8 text-xs leading-8 text-ink hover:border-[#c5c5c7]"
          >
            <option value="dateDesc">最新修改</option>
            <option value="dateAsc">最早修改</option>
            <option value="nameAsc">名称 (A-Z)</option>
            <option value="nameDesc">名称 (Z-A)</option>
          </select>
        </div>
      </div>

      {/* 项目列表区：一栏一栏的行列表 */}
      <div className="space-y-3 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
        {filteredAndSortedProjects.length === 0 ? (
          <div className="text-center py-12 text-ink-muted border border-dashed border-hairline rounded-md">
            暂无项目，点击右上角新建。
          </div>
        ) : (
          filteredAndSortedProjects.map((project) => {
            const isExpanded = expandedProjects.has(project.id);
            const pParts = Array.isArray(project.plates) ? project.plates.map((item, idx) => ({ ...item, id: idx + 1 })) : [];
            const pOrders = Array.isArray(project.orders) ? project.orders.map((item, idx) => ({ ...item, id: idx + 1 })) : [];
            const pOthers = Array.isArray(project.others) ? project.others.map((item, idx) => ({ ...item, id: idx + 1, customer: item.client, quantity: 0 })) : [];

            return (
              <div key={project.id} className="border border-hairline bg-surface rounded shadow-sm hover-lift transition-all">
                {/* 行头区（点击展开收起） */}
                <div 
                  className="group flex cursor-pointer select-none items-center justify-between p-4"
                  onClick={() => toggleExpand(project.id)}
                >
                  <div className="flex flex-1 items-center gap-4 overflow-hidden">
                    <div 
                      className="flex h-8 w-8 shrink-0 items-center justify-center hover:bg-muted rounded transition-colors"
                      onClick={(e) => toggleSelect(project.id, e)}
                    >
                      <input 
                        type="checkbox" 
                        checked={selectedProjects.has(project.id)}
                        readOnly
                        className="w-4 h-4 cursor-pointer accent-[#0284c7]"
                      />
                    </div>
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <button
                        type="button"
                        className="max-w-[200px] cursor-pointer truncate border-0 bg-transparent p-0 text-left text-sm font-medium text-accent hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1 rounded-sm"
                        title="有切板数据时进入首页排版，否则进入项目配置"
                        onClick={(e) => void handleProjectNameNavigate(project.id, e)}
                      >
                        {project.name}
                      </button>
                      {project.details && <span className="badge badge-gray truncate">{project.details}</span>}
                      {project.description && (
                        <span className="truncate text-xs text-ink-muted max-w-[300px]">
                          {project.description}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  {/* 日期及操作区 */}
                  <div className="ml-4 flex shrink-0 items-center gap-4">
                    <div className="text-xs text-ink-muted hidden sm:block has-tooltip">
                      {formatDate(project.updated_at)}
                      <span className="tooltip-text">{new Date(project.updated_at).toLocaleString('zh-CN')}</span>
                    </div>

                    <div className="flex items-center gap-1 opacity-80 transition-opacity group-hover:opacity-100">
                      <button 
                        type="button"
                        onClick={(e) => handleEdit(project.id, e)}
                        className="has-tooltip icon-btn icon-btn-blue"
                        aria-label="项目配置"
                      >
                        <IconNavConfig className="h-4 w-4" />
                        <span className="tooltip-text">项目配置</span>
                      </button>

                      <button 
                        type="button"
                        onClick={(e) => void handleHomeLayout(project.id, e)}
                        className="has-tooltip icon-btn icon-btn-teal"
                        aria-label="首页排版"
                      >
                        <IconNavHomeLayout className="h-4 w-4" />
                        <span className="tooltip-text">首页排版</span>
                      </button>

                      <button 
                        type="button"
                        onClick={(e) => handleSchemeOverview(project.id, e)}
                        className="has-tooltip icon-btn icon-btn-violet"
                        aria-label="方案总览"
                      >
                        <IconNavOverview className="h-4 w-4" />
                        <span className="tooltip-text">方案总览</span>
                      </button>

                    <button 
                      type="button"
                      onClick={(e) => handleDeleteProject(project.id, e)}
                      disabled={isDeleting === project.id}
                      className="has-tooltip icon-btn icon-btn-red"
                    >
                      {isDeleting === project.id ? (
                        <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                      ) : (
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                      )}
                      <span className="tooltip-text">删除项目</span>
                    </button>

                    <div className="mx-1 h-4 w-[1px] bg-border-hairline"></div>

                    {/* 折叠箭头 */}
                    <div className="flex h-6 w-6 items-center justify-center text-ink-muted">
                      <svg 
                        className={`h-4 w-4 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} 
                        fill="none" viewBox="0 0 24 24" stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                  </div>
                </div>

                {/* 展开的详情区 */}
                {isExpanded && (
                  <div className="animate-fade-in-up border-t border-hairline bg-muted/30 p-4" style={{ animationDuration: '0.2s' }}>
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                      {renderTable(pParts, '板件信息', true, false, 'text-[#0284c7]', 'plate')}
                      {renderTable(pOrders, '零件信息', true, false, 'text-[#9333ea]', 'part')}
                      {renderTable(pOthers, '常用尺寸', false, true, 'text-[#d97706]', 'stock')}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
      </div>
    </div>
  )
}
