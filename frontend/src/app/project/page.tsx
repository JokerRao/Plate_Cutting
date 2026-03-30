'use client'

import { useEffect, useState } from 'react'
import { supabase } from '@/utils/supabaseClient'
import { useRouter } from 'next/navigation';

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

export default function ProjectPage() {
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
    
    if (!confirm('确定要删除此项目吗？')) return;

    setIsDeleting(projectId);

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        alert('请先登录');
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
      alert(error instanceof Error ? error.message : '删除失败');
    } finally {
      setIsDeleting(null);
    }
  };

  const handleBatchDelete = async () => {
    if (selectedProjects.size === 0) return;
    if (!confirm(`确定要删除选中的 ${selectedProjects.size} 个项目吗？`)) return;

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        alert('请先登录');
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
      alert(error instanceof Error ? error.message : '批量删除失败');
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

  const renderTable = (
    items: Item[],
    title: string,
    showQuantity: boolean = true,
    showCustomer: boolean = false,
    colorClass: string = "text-ink"
  ) => {
    if (!items || items.length === 0) {
      return (
        <div className="table-container shadow-sm">
          <div className={`table-title ${colorClass} flex items-center gap-2`}>{title}</div>
          <div className="p-4 text-center text-xs text-ink-muted bg-surface">暂无数据</div>
        </div>
      );
    }

    return (
      <div className="table-container shadow-sm">
        <div className={`table-title ${colorClass} flex items-center gap-2`}>
          {title}
        </div>
        <div className="table-content">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="border p-2 w-12 text-center">#</th>
                <th className="border p-2">长 × 宽</th>
                {showQuantity && <th className="border p-2 w-16 text-center">数</th>}
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

  const handleLayout = (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/layout/${projectId}`);
  };

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (!error) {
      router.push('/login');
    } else {
      alert('退出登录失败: ' + error.message);
    }
  };

  const handleNew = async () => {
    if (isCreating) return;
    setIsCreating(true);

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        alert('请先登录');
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

      const defaultPlates = [{ id: 1, width: 1220, length: 2440, quantity: 100, description: "default" }];

      const { data: newProject, error: projectError } = await supabase
        .from('Projects')
        .insert([{ name: newName, uid: user.id, plates: defaultPlates, orders: [], others: [] }])
        .select()
        .single();

      if (projectError) throw projectError;

      const { data: bridgeData, error: bridgeError } = await supabase
        .from('Bridges')
        .select('*')
        .eq('uid', user.id)
        .single();

      const existingProjectIds = bridgeData?.project_ids || [];
      const newProjectIds = [...existingProjectIds, newProject.id];
      const now = new Date().toISOString();

      if (bridgeError?.code === 'PGRST116') {
        await supabase.from('Bridges').insert({ uid: user.id, project_ids: newProjectIds, updated_at: now });
      } else if (!bridgeError) {
        await supabase.from('Bridges').update({ project_ids: newProjectIds, updated_at: now }).eq('uid', user.id);
      }

      await fetchProjects();
      router.push(`/project/${newProject.id}`);
    } catch (error) {
      alert('创建项目失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="page-gallery">
      <div className="page-gallery-inner">
      {/* 顶部导航区 */}
      <div className="mb-8 border-b border-hairline pb-4 flex items-center justify-between animate-fade-in-up">
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

        <div className="flex items-center gap-4">
          <button 
            type="button"
            className="btn-gallery-primary flex items-center gap-1.5 shadow-sm px-4 py-2 text-sm"
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

          <div className="h-4 w-[1px] bg-border-hairline"></div>
          
          <div className="has-tooltip flex items-center gap-2 text-ink-muted cursor-default">
            <svg className="w-5 h-5 bg-muted rounded-full p-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <div className="tooltip-text flex flex-col items-center gap-1">
              <span>{userEmail || '未登录'}</span>
              <span className="text-xs text-white/70">共 {projects.length} 个项目</span>
            </div>
          </div>

          <button onClick={handleLogout} className="has-tooltip icon-btn icon-btn-red ml-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
            <span className="tooltip-text">退出登录</span>
          </button>
        </div>
      </div>

      {/* 工具栏 */}
      <div className="mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
        <div className="flex flex-wrap items-center gap-3 w-full sm:flex-1">
          <button 
            type="button"
            className="btn-gallery-ghost !p-1 !h-8 w-8 flex items-center justify-center border-hairline shrink-0"
            onClick={toggleSelectAll}
            title={selectedProjects.size === filteredAndSortedProjects.length && filteredAndSortedProjects.length > 0 ? "取消全选" : "全选"}
          >
            <input 
              type="checkbox" 
              checked={selectedProjects.size === filteredAndSortedProjects.length && filteredAndSortedProjects.length > 0} 
              readOnly 
              className="w-4 h-4 cursor-pointer accent-[#0284c7]"
            />
          </button>

          <div className="relative shrink-0 flex-1 sm:flex-none">
            <svg className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
            <input 
              type="text" 
              placeholder="搜索项目" 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="field-gallery !pl-9 !pr-4 !py-1.5 w-full sm:w-64 lg:w-72 text-xs placeholder:text-[12px] placeholder:text-ink-muted/70 text-left transition-all focus:ring-1 focus:ring-accent focus:border-accent"
            />
          </div>

          {selectedProjects.size > 0 && (
            <button 
              type="button"
              onClick={handleBatchDelete}
              className="btn-gallery-danger flex items-center gap-1.5 text-xs py-1.5 px-3 border border-[#fecaca] bg-[#fef2f2] hover:bg-[#fee2e2] text-[#dc2626] rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
              批量删除 ({selectedProjects.size})
            </button>
          )}
        </div>
        
        <div className="flex items-center gap-3 shrink-0 self-start sm:self-auto sm:ml-0">
          <span className="text-xs text-ink-muted shrink-0 whitespace-nowrap hidden sm:inline-block mr-2">共 {filteredAndSortedProjects.length} 项</span>
          <select 
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="field-gallery !py-1.5 w-32 text-xs text-ink-muted shrink-0 cursor-pointer hover:border-[#c5c5c7] transition-colors"
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
                      <h2 className="truncate text-sm font-medium text-ink max-w-[200px]">{project.name}</h2>
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

                    <div className="flex items-center gap-1.5 opacity-80 transition-opacity group-hover:opacity-100">
                      <button 
                        type="button"
                        onClick={(e) => handleEdit(project.id, e)}
                        className="has-tooltip icon-btn icon-btn-blue"
                      >
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
                        <span className="tooltip-text">编辑项目</span>
                      </button>
                    
                    <button 
                      type="button"
                      onClick={(e) => handleLayout(project.id, e)}
                      className="has-tooltip icon-btn icon-btn-green"
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>
                      <span className="tooltip-text">切板统计</span>
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
                      {renderTable(pParts, '板件信息', true, false, 'text-[#0284c7]')}
                      {renderTable(pOrders, '零件信息', true, false, 'text-[#9333ea]')}
                      {renderTable(pOthers, '常用尺寸', false, true, 'text-[#d97706]')}
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
