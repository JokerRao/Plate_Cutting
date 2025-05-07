'use client'

import { useEffect, useState } from 'react'
import { supabase } from '@/utils/supabaseClient'
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
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
  const [selectedProject, setSelectedProject] = useState<number | null>(null)
  const [parts, setParts] = useState<Item[]>([])
  const [components, setComponents] = useState<Item[]>([])
  const [dimensions, setDimensions] = useState<Item[]>([])
  const [sortConfig, setSortConfig] = useState<{key: keyof Project, direction: 'asc' | 'desc'} | null>(null)
  const [isEditing, setIsEditing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProjects, setSelectedProjects] = useState<Set<number>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);

  const router = useRouter();

  useEffect(() => {
    fetchProjects()
  }, [])

  const fetchProjects = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      // 用户未登录
      setProjects([]);
      return;
    }

    // 检查 Bridges 表
    const { data: bridgeData, error: bridgeError } = await supabase
      .from('Bridges')
      .select('uid, project_ids')
      .eq('uid', user.id)
      .maybeSingle();

    if (bridgeError) {
      // 查询出错
      setProjects([]);
      return;
    }

    if (!bridgeData) {
      // 没有 Bridges 记录
      setProjects([]);
      return;
    }

    // 有 Bridges 记录，继续查项目
    const projectIds = bridgeData.project_ids;
    if (!projectIds || projectIds.length === 0) {
      setProjects([]);
      return;
    }

    const { data: projectsData, error: projectsError } = await supabase
      .from('Projects')
      .select('id, name, details, description, updated_at, plates, orders, others')
      .in('id', projectIds);

    if (projectsError || !projectsData) {
      setProjects([]);
      return;
    }

    // 按 projectIds 顺序排序
    const sortedProjects = projectIds
      .map((id: number) => projectsData.find((p: any) => p.id === id))
      .filter(Boolean);

    setProjects(sortedProjects);
  };

  const handleSelectProject = (projectId: number) => {
    setSelectedProject(projectId);
    const project = projects.find(p => p.id === projectId);
    if (project) {
      setParts(Array.isArray(project.plates) ? project.plates.map((item, idx) => ({
        id: idx + 1,
        description: item.description ?? '',
        length: item.length ?? 0,
        width: item.width ?? 0,
        quantity: item.quantity ?? 0,
      })) : []);
      setComponents(Array.isArray(project.orders) ? project.orders.map((item, idx) => ({
        id: idx + 1,
        description: item.description ?? '',
        length: item.length ?? 0,
        width: item.width ?? 0,
        quantity: item.quantity ?? 0,
      })) : []);
      setDimensions(Array.isArray(project.others) ? project.others.map((item, idx) => ({
        id: idx + 1,
        description: item.description ?? '',
        length: item.length ?? 0,
        width: item.width ?? 0,
        quantity: 0,
        customer: item.client ?? '',
      })) : []);
    } else {
      setParts([]);
      setComponents([]);
      setDimensions([]);
    }
  };

  const handleCheckProject = (projectId: number, checked: boolean) => {
    const newSelected = new Set(selectedProjects);
    if (checked) {
      newSelected.add(projectId);
    } else {
      newSelected.delete(projectId);
    }
    setSelectedProjects(newSelected);
  };

  const handleDeleteProjects = async () => {
    if (selectedProjects.size === 0) {
      alert('请选择要删除的项目');
      return;
    }

    if (!confirm(`确定要删除选中的 ${selectedProjects.size} 个项目吗？`)) {
      return;
    }

    setIsDeleting(true);

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        alert('请先登录');
        return;
      }

      // 获取当前用户的 Bridges 记录
      const { data: bridgeData, error: bridgeError } = await supabase
        .from('Bridges')
        .select('project_ids')
        .eq('uid', user.id)
        .single();

      if (bridgeError) {
        throw new Error('获取项目列表失败: ' + bridgeError.message);
      }

      // 从 project_ids 中移除要删除的项目
      const projectIds = bridgeData.project_ids.filter((id: number) => !selectedProjects.has(id));

      // 更新 Bridges 表
      const { error: updateError } = await supabase
        .from('Bridges')
        .update({
          project_ids: projectIds,
          updated_at: new Date().toISOString()
        })
        .eq('uid', user.id);

      if (updateError) {
        throw new Error('更新项目列表失败: ' + updateError.message);
      }

      // 删除 Projects 表中的项目
      const { error: deleteError } = await supabase
        .from('Projects')
        .delete()
        .in('id', Array.from(selectedProjects))
        .eq('uid', user.id);

      if (deleteError) {
        throw new Error('删除项目失败: ' + deleteError.message);
      }

      // 清除选中状态
      setSelectedProjects(new Set());
      
      // 如果当前选中的项目被删除，清除选中状态
      if (selectedProject && selectedProjects.has(selectedProject)) {
        setSelectedProject(null);
        setParts([]);
        setComponents([]);
        setDimensions([]);
      }

      // 刷新项目列表
      await fetchProjects();

      alert('删除成功');
    } catch (error) {
      alert(error instanceof Error ? error.message : '删除失败');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleSort = (key: keyof Project) => {
    let direction: 'asc' | 'desc' | null = 'asc'
    if (sortConfig && sortConfig.key === key) {
      if (sortConfig.direction === 'asc') {
      direction = 'desc'
      } else if (sortConfig.direction === 'desc') {
        direction = null
      }
    }
    setSortConfig(direction ? { key, direction } : null)

    const sortedProjects = [...projects].sort((a, b) => {
      if (!direction) return 0;
      if ((a[key] ?? '') < (b[key] ?? '')) return direction === 'asc' ? -1 : 1;
      if ((a[key] ?? '') > (b[key] ?? '')) return direction === 'asc' ? 1 : -1;
      return 0;
    })
    setProjects(sortedProjects)
  }

  // 拖动排序
  const handleDragEnd = async (result: any) => {
    if (!result.destination) return;

    const items = Array.from(projects);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setProjects(items);

    // 保存新的项目 id 顺序到 Bridges
    const newProjectIds = items.map(p => p.id);

    // 获取当前用户
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    // 获取当前时间（ISO 格式）
    const now = new Date().toISOString();

    // 更新 Bridges 表
    await supabase
      .from('Bridges')
      .update({
        project_ids: newProjectIds,
        updated_at: now,
      })
      .eq('uid', user.id);
  };

  // 通用表格渲染
  const renderTable = (
    items: Item[],
    title: string,
    showQuantity: boolean = true,
    showCustomer: boolean = false
  ) => {
    const rowCount = Math.max(items.length, 5);

    return (
      <div>
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        <table className="min-w-full border">
          <thead>
            <tr className="bg-gray-100">
              <th className="border p-2">编号</th>
              <th className="border p-2">描述</th>
              <th className="border p-2">长度</th>
              <th className="border p-2">宽度</th>
              {showQuantity && (
                <th className="border p-2">数量</th>
              )}
              {showCustomer && (
                <th className="border p-2">客户</th>
              )}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: rowCount }).map((_, index) => {
              const item = items[index];
              return item ? (
                <tr key={item.id} className="hover:bg-gray-50">
                  <td className="border p-2">{index + 1}</td>
                  <td className="border p-2">{item.description || '-'}</td>
                  <td className="border p-2">{item.length}</td>
                  <td className="border p-2">{item.width}</td>
                  {showQuantity && <td className="border p-2">{item.quantity}</td>}
                  {showCustomer && <td className="border p-2">{item.customer || ''}</td>}
                </tr>
              ) : (
                <tr key={`empty-${index}`}>
                  <td className="border p-2">{index + 1}</td>
                  <td className="border p-2"></td>
                  <td className="border p-2"></td>
                  <td className="border p-2"></td>
                  {showQuantity && <td className="border p-2"></td>}
                  {showCustomer && <td className="border p-2"></td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  // 窗口样式
  const windowClass = "rounded-lg shadow-lg border bg-white flex flex-col h-full";
  const windowTitleClass = "bg-blue-600 text-white px-4 py-2 rounded-t-lg font-bold text-lg";

  const handleEdit = () => {
    setIsEditing(true);
    if (selectedProject) {
      window.location.href = `/project/${selectedProject}`;
    }
  };

  const handleLayout = () => {
    if (selectedProject) {
      window.location.href = `/layout/${selectedProject}`;
    }
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
    if (isCreating) return; // Prevent multiple clicks
    setIsCreating(true);

    try {
      // 获取当前用户
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        alert('请先登录');
        setIsCreating(false);
        return;
      }

      // 获取所有项目名称
      const { data: projectsData } = await supabase
        .from('Projects')
        .select('name');

      const existingNames = new Set(projectsData?.map(p => p.name) || []);

      // 找到可用的名称
      let i = 1;
      let newName = `new_${i}`;
      while (existingNames.has(newName)) {
        i++;
        newName = `new_${i}`;
      }

      // 创建新项目
      const defaultPlates = [{
        id: 1,
        width: 1220,
        length: 2440,
        quantity: 100,
        description: "default"
      }];

      const { data: newProject, error: projectError } = await supabase
        .from('Projects')
        .insert([{
          name: newName,
          uid: user.id,
          plates: defaultPlates,
          orders: [],
          others: []
        }])
        .select()
        .single();

      if (projectError) {
        alert('创建项目失败: ' + projectError.message);
        setIsCreating(false);
        return;
      }

      // 获取当前用户的 Bridges 记录
      const { data: bridgeData, error: bridgeError } = await supabase
        .from('Bridges')
        .select('*')  // 选择所有字段以确保我们有完整的记录
        .eq('uid', user.id)
        .single();

      // 准备新的 project_ids 数组
      const existingProjectIds = bridgeData?.project_ids || [];
      const newProjectIds = [...existingProjectIds, newProject.id];
      const now = new Date().toISOString();

      let updateError;
      if (bridgeError?.code === 'PGRST116') {
        // 记录不存在，创建新记录
        const { error } = await supabase
          .from('Bridges')
          .insert({
            uid: user.id,
            project_ids: newProjectIds,
            updated_at: now
          });
        updateError = error;
      } else if (!bridgeError) {
        // 记录存在，更新现有记录
        const { error } = await supabase
          .from('Bridges')
          .update({
            project_ids: newProjectIds,
            updated_at: now
          })
          .eq('uid', user.id);
        updateError = error;
      } else {
        // 其他错误
        updateError = bridgeError;
      }

      if (updateError) {
        alert('更新项目列表失败: ' + updateError.message);
        setIsCreating(false);
        return;
      }

      // 成功后刷新项目列表
      await fetchProjects();

      // 跳转到新项目的编辑页面
      router.push(`/project/${newProject.id}`);
    } catch (error) {
      alert('创建项目失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto my-8 rounded-2xl shadow-2xl border bg-white flex flex-col h-[92vh]">
      {/* 顶部栏 - 添加 flex justify-between */}
      <div className="px-6 py-3 border-b bg-blue-50 flex justify-between items-center">
        {/* 左侧项目名 */}
        <div className="text-xl font-bold">
          {selectedProject ? projects.find(p => p.id === selectedProject)?.name : '未选择项目'}
        </div>
        {/* 右侧退出按钮 */}
        <button
          onClick={handleLogout}
          className="text-gray-600 hover:text-gray-800 px-4 py-2 rounded-lg flex items-center gap-2"
        >
          <span>退出登录</span>
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 001 1h12a1 1 0 001-1V4a1 1 0 00-1-1H3zm11 4a1 1 0 10-2 0v4a1 1 0 102 0V7z" clipRule="evenodd" />
            <path d="M13.293 7.293a1 1 0 011.414 0L16 8.586l1.293-1.293a1 1 0 111.414 1.414l-2 2a1 1 0 01-1.414 0l-2-2a1 1 0 010-1.414z" />
          </svg>
        </button>
      </div>

      {/* 内容区 */}
      <div className="flex-1 flex flex-col h-0 p-6">
        {/* 上半部分：项目窗口 */}
        <div className="flex-1 min-h-0">
          <div className={`${windowClass} h-full`}>
            <div className={windowTitleClass}>项目列表</div>
            {/* 固定高度，最多显示10行，超出可滚动 */}
            <div className="overflow-y-auto" style={{ maxHeight: '440px' }}>
              <DragDropContext onDragEnd={handleDragEnd}>
                <Droppable droppableId="projects">
                  {(provided) => (
                    <table className="min-w-full border mb-2">
                      <thead>
                        <tr className="bg-gray-100">
                          <th className="border p-2 group relative">
                            <div className="flex items-center justify-between">
                              <span>编号</span>
                              <div className="flex items-center">
                                <span className="text-gray-400 opacity-0 group-hover:opacity-100 text-xs mr-1">可拖拽排序</span>
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                                </svg>
                              </div>
                            </div>
                          </th>
                          <th 
                            className="border p-2 cursor-pointer hover:bg-gray-200 group relative" 
                            onClick={() => handleSort('name')}
                            title="点击排序"
                          >
                            <div className="flex items-center justify-between">
                              <span>名称</span>
                              <div className="flex items-center">
                                <span className="text-gray-400 opacity-0 group-hover:opacity-100 text-xs mr-1">点击排序</span>
                                <span className="text-gray-500 w-4">
                                  {sortConfig?.key === 'name' && (
                                    sortConfig.direction === 'asc' ? '↑' : '↓'
                                  )}
                                </span>
                              </div>
                            </div>
                          </th>
                          <th 
                            className="border p-2 cursor-pointer hover:bg-gray-200 group relative" 
                            onClick={() => handleSort('details')}
                            title="点击排序"
                          >
                            <div className="flex items-center justify-between">
                              <span>详情</span>
                              <div className="flex items-center">
                                <span className="text-gray-400 opacity-0 group-hover:opacity-100 text-xs mr-1">点击排序</span>
                                <span className="text-gray-500 w-4">
                                  {sortConfig?.key === 'details' && (
                                    sortConfig.direction === 'asc' ? '↑' : '↓'
                                  )}
                                </span>
                              </div>
                            </div>
                          </th>
                          <th 
                            className="border p-2 cursor-pointer hover:bg-gray-200 group relative" 
                            onClick={() => handleSort('description')}
                            title="点击排序"
                          >
                            <div className="flex items-center justify-between">
                              <span>描述</span>
                              <div className="flex items-center">
                                <span className="text-gray-400 opacity-0 group-hover:opacity-100 text-xs mr-1">点击排序</span>
                                <span className="text-gray-500 w-4">
                                  {sortConfig?.key === 'description' && (
                                    sortConfig.direction === 'asc' ? '↑' : '↓'
                                  )}
                                </span>
                              </div>
                            </div>
                          </th>
                          <th 
                            className="border p-2 cursor-pointer hover:bg-gray-200 group relative" 
                            onClick={() => handleSort('updated_at')}
                            title="点击排序"
                          >
                            <div className="flex items-center justify-between">
                              <span>修改时间</span>
                              <div className="flex items-center">
                                <span className="text-gray-400 opacity-0 group-hover:opacity-100 text-xs mr-1">点击排序</span>
                                <span className="text-gray-500 w-4">
                                  {sortConfig?.key === 'updated_at' && (
                                    sortConfig.direction === 'asc' ? '↑' : '↓'
                                  )}
                                </span>
                              </div>
                            </div>
                          </th>
                        </tr>
                      </thead>
                      <tbody {...provided.droppableProps} ref={provided.innerRef}>
                        {(projects.length > 0
                          ? projects
                          : Array.from({ length: 10 }).map(() => undefined)
                        ).map((project, index) => (
                          project ? (
                            <Draggable key={project.id} draggableId={String(project.id)} index={index}>
                              {(provided) => (
                                <tr
                                  ref={provided.innerRef}
                                  {...provided.draggableProps}
                                  className={`hover:bg-gray-50 ${selectedProject === project.id ? 'bg-blue-50' : ''}`}
                                  onClick={() => handleSelectProject(project.id)}
                                >
                                  <td
                                    className="border p-2 cursor-move bg-gray-50 hover:bg-gray-100 group relative"
                                    {...provided.dragHandleProps}
                                    title="拖动排序"
                                  >
                                    <div className="flex items-center justify-between">
                                      <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                                        <input
                                          type="checkbox"
                                          checked={selectedProjects.has(project.id)}
                                          onChange={(e) => handleCheckProject(project.id, e.target.checked)}
                                          className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                                        />
                                        <span>{index + 1}</span>
                                      </div>
                                      <div className="flex items-center">
                                        <span className="text-gray-400 opacity-0 group-hover:opacity-100 text-xs mr-1">拖动排序</span>
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                                        </svg>
                                      </div>
                                    </div>
                                  </td>
                                  <td className="border p-2">{project.name}</td>
                                  <td className="border p-2">{project.details}</td>
                                  <td className="border p-2">{project.description}</td>
                                  <td className="border p-2">
                                    {new Date(project.updated_at).toLocaleString()}
                                  </td>
                                </tr>
                              )}
                            </Draggable>
                          ) : (
                            <tr key={`empty-${index}`}>
                              <td className="border p-2">{index + 1}</td>
                              <td className="border p-2"></td>
                              <td className="border p-2"></td>
                              <td className="border p-2"></td>
                              <td className="border p-2"></td>
                            </tr>
                          )
                        ))}
                        {provided.placeholder}
                      </tbody>
                    </table>
                  )}
                </Droppable>
              </DragDropContext>
            </div>
            {/* 功能键区域 */}
            <div className="p-2 border-t flex justify-end gap-2">
              {selectedProject && (
                <button
                  className="bg-blue-500 text-white px-3 py-1 rounded"
                  onClick={handleEdit}
                >
                  编辑
                </button>
              )}
              <button 
                className={`${isCreating ? 'bg-green-400' : 'bg-green-500'} text-white px-3 py-1 rounded flex items-center gap-2 ${isCreating ? 'cursor-not-allowed' : 'hover:bg-green-600'}`}
                onClick={handleNew}
                disabled={isCreating}
              >
                {isCreating ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>创建中...</span>
                  </>
                ) : (
                  '新增'
                )}
              </button>
              <button 
                className={`${isDeleting ? 'bg-red-400' : 'bg-red-500'} text-white px-3 py-1 rounded flex items-center gap-2 ${isDeleting ? 'cursor-not-allowed' : 'hover:bg-red-600'} ${selectedProjects.size === 0 ? 'opacity-50' : ''}`}
                onClick={handleDeleteProjects}
                disabled={isDeleting || selectedProjects.size === 0}
              >
                {isDeleting ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>删除中...</span>
                  </>
                ) : (
                  <>
                    {selectedProjects.size > 0 ? `删除 (${selectedProjects.size})` : '删除'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* 下半部分：三栏窗口 */}
        <div className="flex-1 min-h-0">
          <div className="h-full grid grid-cols-3 gap-4">
            {/* 板件信息 */}
            <div className={windowClass}>
              <div className={windowTitleClass}>板件信息</div>
              <div className="flex-1 overflow-auto">
                {renderTable(parts, '', true)}
              </div>
            </div>
            {/* 零件信息 */}
            <div className={windowClass}>
              <div className={windowTitleClass}>零件信息</div>
              <div className="flex-1 overflow-auto">
                {renderTable(components, '', true)}
              </div>
            </div>
            {/* 常用尺寸信息 */}
            <div className={windowClass}>
              <div className={windowTitleClass}>常用尺寸信息</div>
              <div className="flex-1 overflow-auto">
                {renderTable(dimensions, '', false, true)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
