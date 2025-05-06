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

  // 新增三个表格的排序状态
  const [partsSort, setPartsSort] = useState<{ key: keyof Item, direction: 'asc' | 'desc' } | null>(null);
  const [componentsSort, setComponentsSort] = useState<{ key: keyof Item, direction: 'asc' | 'desc' } | null>(null);
  const [dimensionsSort, setDimensionsSort] = useState<{ key: keyof Item, direction: 'asc' | 'desc' } | null>(null);

  const [isEditing, setIsEditing] = useState(false);

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

  const handleSort = (key: keyof Project) => {
    let direction: 'asc' | 'desc' = 'asc'
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc'
    }
    setSortConfig({ key, direction })

    const sortedProjects = [...projects].sort((a, b) => {
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

  // 通用排序函数
  function sortItems(items: Item[], sortConfig: { key: keyof Item, direction: 'asc' | 'desc' } | null) {
    if (!sortConfig) return items;
    const sorted = [...items].sort((a, b) => {
      if ((a[sortConfig.key] ?? '') < (b[sortConfig.key] ?? '')) return sortConfig.direction === 'asc' ? -1 : 1;
      if ((a[sortConfig.key] ?? '') > (b[sortConfig.key] ?? '')) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
    return sorted;
  }

  // 三个表格的排序事件
  const handlePartsSort = (key: keyof Item) => {
    setPartsSort((prev) => {
      if (prev && prev.key === key && prev.direction === 'asc') {
        return { key, direction: 'desc' };
      }
      return { key, direction: 'asc' };
    });
  };
  const handleComponentsSort = (key: keyof Item) => {
    setComponentsSort((prev) => {
      if (prev && prev.key === key && prev.direction === 'asc') {
        return { key, direction: 'desc' };
      }
      return { key, direction: 'asc' };
    });
  };
  const handleDimensionsSort = (key: keyof Item) => {
    setDimensionsSort((prev) => {
      if (prev && prev.key === key && prev.direction === 'asc') {
        return { key, direction: 'desc' };
      }
      return { key, direction: 'asc' };
    });
  };

  // 通用表格渲染
  const renderTable = (
    items: Item[],
    title: string,
    sortConfig: { key: keyof Item, direction: 'asc' | 'desc' } | null,
    onSort: (key: keyof Item) => void,
    showQuantity: boolean = true,
    showCustomer: boolean = false
  ) => {
    const sortedItems = sortItems(items, sortConfig);
    const rowCount = Math.max(sortedItems.length, 10);

    return (
      <div>
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        <table className="min-w-full border">
          <thead>
            <tr className="bg-gray-100">
              <th className="border p-2">编号</th>
              <th className="border p-2 cursor-pointer" onClick={() => onSort('description')}>描述</th>
              <th className="border p-2 cursor-pointer" onClick={() => onSort('length')}>长度</th>
              <th className="border p-2 cursor-pointer" onClick={() => onSort('width')}>宽度</th>
              {showQuantity && (
                <th className="border p-2 cursor-pointer" onClick={() => onSort('quantity')}>数量</th>
              )}
              {showCustomer && (
                <th className="border p-2">客户</th>
              )}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: rowCount }).map((_, index) => {
              const item = sortedItems[index];
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
                          <th className="border p-2">编号</th>
                          <th className="border p-2 cursor-pointer" onClick={() => handleSort('name')}>名称</th>
                          <th className="border p-2 cursor-pointer" onClick={() => handleSort('details')}>详情</th>
                          <th className="border p-2 cursor-pointer" onClick={() => handleSort('description')}>描述</th>
                          <th className="border p-2 cursor-pointer" onClick={() => handleSort('updated_at')}>修改时间</th>
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
                                    className="border p-2 cursor-move"
                                    {...provided.dragHandleProps}
                                    title="拖动排序"
                                  >
                                    {index + 1}
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
              <button className="bg-green-500 text-white px-3 py-1 rounded">新增</button>
              <button className="bg-red-500 text-white px-3 py-1 rounded">删除</button>
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
                {renderTable(parts, '', partsSort, handlePartsSort)}
              </div>
            </div>
            {/* 零件信息 */}
            <div className={windowClass}>
              <div className={windowTitleClass}>零件信息</div>
              <div className="flex-1 overflow-auto">
                {renderTable(components, '', componentsSort, handleComponentsSort)}
              </div>
            </div>
            {/* 常用尺寸信息 */}
            <div className={windowClass}>
              <div className={windowTitleClass}>常用尺寸信息</div>
              <div className="flex-1 overflow-auto">
                {renderTable(dimensions, '', dimensionsSort, handleDimensionsSort, false, true)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
