'use client';

import { useParams, useRouter } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { supabase } from '@/utils/supabaseClient';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import UnsavedChangesPrompt from '@/components/UnsavedChangesPrompt';
import { getApiUrl } from '@/config/api';

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;
  const [userId, setUserId] = useState<string | null>(null);
  const [project, setProject] = useState<any>(null);
  const [plates, setPlates] = useState<any[]>([]);
  const [orders, setOrders] = useState<any[]>([]);
  const [others, setOthers] = useState<any[]>([]);
  const [initialData, setInitialData] = useState<any>(null);
  const [projectName, setProjectName] = useState('');
  const [projectDetails, setProjectDetails] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [sawBlade, setSawBlade] = useState<number>(0);
  const [selectedRow, setSelectedRow] = useState<{
    type: 'plates' | 'orders' | 'others';
    index: number;
    cells: HTMLTableCellElement[];
  } | null>(null);
  const [optimization, setOptimization] = useState<number>(1);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUserId(user?.id ?? null);
      const { data } = await supabase.from('Projects').select('*').eq('id', projectId).single();
      setProject(data);
      setPlates(data.plates || []);
      setOrders(data.orders || []);
      setOthers(data.others || []);
      setProjectName(data.name || '');
      setProjectDetails(data.details || '');
      setProjectDescription(data.description || '');
      setSawBlade(data.saw_blade || 0);
      setInitialData({
        name: data.name || '',
        details: data.details || '',
        description: data.description || '',
        saw_blade: data.saw_blade || 0,
        plates: JSON.stringify(data.plates || []),
        orders: JSON.stringify(data.orders || []),
        others: JSON.stringify(data.others || []),
      });
    };
    if (projectId) fetchData();
  }, [projectId]);

  const hasChanges = useMemo(
    () =>
      projectName !== initialData?.name ||
      projectDetails !== initialData?.details ||
      projectDescription !== initialData?.description ||
      sawBlade !== initialData?.saw_blade ||
      JSON.stringify(plates) !== initialData?.plates ||
      JSON.stringify(orders) !== initialData?.orders ||
      JSON.stringify(others) !== initialData?.others,
    [
      initialData,
      projectName,
      projectDetails,
      projectDescription,
      sawBlade,
      plates,
      orders,
      others,
    ]
  );

  const validatePositiveInteger = (value: string): boolean => {
    const num = Number(value);
    return Number.isInteger(num) && num > 0;
  };

  const validatePositiveNumber = (value: string): boolean => {
    const num = Number(value);
    return Number.isFinite(num) && num > 0;
  };

  const sanitizePositiveNumber = (value: string, fallback: number): number => {
    const num = Number(value.trim());
    return Number.isFinite(num) && num > 0 ? num : fallback;
  };

  const validateData = useCallback(() => {
    const validateArray = (arr: any[]) => {
      return arr.every(item => 
        validatePositiveInteger(item.length.toString()) && 
        validatePositiveInteger(item.width.toString()) && 
        validatePositiveInteger(item.quantity?.toString() || '1')
      );
    };

    if (!validatePositiveNumber(sawBlade.toString())) {
      alert('锯片宽度必须为正数（可包含小数）');
      return false;
    }

    if (!validateArray(plates)) {
      alert('板件信息中的长度、宽度和数量必须为正整数');
      return false;
    }

    if (!validateArray(orders)) {
      alert('零件信息中的长度、宽度和数量必须为正整数');
      return false;
    }

    if (!validateArray(others)) {
      alert('常用尺寸信息中的长度、宽度必须为正整数');
      return false;
    }

    return true;
  }, [sawBlade, plates, orders, others]);

  const handleSave = useCallback(async () => {
    if (!validateData()) {
      return;
    }

    const { error } = await supabase.from('Projects').update({
      name: projectName,
      details: projectDetails,
      description: projectDescription,
      saw_blade: sawBlade,
      plates,
      orders,
      others,
      updated_at: new Date().toISOString()
    }).eq('id', projectId).eq('uid', userId);

    if (error) {
      alert('保存失败: ' + error.message);
    } else {
      setProject(prev => ({
        ...prev,
        updated_at: new Date().toISOString()
      }));
      setInitialData({
        name: projectName,
        details: projectDetails,
        description: projectDescription,
        saw_blade: sawBlade,
        plates: JSON.stringify(plates),
        orders: JSON.stringify(orders),
        others: JSON.stringify(others),
      });
      alert('保存成功');
    }
  }, [
    validateData,
    projectId,
    userId,
    projectName,
    projectDetails,
    projectDescription,
    sawBlade,
    plates,
    orders,
    others,
  ]);

  const handleBeforeUnload = useCallback((e: BeforeUnloadEvent) => {
    if (hasChanges) {
      e.preventDefault();
      e.returnValue = '';
    }
  }, [hasChanges]);

  const handleRouteChange = useCallback(() => {
    if (hasChanges) {
      if (!window.confirm('有未保存的更改，是否保存？')) {
        return;
      }
      void handleSave();
    }
  }, [hasChanges, handleSave]);

  useEffect(() => {
    window.addEventListener('beforeunload', handleBeforeUnload);

    const handleNavigation = () => {
      handleRouteChange();
    };

    window.addEventListener('popstate', handleNavigation);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('popstate', handleNavigation);
    };
  }, [handleBeforeUnload, handleRouteChange]);

  const handleBack = async () => {
    if (hasChanges) {
      if (window.confirm('有未保存的更改，是否保存？')) {
        await handleSave();
      }
    }
    router.push('/project');
  };

  const handleCellChange = (type: 'plates' | 'orders' | 'others', rowIndex: number, field: string, value: any) => {
    const setValue = (prev: any[]) => 
      prev.map((row, idx) => 
        idx === rowIndex ? { ...row, [field]: value } : row
      );

    switch(type) {
      case 'plates':
        setPlates(setValue);
        break;
      case 'orders':
        setOrders(setValue);
        break;
      case 'others':
        setOthers(setValue);
        break;
    }
  };

  const handleRowClick = (type: 'plates' | 'orders' | 'others', index: number, e: React.MouseEvent) => {
    const row = e.currentTarget as HTMLTableRowElement;
    const editableCells = Array.from(row.querySelectorAll('td[contenteditable="true"]')) as HTMLTableCellElement[];
    
    if (selectedRow) {
      selectedRow.cells.forEach(cell => {
        cell.style.backgroundColor = '';
      });
    }

    editableCells.forEach(cell => {
      cell.style.backgroundColor = '#f2f3f5';
    });

    setSelectedRow({
      type,
      index,
      cells: editableCells
    });
  };

  const handleKeyDown = async (e: React.KeyboardEvent) => {
    if (!selectedRow) return;

    if (e.key === 'Enter') {
      e.preventDefault();
      (e.target as HTMLElement).blur();
      return;
    }

    if (e.key === 'Tab') {
      e.preventDefault();
      const { type, index, cells } = selectedRow;
      const currentCell = e.target as HTMLElement;
      const currentIndex = cells.indexOf(currentCell as HTMLTableCellElement);
      
      const focusCell = (cell: HTMLElement) => {
        cell.focus();
        // 将光标移动到内容末尾
        const range = document.createRange();
        const selection = window.getSelection();
        range.selectNodeContents(cell);
        range.collapse(false); // false 表示折叠到末尾
        selection?.removeAllRanges();
        selection?.addRange(range);
      };

      const getNextRow = (currentType: string, currentIndex: number) => {
        const nextRow = document.querySelector(`tr[data-row="${currentType}-${currentIndex + 1}"]`);
        if (nextRow) {
          const nextCells = nextRow.querySelectorAll('td[contenteditable="true"]');
          if (nextCells.length > 0) {
            return nextCells[0] as HTMLElement;
          }
        }
        return null;
      };

      const getPrevRow = (currentType: string, currentIndex: number) => {
        const prevRow = document.querySelector(`tr[data-row="${currentType}-${currentIndex - 1}"]`);
        if (prevRow) {
          const prevCells = prevRow.querySelectorAll('td[contenteditable="true"]');
          if (prevCells.length > 0) {
            return prevCells[prevCells.length - 1] as HTMLElement;
          }
        }
        return null;
      };
      
      if (e.shiftKey) {
        // 向前跳转
        if (currentIndex > 0) {
          focusCell(cells[currentIndex - 1]);
        } else {
          const prevCell = getPrevRow(type, index);
          if (prevCell) {
            focusCell(prevCell);
          }
        }
      } else {
        // 向后跳转
        if (currentIndex < cells.length - 1) {
          focusCell(cells[currentIndex + 1]);
        } else {
          const nextCell = getNextRow(type, index);
          if (nextCell) {
            focusCell(nextCell);
            // 更新选中的行
            const nextRow = nextCell.closest('tr');
            if (nextRow) {
              const nextRowIndex = parseInt(nextRow.getAttribute('data-row')?.split('-')[1] || '0');
              const nextRowType = nextRow.getAttribute('data-row')?.split('-')[0] || '';
              const nextRowCells = Array.from(nextRow.querySelectorAll('td[contenteditable="true"]')) as HTMLTableCellElement[];
              setSelectedRow({
                type: nextRowType as 'plates' | 'orders' | 'others',
                index: nextRowIndex,
                cells: nextRowCells
              });
            }
          }
        }
      }
      return;
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
      e.preventDefault();
      const { type, index } = selectedRow;
      const data = type === 'plates' ? plates[index] :
                   type === 'orders' ? orders[index] :
                   others[index];
      const { id: _omittedId, ...copyData } = data;
      await navigator.clipboard.writeText(JSON.stringify(copyData));
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
      e.preventDefault();
      try {
        const { type, index } = selectedRow;
        const text = await navigator.clipboard.readText();
        const pasteData = JSON.parse(text);
        
        const setValue = (prev: any[]) => 
          prev.map((row, idx) => 
            idx === index ? { ...row, ...pasteData } : row
          );

        switch (type) {
          case 'plates':
            setPlates(setValue);
            break;
          case 'orders':
            setOrders(setValue);
            break;
          case 'others':
            setOthers(setValue);
            break;
        }
      } catch (error) {
        console.error('剪贴板读取错误: ', error);
      }
    }
  };

  const addNewRow = (type: 'plates' | 'orders' | 'others') => {
    const newRow = {
      id: type === 'plates' ? plates.length + 1 : type === 'orders' ? orders.length + 1 : others.length + 1,
      length: 0,
      width: 0,
      quantity: type === 'others' ? 0 : 1,
      description: '',
      ...(type === 'others' && { client: '' })
    };
    switch(type) {
      case 'plates':
        setPlates(prev => [...prev, { ...newRow, id: prev.length + 1 }]);
        break;
      case 'orders':
        setOrders(prev => [...prev, { ...newRow, id: prev.length + 1 }]);
        break;
      case 'others':
        setOthers(prev => [...prev, { ...newRow, id: prev.length + 1 }]);
        break;
    }
  };

  const deleteRow = (type: 'plates' | 'orders' | 'others', index: number) => {
    const updateData = (prev: any[]) => 
      prev.filter((_, idx) => idx !== index)
         .map((item, idx) => ({ ...item, id: idx + 1 }));

    switch(type) {
      case 'plates':
        setPlates(updateData);
        break;
      case 'orders':
        setOrders(updateData);
        break;
      case 'others':
        setOthers(updateData);
        break;
    }
  };

  const handleLayoutClick = async () => {
    const { data } = await supabase
      .from('Projects')
      .select('cutted')
      .eq('id', projectId)
      .single();

    if (data?.cutted && data.cutted.length > 0) {
      router.push(`/layout/${projectId}/1`);
    } else {
      alert('请先进行切板操作');
    }
  };

  const handleCutting = async () => {
    if (!userId) {
      alert('请先登录');
      return;
    }

    if (!validateData()) {
      return;
    }

    if (plates.length === 0 || orders.length === 0) {
      alert('请添加板材和零件信息');
      return;
    }

    setIsLoading(true);
    try {
      const { error: updateError } = await supabase
        .from('Projects')
        .update({
          plates,
          orders,
          others,
          saw_blade: sawBlade,
          updated_at: new Date().toISOString()
        })
        .eq('id', projectId);

      if (updateError) throw updateError;

      setProject(prev => ({
        ...prev,
        updated_at: new Date().toISOString()
      }));

      const response = await fetch(getApiUrl('OPTIMIZE'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          uid: userId,
          project_id: projectId,
          plates,
          orders,
          others,
          optimization: Boolean(optimization),
          saw_blade: Number(sawBlade) || 4
        }),
      });

      const data = await response.json();

      if (data.code === 0) {
        const { error: cutError } = await supabase
          .from('Projects')
          .update({
            cutted: data.cutting_plans,
            updated_at: new Date().toISOString()
          })
          .eq('id', projectId);

        if (cutError) throw cutError;

        setProject(prev => ({
          ...prev,
          updated_at: new Date().toISOString()
        }));

        alert('板件、零件和其他尺寸信息已保存');
        router.push(`/layout/${projectId}/1`);
      } else {
        throw new Error(data.message || '切板失败');
      }
    } catch (error: any) {
      alert(error.message || '切板失败');
    } finally {
      setIsLoading(false);
    }
  };

  const handleOthersDragEnd = (result: any) => {
    if (!result.destination) return;

    const items = Array.from(others);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    const updatedItems = items.map((item, index) => ({
      ...item,
      id: index + 1
    }));

    setOthers(updatedItems);
  };

  if (!project) {
    return (
      <div className="page-gallery flex min-h-screen items-center justify-center text-ink-muted">
        <div className="flex items-center gap-3">
          <svg className="w-5 h-5 animate-spin text-accent" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
          加载中…
        </div>
      </div>
    );
  }

  const editableCellClass =
    "border p-2 focus:outline-none focus:border-[#c5c5c7] focus:ring-0 bg-muted";

  return (
    <div className="page-gallery">
      <div className="page-gallery-inner">
      <UnsavedChangesPrompt hasChanges={hasChanges} onSave={handleSave} />
      
      {/* 导航 */}
      <div className="mb-8 flex gap-2">
        <span className="badge badge-gray px-3 py-1.5 text-xs font-medium cursor-default border border-hairline bg-surface shadow-sm">
          项目配置
        </span>
        <button 
          type="button" 
          className="badge badge-gray px-3 py-1.5 text-xs cursor-pointer border border-transparent bg-transparent hover:bg-muted transition-colors" 
          onClick={handleLayoutClick}
        >
          切板统计
        </button>
      </div>
      
      {/* 标题与操作 */}
      <div className="mb-10 flex flex-col gap-6 border-b border-hairline pb-6 md:flex-row md:items-start md:justify-between animate-fade-in-up">
        <h1 className="text-xl font-medium tracking-tight text-ink flex items-center gap-2.5">
          <svg className="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
          {projectName || '未命名项目'}
        </h1>
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="mr-3 flex flex-wrap items-center gap-2 text-xs text-ink-muted">
            <label className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 transition-all ${optimization === 1 ? 'bg-[#e0f2fe] text-[#0284c7] font-medium' : 'hover:bg-muted'}`}>
              <input
                type="radio"
                name="optimization"
                value="1"
                checked={optimization === 1}
                onChange={() => setOptimization(1)}
                className="accent-[#0284c7] w-3 h-3"
              />
              <span>优化模式</span>
            </label>
            <label className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 transition-all ${optimization === 0 ? 'bg-[#f3e8ff] text-[#9333ea] font-medium' : 'hover:bg-muted'}`}>
              <input
                type="radio"
                name="optimization"
                value="0"
                checked={optimization === 0}
                onChange={() => setOptimization(0)}
                className="accent-[#9333ea] w-3 h-3"
              />
              <span>正常模式</span>
            </label>
          </div>
          
          <button
            type="button"
            className="has-tooltip icon-btn icon-btn-blue shadow-sm border border-hairline bg-surface"
            onClick={handleCutting}
            disabled={isLoading}
          >
            {isLoading ? (
              <svg className="w-4 h-4 animate-spin text-accent" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" /></svg>
            )}
            <span className="tooltip-text">执行切板</span>
          </button>
          
          <button 
            type="button" 
            className="has-tooltip icon-btn icon-btn-green shadow-sm border border-hairline bg-surface" 
            onClick={handleSave}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" /></svg>
            <span className="tooltip-text">保存更改</span>
          </button>
          
          <div className="w-[1px] h-4 bg-border-hairline mx-1"></div>
          
          <button 
            type="button" 
            className="has-tooltip icon-btn hover:bg-muted" 
            onClick={handleBack}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
            <span className="tooltip-text">返回列表</span>
          </button>
        </div>
      </div>

      {/* 项目基本信息 */}
      <div className="mb-8">
        <div className="table-container hover-lift shadow-sm animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <div className="table-title flex items-center gap-1.5 text-xs text-ink-muted">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            全局配置
          </div>
          <div className="table-content">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="border p-2">名称</th>
                  <th className="border p-2">详情</th>
                  <th className="border p-2">描述</th>
                  <th className="border p-2">锯片宽度</th>
                  <th className="border p-2">修改时间</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td 
                    className={editableCellClass}
                    contentEditable
                    suppressContentEditableWarning
                    onBlur={(e) => setProjectName(e.currentTarget.textContent || '')}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                    }}
                  >
                    {projectName}
                  </td>
                  <td 
                    className={editableCellClass}
                    contentEditable
                    suppressContentEditableWarning
                    onBlur={(e) => setProjectDetails(e.currentTarget.textContent || '')}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                    }}
                  >
                    {projectDetails}
                  </td>
                  <td 
                    className={editableCellClass}
                    contentEditable
                    suppressContentEditableWarning
                    onBlur={(e) => setProjectDescription(e.currentTarget.textContent || '')}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                    }}
                  >
                    {projectDescription}
                  </td>
                  <td 
                    className={editableCellClass}
                    contentEditable
                    suppressContentEditableWarning
                    onBlur={(e) => {
                      const value = e.currentTarget.textContent || '0';
                      if (validatePositiveNumber(value)) {
                        setSawBlade(parseFloat(value));
                      } else {
                        alert('锯片宽度必须为正数（可包含小数）');
                        e.currentTarget.textContent = sawBlade.toString();
                      }
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                    }}
                  >
                    {sawBlade}
                  </td>
                  <td className="border p-2 text-ink-muted text-xs">
                    {new Date(project.updated_at).toLocaleString()}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 数据表格 */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* 板件信息 */}
        <div className="table-container hover-lift shadow-sm animate-fade-in-up flex flex-col h-full" style={{ animationDelay: '0.2s' }}>
          <div className="table-title flex items-center gap-1.5 text-xs">
            <svg className="w-3.5 h-3.5 text-[#0284c7]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>
            板件清单
            <span className="badge badge-blue ml-auto">{plates.length}</span>
          </div>
          <div className="table-content flex-1 overflow-auto max-h-[400px]">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="border p-2 w-10 text-center">#</th>
                  <th className="border p-2">L</th>
                  <th className="border p-2">W</th>
                  <th className="border p-2">数</th>
                  <th className="border p-2">说明</th>
                  <th className="border p-2 w-10 text-center"></th>
                </tr>
              </thead>
              <tbody onKeyDown={handleKeyDown}>
                {plates.map((plate, index) => (
                  <tr 
                    key={index} 
                    className="cursor-default"
                    onClick={(e) => handleRowClick('plates', index, e)}
                    data-row={`plates-${index}`}
                  >
                    <td className="border p-2 text-center text-ink-muted">{index + 1}</td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                      onBlur={(e) => {
                        const value = sanitizePositiveNumber(e.currentTarget.textContent || '0', plate.length);
                        handleCellChange('plates', index, 'length', value);
                        e.currentTarget.textContent = value.toString();
                      }}
                    >
                      {plate.length}
                    </td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                      onBlur={(e) => {
                        const value = sanitizePositiveNumber(e.currentTarget.textContent || '0', plate.width);
                        handleCellChange('plates', index, 'width', value);
                        e.currentTarget.textContent = value.toString();
                      }}
                    >
                      {plate.width}
                    </td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                      onBlur={(e) => {
                        const value = sanitizePositiveNumber(e.currentTarget.textContent || '1', plate.quantity || 1);
                        handleCellChange('plates', index, 'quantity', value);
                        e.currentTarget.textContent = value.toString();
                      }}
                    >
                      {plate.quantity}
                    </td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                      onBlur={(e) => handleCellChange('plates', index, 'description', e.currentTarget.textContent || '')}
                    >
                      {plate.description}
                    </td>
                    <td className="border p-2 text-center">
                      <button
                        type="button"
                        onClick={() => deleteRow('plates', index)}
                        className="has-tooltip icon-btn icon-btn-red !w-5 !h-5"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                        <span className="tooltip-text">删除板件</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-actions flex justify-center py-2 bg-surface border-t-0">
            <button type="button" onClick={() => addNewRow('plates')} className="btn-gallery-ghost text-xs flex items-center gap-1 py-1 w-full justify-center text-[#0284c7] hover:bg-[#eff6ff] border-dashed border">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
              添加新行
            </button>
          </div>
        </div>

        {/* 零件信息 */}
        <div className="table-container hover-lift shadow-sm animate-fade-in-up flex flex-col h-full" style={{ animationDelay: '0.3s' }}>
          <div className="table-title flex items-center gap-1.5 text-xs">
            <svg className="w-3.5 h-3.5 text-[#9333ea]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
            待切零件
            <span className="badge badge-purple ml-auto">{orders.length}</span>
          </div>
          <div className="table-content flex-1 overflow-auto max-h-[400px]">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="border p-2 w-10 text-center">#</th>
                  <th className="border p-2">L</th>
                  <th className="border p-2">W</th>
                  <th className="border p-2">数</th>
                  <th className="border p-2">说明</th>
                  <th className="border p-2 w-10 text-center"></th>
                </tr>
              </thead>
              <tbody onKeyDown={handleKeyDown}>
                {orders.map((order, index) => (
                  <tr 
                    key={index} 
                    className="cursor-default"
                    onClick={(e) => handleRowClick('orders', index, e)}
                    data-row={`orders-${index}`}
                  >
                    <td className="border p-2 text-center text-ink-muted">{index + 1}</td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                        onBlur={(e) => {
                        const value = sanitizePositiveNumber(e.currentTarget.textContent || '0', order.length);
                        handleCellChange('orders', index, 'length', value);
                        e.currentTarget.textContent = value.toString();
                      }}
                    >
                      {order.length}
                    </td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                        onBlur={(e) => {
                        const value = sanitizePositiveNumber(e.currentTarget.textContent || '0', order.width);
                        handleCellChange('orders', index, 'width', value);
                        e.currentTarget.textContent = value.toString();
                      }}
                    >
                      {order.width}
                    </td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                        onBlur={(e) => {
                        const value = sanitizePositiveNumber(e.currentTarget.textContent || '1', order.quantity || 1);
                        handleCellChange('orders', index, 'quantity', value);
                        e.currentTarget.textContent = value.toString();
                      }}
                    >
                      {order.quantity}
                    </td>
                    <td 
                      className={editableCellClass}
                      contentEditable
                      suppressContentEditableWarning
                      onBlur={(e) => handleCellChange('orders', index, 'description', e.currentTarget.textContent || '')}
                    >
                      {order.description}
                    </td>
                    <td className="border p-2 text-center">
                      <button
                        type="button"
                        onClick={() => deleteRow('orders', index)}
                        className="has-tooltip icon-btn icon-btn-red !w-5 !h-5"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                        <span className="tooltip-text">删除零件</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-actions flex justify-center py-2 bg-surface border-t-0">
            <button type="button" onClick={() => addNewRow('orders')} className="btn-gallery-ghost text-xs flex items-center gap-1 py-1 w-full justify-center text-[#9333ea] hover:bg-[#faf5ff] border-dashed border">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
              添加新行
            </button>
          </div>
        </div>

        {/* 常用尺寸信息 */}
        <div className="table-container hover-lift shadow-sm animate-fade-in-up flex flex-col h-full" style={{ animationDelay: '0.4s' }}>
          <div className="table-title flex items-center gap-1.5 text-xs">
            <svg className="w-3.5 h-3.5 text-[#d97706]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>
            常用尺寸/余料
            <span className="badge badge-amber ml-auto">{others.length}</span>
          </div>
          <div className="table-content flex-1 overflow-auto max-h-[400px]">
            <DragDropContext onDragEnd={handleOthersDragEnd}>
              <Droppable droppableId="others">
                {(provided) => (
                  <table className="min-w-full">
                    <thead>
                      <tr>
                        <th className="border p-2 w-10 text-center">
                          <div className="flex items-center justify-center gap-0.5" title="可拖拽排序">
                            <span>#</span>
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 text-ink-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                            </svg>
                          </div>
                        </th>
                        <th className="border p-2">L</th>
                        <th className="border p-2">W</th>
                        <th className="border p-2">客户</th>
                        <th className="border p-2">说明</th>
                        <th className="border p-2 w-10 text-center"></th>
                      </tr>
                    </thead>
                    <tbody {...provided.droppableProps} ref={provided.innerRef} onKeyDown={handleKeyDown}>
                      {others.map((other, index) => (
                        <Draggable key={index} draggableId={`other-${index}`} index={index}>
                          {(provided) => (
                            <tr
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              className="cursor-default"
                              onClick={(e) => handleRowClick('others', index, e)}
                              data-row={`others-${index}`}
                            >
                              <td 
                                className="border cursor-move bg-muted/50 p-2 text-center text-ink-muted hover:bg-muted"
                                {...provided.dragHandleProps}
                                title="拖动排序"
                              >
                                {index + 1}
                              </td>
                              <td 
                                className={editableCellClass}
                                contentEditable
                                suppressContentEditableWarning
                                onBlur={(e) => {
                                  const value = sanitizePositiveNumber(e.currentTarget.textContent || '0', other.length);
                                  handleCellChange('others', index, 'length', value);
                                  e.currentTarget.textContent = value.toString();
                                }}
                              >
                                {other.length}
                              </td>
                              <td 
                                className={editableCellClass}
                                contentEditable
                                suppressContentEditableWarning
                                onBlur={(e) => {
                                  const value = sanitizePositiveNumber(e.currentTarget.textContent || '0', other.width);
                                  handleCellChange('others', index, 'width', value);
                                  e.currentTarget.textContent = value.toString();
                                }}
                              >
                                {other.width}
                              </td>
                              <td 
                                className={editableCellClass}
                                contentEditable
                                suppressContentEditableWarning
                                onBlur={(e) => handleCellChange('others', index, 'client', e.currentTarget.textContent || '')}
                              >
                                {other.client}
                              </td>
                              <td 
                                className={editableCellClass}
                                contentEditable
                                suppressContentEditableWarning
                                onBlur={(e) => handleCellChange('others', index, 'description', e.currentTarget.textContent || '')}
                              >
                                {other.description}
                              </td>
                              <td className="border p-2 text-center">
                                <button
                                  type="button"
                                  onClick={() => deleteRow('others', index)}
                                  className="has-tooltip icon-btn icon-btn-red !w-5 !h-5"
                                >
                                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                                  <span className="tooltip-text">删除尺寸</span>
                                </button>
                              </td>
                            </tr>
                          )}
                        </Draggable>
                      ))}
                      {provided.placeholder}
                    </tbody>
                  </table>
                )}
              </Droppable>
            </DragDropContext>
          </div>
          <div className="table-actions flex justify-center py-2 bg-surface border-t-0">
            <button type="button" onClick={() => addNewRow('others')} className="btn-gallery-ghost text-xs flex items-center gap-1 py-1 w-full justify-center text-[#d97706] hover:bg-[#fffbeb] border-dashed border">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
              添加新行
            </button>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}