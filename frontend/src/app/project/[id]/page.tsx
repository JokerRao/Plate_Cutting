'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { supabase } from '@/utils/supabaseClient';

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

  const validateNumber = (value: string): boolean => {
    const num = Number(value);
    return Number.isInteger(num) && num >= 0;
  };

  const hasUnsavedChanges = () =>
    projectName !== initialData?.name ||
    projectDetails !== initialData?.details ||
    projectDescription !== initialData?.description ||
    sawBlade !== initialData?.saw_blade ||
    JSON.stringify(plates) !== initialData?.plates ||
    JSON.stringify(orders) !== initialData?.orders ||
    JSON.stringify(others) !== initialData?.others;

  const handleSave = async () => {
    // 验证所有数值是否为整数
    const validateData = () => {
      const validateArray = (arr: any[]) => {
        return arr.every(item => 
          validateNumber(item.length.toString()) && 
          validateNumber(item.width.toString()) && 
          validateNumber(item.quantity?.toString() || '1')
        );
      };

      if (!validateNumber(sawBlade.toString())) {
        alert('锯片宽度必须为非负整数');
        return false;
      }

      if (!validateArray(plates)) {
        alert('板件信息中的长度、宽度和数量必须为非负整数');
        return false;
      }

      if (!validateArray(orders)) {
        alert('零件信息中的长度、宽度和数量必须为非负整数');
        return false;
      }

      if (!validateArray(others)) {
        alert('常用尺寸信息中的长度、宽度必须为非负整数');
        return false;
      }

      return true;
    };

    if (!validateData()) {
      return;
    }

    const { error, data } = await supabase.from('Projects').update({
      name: projectName,
      details: projectDetails,
      description: projectDescription,
      saw_blade: sawBlade,
      plates,
      orders,
      others,
    }).eq('id', projectId).eq('uid', userId);

    if (error) {
      alert('保存失败: ' + error.message);
    } else {
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
  };

  const handleBack = async () => {
    if (hasUnsavedChanges()) {
      if (window.confirm('是否保存数据？')) {
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
    
    // Remove highlight from previously selected row
    if (selectedRow) {
      selectedRow.cells.forEach(cell => {
        cell.style.backgroundColor = '';
      });
    }

    // Highlight new selected row
    editableCells.forEach(cell => {
      cell.style.backgroundColor = '#e5e7eb';
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

    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
      e.preventDefault();
      const { type, index } = selectedRow;
      const data = type === 'plates' ? plates[index] :
                   type === 'orders' ? orders[index] :
                   others[index];
      const { id, ...copyData } = data;
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
    const newRow = { id: 0, width: 0, length: 0, quantity: 1, description: '', client: '' };
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

  if (!project) return <div>加载中...</div>;

  const windowClass = "rounded-lg shadow-lg border bg-white flex flex-col h-full";
  const windowTitleClass = "bg-blue-600 text-white px-4 py-2 rounded-t-lg font-bold text-lg";
  const cellClass = "border p-2 focus:outline-none focus:bg-blue-50";

  return (
    <div className="relative max-w-7xl mx-auto my-8 rounded-2xl shadow-2xl border bg-white flex flex-col h-[92vh]">
      <div className="flex items-center px-6 pt-4">
        <div className="flex gap-2 mb-4">
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold">
            项目
          </button>
          <button 
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold"
            onClick={handleLayoutClick}
          >
            排版
          </button>
        </div>
      </div>
      
      <div className="px-6 pb-2 border-b">
        <h1 className="text-xl font-bold">
          {projectName || '未命名项目'}
        </h1>
      </div>

      <div className="absolute top-4 right-4 flex gap-2">
        <button className="bg-yellow-500 text-white px-3 py-1 rounded">切板</button>
        <button className="bg-green-500 text-white px-3 py-1 rounded" onClick={handleSave}>保存</button>
        <button className="bg-gray-500 text-white px-3 py-1 rounded" onClick={handleBack}>返回</button>
      </div>

      <div className="flex-1 min-h-0 flex flex-col p-6">
        <div className={`${windowClass} mb-4`}>
          <div className={windowTitleClass}>项目列表</div>
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
              <tr className="hover:bg-gray-50">
                <td 
                  className="border p-2"
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
                  className="border p-2"
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
                  className="border p-2"
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
                  className="border p-2"
                  contentEditable
                  suppressContentEditableWarning
                  onBlur={(e) => {
                    const value = e.currentTarget.textContent || '0';
                    if (validateNumber(value)) {
                      setSawBlade(parseInt(value));
                    } else {
                      alert('锯片宽度必须为非负整数');
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
                <td className="border p-2">
                  {new Date(project.updated_at).toLocaleString()}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="flex-1 grid grid-cols-3 gap-4">
          <div className={windowClass}>
            <div className={windowTitleClass}>板件信息</div>
            <div className="flex-1 overflow-auto p-4">
              <table className="min-w-full">
                <thead>
                  <tr>
                    <th className="border p-2">编号</th>
                    <th className="border p-2">长度</th>
                    <th className="border p-2">宽度</th>
                    <th className="border p-2">数量</th>
                    <th className="border p-2">描述</th>
                    <th className="border p-2">操作</th>
                  </tr>
                </thead>
                <tbody onKeyDown={handleKeyDown}>
                  {plates.map((plate, index) => (
                    <tr 
                      key={index} 
                      className="hover:bg-gray-50"
                      onClick={(e) => handleRowClick('plates', index, e)}
                    >
                      <td className="border p-2">{index + 1}</td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '0';
                          if (validateNumber(value)) {
                            handleCellChange('plates', index, 'length', parseInt(value));
                          } else {
                            alert('长度必须为非负整数');
                            e.currentTarget.textContent = plate.length.toString();
                          }
                        }}
                      >
                        {plate.length}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '0';
                          if (validateNumber(value)) {
                            handleCellChange('plates', index, 'width', parseInt(value));
                          } else {
                            alert('宽度必须为非负整数');
                            e.currentTarget.textContent = plate.width.toString();
                          }
                        }}
                      >
                        {plate.width}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '1';
                          if (validateNumber(value)) {
                            handleCellChange('plates', index, 'quantity', parseInt(value));
                          } else {
                            alert('数量必须为非负整数');
                            e.currentTarget.textContent = plate.quantity.toString();
                          }
                        }}
                      >
                        {plate.quantity}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => handleCellChange('plates', index, 'description', e.currentTarget.textContent || '')}
                      >
                        {plate.description}
                      </td>
                      <td className="border p-2">
                        <button
                          onClick={() => deleteRow('plates', index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button onClick={() => addNewRow('plates')} className="mt-2 bg-blue-500 text-white px-3 py-1 rounded">
                添加
              </button>
            </div>
          </div>

          <div className={windowClass}>
            <div className={windowTitleClass}>零件信息</div>
            <div className="flex-1 overflow-auto p-4">
              <table className="min-w-full">
                <thead>
                  <tr>
                    <th className="border p-2">编号</th>
                    <th className="border p-2">长度</th>
                    <th className="border p-2">宽度</th>
                    <th className="border p-2">数量</th>
                    <th className="border p-2">描述</th>
                    <th className="border p-2">操作</th>
                  </tr>
                </thead>
                <tbody onKeyDown={handleKeyDown}>
                  {orders.map((order, index) => (
                    <tr 
                      key={index} 
                      className="hover:bg-gray-50"
                      onClick={(e) => handleRowClick('orders', index, e)}
                    >
                      <td className="border p-2">{index + 1}</td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '0';
                          if (validateNumber(value)) {
                            handleCellChange('orders', index, 'length', parseInt(value));
                          } else {
                            alert('长度必须为非负整数');
                            e.currentTarget.textContent = order.length.toString();
                          }
                        }}
                      >
                        {order.length}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '0';
                          if (validateNumber(value)) {
                            handleCellChange('orders', index, 'width', parseInt(value));
                          } else {
                            alert('宽度必须为非负整数');
                            e.currentTarget.textContent = order.width.toString();
                          }
                        }}
                      >
                        {order.width}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '1';
                          if (validateNumber(value)) {
                            handleCellChange('orders', index, 'quantity', parseInt(value));
                          } else {
                            alert('数量必须为非负整数');
                            e.currentTarget.textContent = order.quantity.toString();
                          }
                        }}
                      >
                        {order.quantity}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => handleCellChange('orders', index, 'description', e.currentTarget.textContent || '')}
                      >
                        {order.description}
                      </td>
                      <td className="border p-2">
                        <button
                          onClick={() => deleteRow('orders', index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button onClick={() => addNewRow('orders')} className="mt-2 bg-blue-500 text-white px-3 py-1 rounded">
                添加
              </button>
            </div>
          </div>

          <div className={windowClass}>
            <div className={windowTitleClass}>常用尺寸信息</div>
            <div className="flex-1 overflow-auto p-4">
              <table className="min-w-full">
                <thead>
                  <tr>
                    <th className="border p-2">编号</th>
                    <th className="border p-2">长度</th>
                    <th className="border p-2">宽度</th>
                    <th className="border p-2">客户</th>
                    <th className="border p-2">描述</th>
                    <th className="border p-2">操作</th>
                  </tr>
                </thead>
                <tbody onKeyDown={handleKeyDown}>
                  {others.map((other, index) => (
                    <tr 
                      key={index} 
                      className="hover:bg-gray-50"
                      onClick={(e) => handleRowClick('others', index, e)}
                    >
                      <td className="border p-2">{index + 1}</td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '0';
                          if (validateNumber(value)) {
                            handleCellChange('others', index, 'length', parseInt(value));
                          } else {
                            alert('长度必须为非负整数');
                            e.currentTarget.textContent = other.length.toString();
                          }
                        }}
                      >
                        {other.length}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => {
                          const value = e.currentTarget.textContent || '0';
                          if (validateNumber(value)) {
                            handleCellChange('others', index, 'width', parseInt(value));
                          } else {
                            alert('宽度必须为非负整数');
                            e.currentTarget.textContent = other.width.toString();
                          }
                        }}
                      >
                        {other.width}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => handleCellChange('others', index, 'client', e.currentTarget.textContent || '')}
                      >
                        {other.client}
                      </td>
                      <td 
                        className="border p-2"
                        contentEditable
                        suppressContentEditableWarning
                        onBlur={(e) => handleCellChange('others', index, 'description', e.currentTarget.textContent || '')}
                      >
                        {other.description}
                      </td>
                      <td className="border p-2">
                        <button
                          onClick={() => deleteRow('others', index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button onClick={() => addNewRow('others')} className="mt-2 bg-blue-500 text-white px-3 py-1 rounded">
                添加
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
