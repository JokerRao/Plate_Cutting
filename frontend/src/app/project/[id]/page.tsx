'use client';

import { useParams, useRouter } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { supabase } from '@/utils/supabaseClient';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import UnsavedChangesPrompt from '@/components/UnsavedChangesPrompt';
import GalleryToast from '@/components/GalleryToast';
import ProjectLayoutNavPills, { ProjectListNavButton } from '@/components/ProjectLayoutNavPills';
import { useAppDialog } from '@/components/AppDialog';
import { getApiUrl, API_CONFIG } from '@/config/api';
import { invalidateLayoutCache } from '@/utils/layoutCache';

/** 用于脏检查：对每个对象的键排序后再序列化，避免 JSON.stringify 键序不一致导致误判「有未保存更改」 */
function stableStringifyForDirtyCheck(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value) ?? 'null';
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableStringifyForDirtyCheck).join(',')}]`;
  }
  const o = value as Record<string, unknown>;
  const keys = Object.keys(o).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringifyForDirtyCheck(o[k])}`).join(',')}}`;
}

/** 与切板算法相关的数据签名（不含板件/零件说明；常用尺寸不含客户与说明） */
function buildCuttingSignature(
  sawBlade: number,
  plates: any[],
  orders: any[],
  others: any[]
): string {
  const n = (x: unknown) => {
    const v = Number(x);
    return Number.isFinite(v) ? v : 0;
  };
  return JSON.stringify({
    saw: n(sawBlade),
    plates: plates.map((row) => ({
      id: row.id,
      length: n(row.length),
      width: n(row.width),
      q: n(row.quantity ?? 1),
    })),
    orders: orders.map((row) => ({
      id: row.id,
      length: n(row.length),
      width: n(row.width),
      q: n(row.quantity ?? 1),
    })),
    others: others.map((row) => ({
      id: row.id,
      length: n(row.length),
      width: n(row.width),
      q: n(row.quantity ?? 0),
    })),
  });
}

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { alert: dialogAlert, confirm: dialogConfirm } = useAppDialog();
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
  const [optimization, setOptimization] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [userEmail, setUserEmail] = useState<string>('');
  const [galleryToast, setGalleryToast] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<'loading' | 'error' | 'ready'>('loading');
  const [loadErrorMessage, setLoadErrorMessage] = useState<string | null>(null);

  const showGalleryToast = useCallback((msg: string) => {
    setGalleryToast(msg);
  }, []);

  useEffect(() => {
    if (!galleryToast) return;
    const id = window.setTimeout(() => setGalleryToast(null), 2400);
    return () => window.clearTimeout(id);
  }, [galleryToast]);

  useEffect(() => {
    const fetchData = async () => {
      if (!projectId) {
        setLoadState('error');
        setLoadErrorMessage('无效的项目链接');
        setProject(null);
        setInitialData(null);
        return;
      }
      setLoadState('loading');
      setLoadErrorMessage(null);
      const { data: { user } } = await supabase.auth.getUser();
      setUserId(user?.id ?? null);
      setUserEmail(user?.email || '未知用户');
      const { data, error } = await supabase.from('Projects').select('*').eq('id', projectId).single();
      if (error || !data) {
        setLoadState('error');
        setLoadErrorMessage(error?.message ?? '未找到该项目或无权访问');
        setProject(null);
        setInitialData(null);
        return;
      }
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
        plates: stableStringifyForDirtyCheck(data.plates || []),
        orders: stableStringifyForDirtyCheck(data.orders || []),
        others: stableStringifyForDirtyCheck(data.others || []),
      });
      setLoadState('ready');
    };
    void fetchData();
  }, [projectId]);

  const hasChanges = useMemo(() => {
    // 未拉取到基准快照前不能视为「有改动」，否则 beforeunload 与返回列表等逻辑会误判
    if (!initialData) return false;
    return (
      projectName !== initialData.name ||
      projectDetails !== initialData.details ||
      projectDescription !== initialData.description ||
      sawBlade !== initialData.saw_blade ||
      stableStringifyForDirtyCheck(plates) !== initialData.plates ||
      stableStringifyForDirtyCheck(orders) !== initialData.orders ||
      stableStringifyForDirtyCheck(others) !== initialData.others
    );
  }, [
    initialData,
    projectName,
    projectDetails,
    projectDescription,
    sawBlade,
    plates,
    orders,
    others,
  ]);

  const projectHasCutResults = useMemo(() => {
    const c = project?.cutted;
    if (!Array.isArray(c) || c.length === 0) return false;
    const last = c[c.length - 1];
    const hasMetadata = last != null && typeof last === 'object' && 'metadata' in (last as object);
    return hasMetadata ? c.length > 1 : c.length > 0;
  }, [project]);

  const savedCuttingSignature = useMemo(() => {
    if (!initialData) return '';
    try {
      const p = JSON.parse(initialData.plates) as any[];
      const o = JSON.parse(initialData.orders) as any[];
      const ot = JSON.parse(initialData.others) as any[];
      return buildCuttingSignature(initialData.saw_blade, p, o, ot);
    } catch {
      return '';
    }
  }, [initialData]);

  const currentCuttingSignature = useMemo(
    () => buildCuttingSignature(sawBlade, plates, orders, others),
    [sawBlade, plates, orders, others]
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

  const validateData = useCallback(async (): Promise<boolean> => {
    const validateArray = (arr: any[]) => {
      return arr.every(item => 
        validatePositiveInteger(item.length.toString()) && 
        validatePositiveInteger(item.width.toString()) && 
        validatePositiveInteger(item.quantity?.toString() || '1')
      );
    };

    /** 常用尺寸无数量列，行内 quantity 常为 0，不得与板件/零件共用带 quantity 的校验 */
    const validateOthersRows = (arr: any[]) => {
      return arr.every((item) =>
        validatePositiveInteger(String(item.length ?? '')) &&
        validatePositiveInteger(String(item.width ?? ''))
      );
    };

    if (!validatePositiveNumber(sawBlade.toString())) {
      await dialogAlert('锯片宽度必须为正数（可包含小数）', '提示');
      return false;
    }

    if (!validateArray(plates)) {
      await dialogAlert('板件信息中的长度、宽度和数量必须为正整数', '提示');
      return false;
    }

    if (!validateArray(orders)) {
      await dialogAlert('零件信息中的长度、宽度和数量必须为正整数', '提示');
      return false;
    }

    if (!validateOthersRows(others)) {
      await dialogAlert('常用尺寸信息中的长度、宽度必须为正整数', '提示');
      return false;
    }

    return true;
  }, [sawBlade, plates, orders, others, dialogAlert]);

  const rollbackFormFromInitial = useCallback(() => {
    if (!initialData) return;
    setProjectName(initialData.name);
    setProjectDetails(initialData.details);
    setProjectDescription(initialData.description);
    setSawBlade(initialData.saw_blade);
    try {
      setPlates(JSON.parse(initialData.plates));
      setOrders(JSON.parse(initialData.orders));
      setOthers(JSON.parse(initialData.others));
    } catch {
      setPlates([]);
      setOrders([]);
      setOthers([]);
    }
  }, [initialData]);

  const runCuttingPipeline = useCallback(
    async (includeNavigation: boolean): Promise<boolean> => {
      if (!userId) {
        await dialogAlert('请先登录', '提示');
        return false;
      }
      if (!(await validateData())) return false;
      if (plates.length === 0 || orders.length === 0) {
        await dialogAlert('请添加板材和零件信息', '提示');
        return false;
      }

      setIsLoading(true);
      try {
        const { error: updateError } = await supabase
          .from('Projects')
          .update({
            name: projectName,
            details: projectDetails,
            description: projectDescription,
            plates,
            orders,
            others,
            saw_blade: sawBlade,
            updated_at: new Date().toISOString(),
          })
          .eq('id', projectId)
          .eq('uid', userId);

        if (updateError) throw updateError;

        const submitResponse = await fetch(getApiUrl('OPTIMIZE_ASYNC'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            uid: userId,
            project_id: projectId,
            plates,
            orders,
            others,
            optimization: Boolean(optimization),
            saw_blade: Number(sawBlade) || 4,
            multistart_runs: 1, // 缺省填 1，如有表单控制可按需传入
          }),
        });

        if (!submitResponse.ok) {
          let errorMsg = '异步任务提交失败';
          try {
            const errData = await submitResponse.json();
            if (errData.detail?.message) errorMsg = errData.detail.message;
            else if (errData.message) errorMsg = errData.message;
          } catch {}
          throw new Error(errorMsg);
        }

        const submitData = await submitResponse.json();
        const jobId = submitData.job_id;
        if (!jobId) {
          throw new Error('未获取到任务ID');
        }

        let runResult = null;
        let jobError = null;

        // 轮询后台状态
        let currentInterval = 1000; // 首次轮询给 1 秒的快速检测窗口
        while (true) {
          // 在轮询前按需等待
          await new Promise(resolve => setTimeout(resolve, currentInterval));
          currentInterval = API_CONFIG.POLLING_INTERVAL_MS; // 之后全部切为配置设定的长间隔
          
          const pollResponse = await fetch(`${getApiUrl('OPTIMIZE_JOB')}/${jobId}`);
          
          if (!pollResponse.ok) {
            throw new Error('轮询查询排版进度失败');
          }
          
          const pollData = await pollResponse.json();
          
          if (pollData.status === 'completed') {
            runResult = pollData.result;
            break;
          } else if (pollData.status === 'failed') {
            jobError = pollData.error || '排版后台发生未知错误';
            break;
          }
          // 如果仍是 pending / running，下一次循环将等待 10s
        }

        if (jobError) {
          throw new Error(jobError);
        }

        if (runResult && runResult.code === 0) {
          const { error: cutError } = await supabase
            .from('Projects')
            .update({
              cutted: runResult.cutting_plans,
              updated_at: new Date().toISOString(),
            })
            .eq('id', projectId)
            .eq('uid', userId);

          if (cutError) throw cutError;

          const now = new Date().toISOString();
          setProject((prev) => ({
            ...prev,
            name: projectName,
            details: projectDetails,
            description: projectDescription,
            saw_blade: sawBlade,
            plates,
            orders,
            others,
            cutted: runResult.cutting_plans,
            updated_at: now,
          }));

          setInitialData({
            name: projectName,
            details: projectDetails,
            description: projectDescription,
            saw_blade: sawBlade,
            plates: stableStringifyForDirtyCheck(plates),
            orders: stableStringifyForDirtyCheck(orders),
            others: stableStringifyForDirtyCheck(others),
          });

          invalidateLayoutCache(projectId);

          if (includeNavigation) {
            await dialogAlert('板件、零件和其他尺寸信息已保存', '保存成功');
            router.push(`/layout/${projectId}/1`);
          } else {
            await dialogAlert('已重新切板并保存', '保存成功');
          }
          return true;
        }
        
        throw new Error(runResult?.message || '切板请求失败');
      } catch (error: unknown) {
        const msg =
          error instanceof Error
            ? error.message
            : typeof error === 'object' &&
                error !== null &&
                'message' in error &&
                typeof (error as { message: string }).message === 'string'
              ? (error as { message: string }).message
              : '切板失败';
        await dialogAlert(msg, '切板失败');
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [
      userId,
      validateData,
      plates,
      orders,
      others,
      sawBlade,
      projectName,
      projectDetails,
      projectDescription,
      projectId,
      optimization,
      router,
      dialogAlert,
    ]
  );

  const handleSave = useCallback(async (): Promise<boolean> => {
    if (!(await validateData())) {
      return false;
    }

    const cuttingDirty =
      projectHasCutResults && currentCuttingSignature !== savedCuttingSignature;

    if (cuttingDirty) {
      const ok = await dialogConfirm(
        '您已修改会影响切板结果的数据（锯片宽度、板件清单、待切零件、常用尺寸等）。\n' +
          '在已有切板方案的情况下，保存前必须重新执行切板，否则无法保存。\n\n' +
          '确定：重新切板并保存全部数据\n取消：放弃本次修改，恢复为上次保存的内容',
        '保存前确认'
      );
      if (!ok) {
        rollbackFormFromInitial();
        return false;
      }
      return runCuttingPipeline(false);
    }

    const { error } = await supabase
      .from('Projects')
      .update({
        name: projectName,
        details: projectDetails,
        description: projectDescription,
        saw_blade: sawBlade,
        plates,
        orders,
        others,
        updated_at: new Date().toISOString(),
      })
      .eq('id', projectId)
      .eq('uid', userId);

    if (error) {
      await dialogAlert('保存失败: ' + error.message, '保存失败');
      return false;
    }

    const now = new Date().toISOString();
    setProject((prev) => ({
      ...prev,
      name: projectName,
      details: projectDetails,
      description: projectDescription,
      saw_blade: sawBlade,
      plates,
      orders,
      others,
      updated_at: now,
    }));
    setInitialData({
      name: projectName,
      details: projectDetails,
      description: projectDescription,
      saw_blade: sawBlade,
      plates: stableStringifyForDirtyCheck(plates),
      orders: stableStringifyForDirtyCheck(orders),
      others: stableStringifyForDirtyCheck(others),
    });
    await dialogAlert('保存成功', '保存成功');
    return true;
  }, [
    validateData,
    dialogConfirm,
    dialogAlert,
    projectHasCutResults,
    currentCuttingSignature,
    savedCuttingSignature,
    rollbackFormFromInitial,
    runCuttingPipeline,
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

  useEffect(() => {
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [handleBeforeUnload]);

  const handleBack = async () => {
    if (hasChanges) {
      if (await dialogConfirm('有未保存的更改，是否保存？', '未保存的更改')) {
        const ok = await handleSave();
        if (!ok) return;
      }
    }
    router.push('/project');
  };

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (!error) {
      router.push('/login');
    } else {
      await dialogAlert('退出登录失败: ' + error.message, '退出失败');
    }
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

  /** 焦点在可编辑单元格内时应交给浏览器处理复制/粘贴，避免整行 JSON 逻辑抢快捷键 */
  const isEventFromEditableCell = (target: EventTarget | null) => {
    const el = target as HTMLElement | null;
    if (!el) return false;
    if (el.isContentEditable) return true;
    return Boolean(el.closest('td[contenteditable="true"]'));
  };

  /** 从剪贴板 JSON 解析为可合并进当前行的字段（拒绝无关 JSON） */
  const parsePastedRowFields = (
    text: string,
    kind: 'plates' | 'orders' | 'others'
  ): Record<string, unknown> | null => {
    const t = text.trim();
    if (!t.startsWith('{')) return null;
    let o: unknown;
    try {
      o = JSON.parse(t);
    } catch {
      return null;
    }
    if (typeof o !== 'object' || o === null || Array.isArray(o)) return null;
    const r = o as Record<string, unknown>;
    const num = (v: unknown) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : NaN;
    };
    if (kind === 'plates' || kind === 'orders') {
      const length = num(r.length);
      const width = num(r.width);
      if (!Number.isFinite(length) || !Number.isFinite(width)) return null;
      const quantity = r.quantity != null ? num(r.quantity) : 1;
      if (!Number.isFinite(quantity)) return null;
      return {
        length,
        width,
        quantity,
        description: typeof r.description === 'string' ? r.description : '',
      };
    }
    const length = num(r.length);
    const width = num(r.width);
    if (!Number.isFinite(length) || !Number.isFinite(width)) return null;
    const quantity = r.quantity != null ? num(r.quantity) : 0;
    if (!Number.isFinite(quantity)) return null;
    return {
      length,
      width,
      quantity,
      client: typeof r.client === 'string' ? r.client : '',
      description: typeof r.description === 'string' ? r.description : '',
    };
  };

  const duplicateRow = (type: 'plates' | 'orders' | 'others', index: number) => {
    const row = type === 'plates' ? plates[index] : type === 'orders' ? orders[index] : others[index];
    if (!row) return;
    const clone = {
      ...row,
      description: row.description ?? '',
      ...(type === 'others' && { client: row.client ?? '', quantity: row.quantity ?? 0 }),
    };
    const insertAt = (prev: any[]) => {
      const next = [...prev];
      next.splice(index + 1, 0, { ...clone });
      return next.map((item, i) => ({ ...item, id: i + 1 }));
    };
    switch (type) {
      case 'plates':
        setPlates(insertAt);
        break;
      case 'orders':
        setOrders(insertAt);
        break;
      case 'others':
        setOthers(insertAt);
        break;
    }
    const tableLabel = type === 'plates' ? '板件清单' : type === 'orders' ? '待切零件' : '常用尺寸/余料';
    showGalleryToast(`已在「${tableLabel}」中插入相同一行`);
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
      if (isEventFromEditableCell(e.target)) return;
      e.preventDefault();
      const { type, index } = selectedRow;
      const data = type === 'plates' ? plates[index] : type === 'orders' ? orders[index] : others[index];
      if (!data) return;
      const { id: _omittedId, ...copyData } = data;
      try {
        await navigator.clipboard.writeText(JSON.stringify(copyData));
        showGalleryToast('已复制整行数据到剪贴板（JSON）');
      } catch (err) {
        console.error('复制失败', err);
        showGalleryToast('复制失败，请检查剪贴板权限或浏览器设置');
      }
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
      if (isEventFromEditableCell(e.target)) return;
      let text: string;
      try {
        text = await navigator.clipboard.readText();
      } catch {
        return;
      }
      const { type, index } = selectedRow;
      const fields = parsePastedRowFields(text, type);
      if (!fields) return;

      e.preventDefault();
      const setValue = (prev: any[]) =>
        prev.map((row, idx) => (idx === index ? { ...row, ...fields, id: row.id } : row));

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

    if (data?.cutted && Array.isArray(data.cutted)) {
      const last = data.cutted[data.cutted.length - 1];
      const hasMetadata = last != null && typeof last === 'object' && 'metadata' in last;
      const plans = hasMetadata ? data.cutted.slice(0, -1) : data.cutted;
      if (plans.length > 0) {
        router.push(`/layout/${projectId}/1`);
        return;
      }
    }
    
    await dialogAlert('请先进行切板操作，或当前暂无有效切板方案', '提示');
  };

  const handleCutting = () => void runCuttingPipeline(true);

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

  if (loadState === 'loading') {
    return (
      <div className="page-gallery flex min-h-screen items-center justify-center text-ink-muted">
        <div className="flex items-center gap-3">
          <svg className="w-5 h-5 animate-spin text-accent" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
          加载中…
        </div>
      </div>
    );
  }

  if (loadState === 'error') {
    return (
      <div className="page-gallery flex min-h-screen flex-col items-center justify-center gap-4 px-4 text-center text-ink-muted">
        <p className="max-w-md text-sm text-ink">{loadErrorMessage ?? '加载失败'}</p>
        <button
          type="button"
          className="btn-gallery-primary inline-flex h-9 items-center px-4 text-xs shadow-sm"
          onClick={() => router.push('/project')}
        >
          返回项目列表
        </button>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="page-gallery flex min-h-screen flex-col items-center justify-center gap-4 px-4 text-center text-ink-muted">
        <p className="text-sm text-ink">项目数据异常</p>
        <button
          type="button"
          className="btn-gallery-primary inline-flex h-9 items-center px-4 text-xs shadow-sm"
          onClick={() => router.push('/project')}
        >
          返回项目列表
        </button>
      </div>
    );
  }

  const editableCellClass =
    "border p-2 focus:outline-none focus:border-[#c5c5c7] focus:ring-0 bg-[#f8fafc] hover:bg-[#f1f5f9] transition-colors";

  return (
    <div className="page-gallery">
      <div className="page-gallery-inner">
      <UnsavedChangesPrompt hasChanges={hasChanges} />
      <GalleryToast message={galleryToast} />

      {/* 顶栏：① 左标题 | 右：当前 + 项目列表 + 用户 ② 左优化+切板+保存 | 右导航 */}
      <div className="pt-3 space-y-6 mb-6 animate-fade-in-up border-b border-hairline pb-4">
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <h1 className="flex min-w-0 items-center gap-2 text-xl font-semibold tracking-tight text-ink sm:gap-3 sm:text-2xl">
            <svg className="h-5 w-5 shrink-0 text-accent sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            <span className="truncate">{projectName || '未命名项目'}</span>
          </h1>
          <div className="relative z-30 flex shrink-0 flex-wrap items-center justify-end gap-2">
            <span
              className="inline-flex h-8 items-center rounded-md border border-hairline bg-muted/40 px-2.5 text-xs font-medium text-ink-muted"
              title="当前所在界面"
            >
              项目配置
              <span className="ml-1.5 rounded bg-surface/90 px-1.5 py-0.5 text-[10px] font-semibold text-ink">当前</span>
            </span>
            <ProjectListNavButton size="toolbar" onClick={() => void handleBack()} />
            <div
              className="has-tooltip has-tooltip-user flex cursor-default items-center gap-0 text-ink-muted min-w-0"
              aria-label={userEmail && userEmail !== '未知用户' ? `当前用户 ${userEmail}，悬停查看详情` : '当前用户，悬停查看详情'}
            >
              <svg className="h-5 w-5 shrink-0 rounded-full bg-muted p-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              <div className="tooltip-text flex min-w-[12rem] max-w-[min(90vw,18rem)] flex-col items-stretch gap-2">
                <div className="text-[10px] font-medium uppercase tracking-wider text-ink-muted">当前用户</div>
                <div className="break-all text-xs leading-snug text-ink">{userEmail || '未登录'}</div>
              </div>
            </div>
            <button type="button" onClick={handleLogout} className="has-tooltip icon-btn icon-btn-red shrink-0" aria-label="退出登录">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              <span className="tooltip-text">退出登录</span>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-2 sm:gap-x-3 mt-4">
          <div className="flex flex-wrap items-center gap-2 sm:gap-2.5">
          <div className="mr-0 sm:mr-1" role="radiogroup" aria-label="切板优化模式">
            <div
              className={`relative flex h-8 min-w-[10.5rem] items-stretch gap-1 rounded-lg border p-0.5 shadow-[inset_0_1px_3px_rgba(0,0,0,0.05)] transition-[background-color,border-color,box-shadow] duration-500 ease-out motion-reduce:duration-200 ${
                optimization === 1
                  ? 'border-sky-300/55 bg-gradient-to-r from-sky-100/95 via-sky-50/80 to-sky-100/60'
                  : 'border-violet-300/55 bg-gradient-to-r from-violet-100/95 via-violet-50/80 to-violet-100/60'
              }`}
            >
              <div
                className={`pointer-events-none absolute inset-y-0.5 left-0.5 w-[calc(50%-4px)] rounded-md bg-white/95 transition-[transform,box-shadow] duration-300 ease-[cubic-bezier(0.34,1.25,0.64,1)] motion-reduce:transition-none motion-reduce:duration-0 ${
                  optimization === 1
                    ? 'shadow-[0_2px_10px_rgba(14,165,233,0.2),0_0_0_1px_rgba(14,165,233,0.22)]'
                    : 'shadow-[0_2px_10px_rgba(139,92,246,0.18),0_0_0_1px_rgba(139,92,246,0.22)]'
                }`}
                style={{
                  transform:
                    optimization === 1 ? 'translateX(0)' : 'translateX(calc(100% + 4px))',
                }}
                aria-hidden
              />
              <label
                className={`has-tooltip relative z-10 flex flex-1 cursor-pointer select-none items-center justify-center gap-1 rounded-md px-1.5 text-[11px] font-medium transition-colors duration-300 ${
                  optimization === 1
                    ? 'text-sky-700'
                    : 'text-ink-muted hover:bg-black/[0.04] hover:text-ink'
                }`}
              >
                <input
                  type="radio"
                  name="optimization"
                  value="1"
                  checked={optimization === 1}
                  onChange={() => setOptimization(1)}
                  className="sr-only"
                />
                <span className="tooltip-text !max-w-[14rem] whitespace-normal !text-left text-[11px] leading-snug">
                  尽量提高板材利用率，搜索更充分，计算时间往往更长。适合对出材率要求高、可接受等待的订单。
                </span>
                <svg className="h-3.5 w-3.5 shrink-0 opacity-90" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>优化模式</span>
              </label>
              <label
                className={`has-tooltip relative z-10 flex flex-1 cursor-pointer select-none items-center justify-center gap-1 rounded-md px-1.5 text-[11px] font-medium transition-colors duration-300 ${
                  optimization === 0
                    ? 'text-violet-800'
                    : 'text-ink-muted hover:bg-black/[0.04] hover:text-ink'
                }`}
              >
                <input
                  type="radio"
                  name="optimization"
                  value="0"
                  checked={optimization === 0}
                  onChange={() => setOptimization(0)}
                  className="sr-only"
                />
                <span className="tooltip-text !max-w-[14rem] whitespace-normal !text-left text-[11px] leading-snug">
                  按常规规则排版，计算更快，适合日常下单与快速试算。系统默认使用本模式。
                </span>
                <svg className="h-3.5 w-3.5 shrink-0 opacity-90" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
                <span>正常模式</span>
              </label>
            </div>
          </div>
          
          <button
            type="button"
            className="btn-gallery-primary inline-flex h-8 items-center gap-1.5 px-3 text-xs shadow-sm"
            onClick={handleCutting}
            disabled={isLoading}
          >
            {isLoading ? (
              <svg className="w-4 h-4 animate-spin text-white" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" /></svg>
            )}
            <span>执行切板</span>
          </button>
          <button
            type="button"
            className="btn-gallery-green inline-flex h-8 items-center gap-1.5 px-3 text-xs shadow-sm"
            onClick={handleSave}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" /></svg>
            <span>保存更改</span>
          </button>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2 sm:gap-2.5">
            <ProjectLayoutNavPills
              projectId={projectId}
              active="config"
              onGoLayout={handleLayoutClick}
              className="mb-0"
              size="toolbar"
              show={{ config: false, projectList: false }}
              suppressPillCurrentLabel
            />
          </div>
        </div>
      </div>

      {/* 项目基本信息 */}
      <div className="mb-6">
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
                      void (async () => {
                        const value = e.currentTarget.textContent || '0';
                        if (validatePositiveNumber(value)) {
                          setSawBlade(parseFloat(value));
                        } else {
                          await dialogAlert('锯片宽度必须为正数（可包含小数）', '提示');
                          e.currentTarget.textContent = sawBlade.toString();
                        }
                      })();
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
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* 板件信息 */}
        <div className="table-container hover-lift shadow-sm animate-fade-in-up flex flex-col h-full lg:col-span-4" style={{ animationDelay: '0.2s', backgroundColor: '#f8fafc' }}>
          <div
            className="table-title flex items-center gap-1.5 text-xs bg-transparent"
            title="行末「复制为新行」可快速插入相同数据。选中行且焦点不在单元格内时，⌘C / ⌘V 以 JSON 整行复制或覆盖粘贴；在单元格内编辑时 ⌘C / ⌘V 为系统复制粘贴。"
          >
            <svg className="w-3.5 h-3.5 text-[#0284c7]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>
            板件清单
            <span className="badge badge-blue ml-auto">{plates.length}</span>
          </div>
          <div className="table-content flex-1 overflow-auto max-h-[400px]">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="border p-2 w-10 text-center">#</th>
                  <th className="border p-2">长</th>
                  <th className="border p-2">宽</th>
                  <th className="border p-2">数量</th>
                  <th className="border p-2">说明</th>
                  <th className="border p-2 w-[4.25rem] text-center text-ink-muted text-[10px] font-medium normal-case tracking-normal">
                    操作
                  </th>
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
                    <td className="border p-1 text-center">
                      <div className="flex items-center justify-center gap-0.5">
                        <button
                          type="button"
                          onClick={(ev) => {
                            ev.stopPropagation();
                            duplicateRow('plates', index);
                          }}
                          className="has-tooltip icon-btn icon-btn-blue !h-5 !w-5 shrink-0"
                        >
                          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-2" />
                          </svg>
                          <span className="tooltip-text">复制为新行</span>
                        </button>
                        <button
                          type="button"
                          onClick={(ev) => {
                            ev.stopPropagation();
                            deleteRow('plates', index);
                          }}
                          className="has-tooltip icon-btn icon-btn-red !h-5 !w-5 shrink-0"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                          <span className="tooltip-text">删除板件</span>
                        </button>
                      </div>
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
        <div className="table-container hover-lift shadow-sm animate-fade-in-up flex flex-col h-full lg:col-span-4" style={{ animationDelay: '0.3s', backgroundColor: '#faf5ff' }}>
          <div
            className="table-title flex items-center gap-1.5 text-xs bg-transparent"
            title="行末「复制为新行」可快速插入相同数据。选中行且焦点不在单元格内时，⌘C / ⌘V 以 JSON 整行复制或覆盖粘贴；在单元格内编辑时 ⌘C / ⌘V 为系统复制粘贴。"
          >
            <svg className="w-3.5 h-3.5 text-[#9333ea]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
            待切零件
            <span className="badge badge-purple ml-auto">{orders.length}</span>
          </div>
          <div className="table-content flex-1 overflow-auto max-h-[400px]">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="border p-2 w-10 text-center">#</th>
                  <th className="border p-2">长</th>
                  <th className="border p-2">宽</th>
                  <th className="border p-2">数量</th>
                  <th className="border p-2">说明</th>
                  <th className="border p-2 w-[4.25rem] text-center text-ink-muted text-[10px] font-medium normal-case tracking-normal">
                    操作
                  </th>
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
                    <td className="border p-1 text-center">
                      <div className="flex items-center justify-center gap-0.5">
                        <button
                          type="button"
                          onClick={(ev) => {
                            ev.stopPropagation();
                            duplicateRow('orders', index);
                          }}
                          className="has-tooltip icon-btn icon-btn-blue !h-5 !w-5 shrink-0"
                        >
                          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-2" />
                          </svg>
                          <span className="tooltip-text">复制为新行</span>
                        </button>
                        <button
                          type="button"
                          onClick={(ev) => {
                            ev.stopPropagation();
                            deleteRow('orders', index);
                          }}
                          className="has-tooltip icon-btn icon-btn-red !h-5 !w-5 shrink-0"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                          <span className="tooltip-text">删除零件</span>
                        </button>
                      </div>
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
        <div className="table-container hover-lift shadow-sm animate-fade-in-up flex flex-col h-full lg:col-span-4" style={{ animationDelay: '0.4s', backgroundColor: '#fffbeb' }}>
          <div
            className="table-title flex items-center gap-1.5 text-xs bg-transparent"
            title="行末「复制为新行」可快速插入相同数据。选中行且焦点不在单元格内时，⌘C / ⌘V 以 JSON 整行复制或覆盖粘贴；在单元格内编辑时 ⌘C / ⌘V 为系统复制粘贴。"
          >
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
                        <th className="border p-2">长</th>
                        <th className="border p-2">宽</th>
                        <th className="border p-2">客户</th>
                        <th className="border p-2">说明</th>
                        <th className="border p-2 w-[4.25rem] text-center text-ink-muted text-[10px] font-medium normal-case tracking-normal">
                          操作
                        </th>
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
                              <td className="border p-1 text-center">
                                <div className="flex items-center justify-center gap-0.5">
                                  <button
                                    type="button"
                                    onClick={(ev) => {
                                      ev.stopPropagation();
                                      duplicateRow('others', index);
                                    }}
                                    className="has-tooltip icon-btn icon-btn-blue !h-5 !w-5 shrink-0"
                                  >
                                    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-2" />
                                    </svg>
                                    <span className="tooltip-text">复制为新行</span>
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(ev) => {
                                      ev.stopPropagation();
                                      deleteRow('others', index);
                                    }}
                                    className="has-tooltip icon-btn icon-btn-red !h-5 !w-5 shrink-0"
                                  >
                                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                                    <span className="tooltip-text">删除尺寸</span>
                                  </button>
                                </div>
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