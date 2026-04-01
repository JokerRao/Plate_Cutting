'use client';

import { useParams, useRouter } from 'next/navigation';
import { useCallback, useEffect, useState, useRef, useMemo } from 'react';
import { supabase } from '@/utils/supabaseClient';
import ProjectLayoutNavPills, {
  ProjectListNavButton,
  IconNavConfig,
  IconNavHomeLayout,
  IconNavOverview,
  IconNavList,
} from '@/components/ProjectLayoutNavPills';
import { groupCutPlansInOrder, getGroupAccent } from '@/utils/cutPlanGroup';

const DIM_LINE = '#94a3b8';
const DIM_TEXT = '#0f172a';
const PLATE_STROKE = 'rgba(15,23,42,0.55)';
const PLATE_FILL = '#f8fafc';
const PART_OUTLINE = 'rgba(51,65,85,0.72)';
const FONT_DIM = '600 10px ui-monospace, "SF Mono", Menlo, monospace';
/** 板件编号：等宽字，与尺寸标注同系工程图风格 */
const FONT_LABEL_MONO = 'ui-monospace, "SF Mono", Menlo, monospace';

const HATCH_LINE_WIDTH = 0.18;
const STROKE_PLATE = 0.72;
const STROKE_PART = 0.48;
const STROKE_DIM = 0.52;

/** 余料：固定一种剖面线（与订单件明显区分） */
const STOCK_HATCH = {
  base: '#fffbeb',
  line: 'rgba(180,83,9,0.42)',
  angles: [0] as const,
  space: 4.5,
};

/** 订单件：高对比度多套路，按编号轮换 */
const ORDER_HATCH_STYLES: ReadonlyArray<{
  base: string;
  line: string;
  angles: readonly number[];
  space: number;
}> = [
  { base: '#bfdbfe', line: 'rgba(29,78,216,0.55)', angles: [45], space: 4.5 },
  { base: '#fbcfe8', line: 'rgba(190,24,93,0.52)', angles: [-45], space: 4.5 },
  { base: '#a7f3d0', line: 'rgba(4,120,87,0.52)', angles: [60], space: 4.5 },
  { base: '#ddd6fe', line: 'rgba(91,33,182,0.52)', angles: [-60], space: 4.5 },
  { base: '#a5f3fc', line: 'rgba(8,145,178,0.52)', angles: [30], space: 4.5 },
  { base: '#fed7aa', line: 'rgba(194,65,12,0.5)', angles: [90], space: 4.5 },
  { base: '#c7d2fe', line: 'rgba(67,56,202,0.5)', angles: [45, -45], space: 6 },
  { base: '#fecaca', line: 'rgba(185,28,28,0.48)', angles: [0, 90], space: 5.5 },
];

function orderHatchIndex(id: unknown): number {
  const n = Number(id);
  if (Number.isFinite(n)) return Math.abs(Math.floor(n)) % ORDER_HATCH_STYLES.length;
  const s = String(id);
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % ORDER_HATCH_STYLES.length;
}

type HatchStyleDef = {
  base: string;
  line: string;
  angles: readonly number[];
  space: number;
};

/** 在矩形内绘制工程图式剖面线（clip 后多方向平行线） */
function fillRectWithCadHatch(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  style: HatchStyleDef
) {
  const minDim = Math.min(w, h);
  if (minDim < 2) return;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();

  ctx.fillStyle = style.base;
  ctx.fillRect(x, y, w, h);

  const spacing = minDim < 22 ? Math.max(2, minDim / 7) : style.space;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const span = Math.hypot(w, h) * 1.25;

  for (const deg of style.angles) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.translate(cx, cy);
    ctx.rotate((deg * Math.PI) / 180);
    ctx.strokeStyle = style.line;
    ctx.lineWidth = HATCH_LINE_WIDTH;
    ctx.lineCap = 'butt';
    ctx.beginPath();
    for (let d = -span; d <= span; d += spacing) {
      ctx.moveTo(d, -span);
      ctx.lineTo(d, span);
    }
    ctx.stroke();
    ctx.restore();
  }

  ctx.restore();
}

/** 工程图风格：水平尺寸线（屏幕坐标，x1<x2，y 在板下方） */
function drawHorizontalOverallDim(
  ctx: CanvasRenderingContext2D,
  x1: number,
  x2: number,
  y: number,
  text: string
) {
  const ext = 8;
  const arr = 5;
  ctx.save();
  ctx.strokeStyle = DIM_LINE;
  ctx.fillStyle = DIM_TEXT;
  ctx.lineWidth = STROKE_DIM;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x1, y - ext);
  ctx.moveTo(x2, y);
  ctx.lineTo(x2, y - ext);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1 + arr, y);
  ctx.lineTo(x2 - arr, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x1 + arr, y - 2.5);
  ctx.lineTo(x1 + arr, y + 2.5);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x2, y);
  ctx.lineTo(x2 - arr, y - 2.5);
  ctx.lineTo(x2 - arr, y + 2.5);
  ctx.closePath();
  ctx.fill();
  ctx.font = FONT_DIM;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = DIM_TEXT;
  ctx.fillText(text, (x1 + x2) / 2, y + 4);
  ctx.restore();
}

/** 工程图风格：竖直尺寸线（x 在板右侧，yTop < yBot 为屏幕坐标） */
function drawVerticalOverallDim(
  ctx: CanvasRenderingContext2D,
  x: number,
  yTop: number,
  yBot: number,
  text: string
) {
  const ext = 8;
  const arr = 5;
  ctx.save();
  ctx.strokeStyle = DIM_LINE;
  ctx.fillStyle = DIM_TEXT;
  ctx.lineWidth = STROKE_DIM;
  ctx.beginPath();
  ctx.moveTo(x, yTop);
  ctx.lineTo(x - ext, yTop);
  ctx.moveTo(x, yBot);
  ctx.lineTo(x - ext, yBot);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, yTop + arr);
  ctx.lineTo(x, yBot - arr);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, yTop);
  ctx.lineTo(x - 2.5, yTop + arr);
  ctx.lineTo(x + 2.5, yTop + arr);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x, yBot);
  ctx.lineTo(x - 2.5, yBot - arr);
  ctx.lineTo(x + 2.5, yBot - arr);
  ctx.closePath();
  ctx.fill();
  /** 宽度标注：在竖向尺寸线右侧、板底之下横向书写，避免压到板内排版 */
  ctx.font = FONT_DIM;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = DIM_TEXT;
  ctx.fillText(text, x + 4, yBot + 4);
  ctx.restore();
}

type LayoutPieceForHit = {
  id: unknown;
  is_stock?: boolean;
  length: number;
  width: number;
  start_x: number;
  start_y: number;
};

type LayoutPieceHit = {
  left: number;
  top: number;
  right: number;
  bottom: number;
  piece: LayoutPieceForHit;
};

function findHitRegion(x: number, y: number, regions: LayoutPieceHit[]): LayoutPieceHit | null {
  for (let i = regions.length - 1; i >= 0; i--) {
    const r = regions[i];
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return r;
  }
  return null;
}

function lookupOrderRow(orders: any[], id: unknown) {
  const n = Number(id);
  return orders.find((o) => Number(o.id) === n);
}

function lookupOtherRow(others: any[], id: unknown) {
  const n = Number(id);
  return others.find((o) => Number(o.id) === n);
}

function formatLayoutPieceTooltip(piece: LayoutPieceForHit, orders: any[], others: any[]): string {
  const lines: string[] = [];
  const dim = `${piece.length} × ${piece.width} mm`;
  if (piece.is_stock) {
    lines.push(`余料 R${piece.id}`);
    lines.push(`尺寸：${dim}`);
    const row = lookupOtherRow(others, piece.id);
    if (row?.client) lines.push(`客户：${row.client}`);
    if (row?.description) lines.push(`说明：${row.description}`);
  } else {
    lines.push(`零件 #${piece.id}`);
    lines.push(`尺寸：${dim}`);
    const row = lookupOrderRow(orders, piece.id);
    if (row?.description) lines.push(`说明：${row.description}`);
  }
  lines.push(`左下角坐标：${piece.start_x}, ${piece.start_y} mm`);
  return lines.join('\n');
}

/**
 * 工程图式编号：角部 + 等宽字 + 浅色描边勾字（无实心底），与尺寸标注同系。
 * 默认左下内角；高度不足时改左上；仍放不下则缩小字号，极小件居中。
 */
function drawPieceLabel(
  ctx: CanvasRenderingContext2D,
  left: number,
  top: number,
  pieceW: number,
  pieceH: number,
  label: string,
  isStock: boolean
) {
  const minDim = Math.min(pieceW, pieceH);
  const pad = Math.max(1.5, Math.min(5, minDim * 0.07));
  const maxW = Math.max(0, pieceW - pad * 2);
  const maxH = Math.max(0, pieceH - pad * 2);

  let fontPx = minDim < 20 ? 8 : minDim < 36 ? 9 : minDim < 64 ? 10 : minDim < 100 ? 11 : 12;
  const fillColor = isStock ? '#92400e' : '#0f172a';
  const halo = 'rgba(248,250,252,0.95)';

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.miterLimit = 2;

  const measureW = (px: number) => {
    ctx.font = `600 ${px}px ${FONT_LABEL_MONO}`;
    return ctx.measureText(label).width;
  };

  while (fontPx > 7 && measureW(fontPx) > maxW) fontPx -= 1;

  const tw = measureW(fontPx);
  const textH = fontPx * 1.2;

  const strokeFillAt = (x: number, y: number, align: CanvasTextAlign, baseline: CanvasTextBaseline) => {
    ctx.font = `600 ${fontPx}px ${FONT_LABEL_MONO}`;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = halo;
    ctx.strokeText(label, x, y);
    ctx.lineWidth = 0.9;
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.strokeText(label, x, y);
    ctx.fillStyle = fillColor;
    ctx.fillText(label, x, y);
  };

  if (maxW < 4 || maxH < 4 || pieceW < 8 || pieceH < 8) {
    ctx.restore();
    return;
  }

  if (textH > maxH) {
    const cx = left + pieceW / 2;
    const cy = top + pieceH / 2;
    strokeFillAt(cx, cy, 'center', 'middle');
    ctx.restore();
    return;
  }

  if (pieceH >= 16) {
    const align: CanvasTextAlign = tw <= maxW ? 'left' : 'right';
    const x = align === 'left' ? left + pad : left + pieceW - pad;
    const y = top + pieceH - pad;
    strokeFillAt(x, y, align, 'bottom');
  } else {
    const align: CanvasTextAlign = tw <= maxW ? 'left' : 'right';
    const x = align === 'left' ? left + pad : left + pieceW - pad;
    const y = top + pad;
    strokeFillAt(x, y, align, 'top');
  }

  ctx.restore();
}

export default function LayoutPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;
  const pageNum = parseInt(params.page as string);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layoutHitsRef = useRef<LayoutPieceHit[]>([]);
  const [layoutHover, setLayoutHover] = useState<{
    clientX: number;
    clientY: number;
    piece: LayoutPieceForHit;
  } | null>(null);
  const [notification, setNotification] = useState<{message: string, type: 'warning'} | null>(null);
  const [projectName, setProjectName] = useState('');
  const [layoutData, setLayoutData] = useState<any>(null);
  const [orders, setOrders] = useState<any[]>([]);
  const [others, setOthers] = useState<any[]>([]);
  const [totalPages, setTotalPages] = useState(0);
  const [allCutted, setAllCutted] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);  // 添加加载状态
  const [isTransitioning, setIsTransitioning] = useState(false);  // 添加过渡状态
  const cutGroups = useMemo(() => groupCutPlansInOrder(allCutted), [allCutted]);

  // 显示通知的函数
  const showNotification = (message: string) => {
    setNotification({ message, type: 'warning' });
    setTimeout(() => {
      setNotification(null);
    }, 5000); // 5秒后消失
  };

  // 预加载下一页数据
  const preloadNextPage = async (nextPageNum: number) => {
    if (nextPageNum > 0 && nextPageNum <= totalPages) {
      const nextPageData = allCutted[nextPageNum - 1];
      if (nextPageData) {
        setLayoutData(nextPageData);
      }
    }
  };

  // 处理页面切换
  const handlePageChange = async (newPageNum: number) => {
    if (newPageNum < 1 || newPageNum > totalPages || newPageNum === pageNum) return;
    
    setIsTransitioning(true);
    setIsLoading(true);
    
    try {
      // 预加载下一页数据
      await preloadNextPage(newPageNum);
      
      // 使用 router.push 进行页面切换
      router.push(`/layout/${projectId}/${newPageNum}`);
    } catch (error) {
      console.error('Error changing page:', error);
      showNotification('页面切换失败，请重试');
    } finally {
      setIsLoading(false);
      setIsTransitioning(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
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
            setOrders(metadata.orders || []);
            setOthers(metadata.others || []);
            const cuttingPlans = hasMetadata ? data.cutted.slice(0, -1) : data.cutted;
            setTotalPages(cuttingPlans.length);
            setAllCutted(cuttingPlans);
            if (cuttingPlans[pageNum - 1]) {
              setLayoutData(cuttingPlans[pageNum - 1]);
            }
          } else {
            setTotalPages(0);
            setAllCutted([]);
            setLayoutData(null);
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        showNotification('数据加载失败，请刷新页面重试');
      } finally {
        setIsLoading(false);
      }
    };
    
    if (projectId) fetchData();
  }, [projectId, pageNum]);

  useEffect(() => {
    if (isLoading) return;
    const badIndex = Number.isNaN(pageNum) || pageNum < 1;
    if (totalPages > 0 && (badIndex || pageNum > totalPages)) {
      router.replace(`/layout/${projectId}/1`);
    }
  }, [isLoading, totalPages, pageNum, projectId, router]);

  useEffect(() => {
    setLayoutHover(null);
  }, [layoutData]);

  useEffect(() => {
    if (!layoutData || !canvasRef.current) {
      layoutHitsRef.current = [];
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const plate = layoutData.plate;
    if (!Array.isArray(plate) || plate.length < 2) return;
    const plateLength = Number(plate[0]);
    const plateWidth = Number(plate[1]);
    if (!Number.isFinite(plateLength) || !Number.isFinite(plateWidth) || plateLength <= 0 || plateWidth <= 0) {
      return;
    }
    /** 内边距 + 右侧竖向尺寸与文字占位；基准尺寸 ×1.5 相对早期 880 宽度版 */
    const margin = { top: 15, right: 18, bottom: 15, left: 15 };
    const DIM_RIGHT_RESERVE = 81;
    const baseWidth = 1320;
    const innerW = baseWidth - margin.left - margin.right - DIM_RIGHT_RESERVE;
    /** 极瘦长板限制高度，避免单页过长 */
    const MAX_PH = 1980;
    const scale = Math.min(innerW / plateLength, MAX_PH / plateWidth);
    const pw = plateLength * scale;
    const ph = plateWidth * scale;
    /** 动态高度：按板高 + 下方水平尺寸线/文字，去掉固定 840 带来的上下大块留白 */
    const belowPlate = 54;
    canvas.width = baseWidth;
    canvas.height = Math.ceil(margin.top + ph + belowPlate + margin.bottom);
    const ox = margin.left;
    const oy = margin.top;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hits: LayoutPieceHit[] = [];

    ctx.fillStyle = PLATE_FILL;
    ctx.fillRect(ox, oy, pw, ph);

    if (layoutData.cutted && Array.isArray(layoutData.cutted)) {
      layoutData.cutted.forEach((piece: any) => {
        const { start_x, start_y, length, width, is_stock, id } = piece;
        const left = ox + start_x * scale;
        const top = oy + ph - (start_y + width) * scale;
        const w = length * scale;
        const h = width * scale;

        const hatch = is_stock ? STOCK_HATCH : ORDER_HATCH_STYLES[orderHatchIndex(id)];
        fillRectWithCadHatch(ctx, left, top, w, h, hatch);

        ctx.strokeStyle = PART_OUTLINE;
        ctx.lineWidth = STROKE_PART;
        ctx.strokeRect(left, top, w, h);

        const label = is_stock ? `R${id}` : String(id);
        drawPieceLabel(ctx, left, top, w, h, label, Boolean(is_stock));

        hits.push({
          left,
          top,
          right: left + w,
          bottom: top + h,
          piece: {
            id,
            is_stock,
            length: Number(length),
            width: Number(width),
            start_x: Number(start_x),
            start_y: Number(start_y),
          },
        });
      });
    }

    layoutHitsRef.current = hits;

    ctx.strokeStyle = PLATE_STROKE;
    ctx.lineWidth = STROKE_PLATE;
    ctx.strokeRect(ox, oy, pw, ph);

    /** 水平总长在宽度标注文字之下，与画布 ×1.5 比例一致 */
    const dimY = oy + ph + 24;
    drawHorizontalOverallDim(ctx, ox, ox + pw, dimY, `${plateLength} mm`);

    const dimX = ox + pw + 15;
    drawVerticalOverallDim(ctx, dimX, oy, oy + ph, `${plateWidth} mm`);
  }, [layoutData]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const sx = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const sy = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const hit = findHitRegion(sx, sy, layoutHitsRef.current);
    if (hit) {
      setLayoutHover({ clientX: e.clientX, clientY: e.clientY, piece: hit.piece });
    } else {
      setLayoutHover(null);
    }
  }, []);

  const handleCanvasLeave = useCallback(() => setLayoutHover(null), []);

  // 在客户端添加动画样式
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeOut {
        0% { opacity: 1; transform: translateY(0); }
        70% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-100%); }
      }
      .animate-fade-out {
        animation: fadeOut 5s forwards;
      }
    `;
    document.head.appendChild(style);

    return () => {
      style.remove();
    };
  }, []);

  const renderTable = (title: string, data: any[], type: 'orders' | 'others') => {
    // 计算当前页面中每个ID的使用数量
    const pageUsageCount = new Map<string, number>();
    if (layoutData && layoutData.cutted) {
      layoutData.cutted.forEach((piece: any) => {
        const id = String(piece.id);
        if ((type === 'orders' && !piece.is_stock) || (type === 'others' && piece.is_stock)) {
          pageUsageCount.set(id, (pageUsageCount.get(id) || 0) + 1);
        }
      });
    }

    // 计算所有页面中的使用数量
    const totalUsageCount = new Map<string, number>();
    if (allCutted) {
      allCutted.forEach(page => {
        if (page.cutted) {
          page.cutted.forEach((piece: any) => {
            const id = String(piece.id);
            if ((type === 'orders' && !piece.is_stock) || (type === 'others' && piece.is_stock)) {
              totalUsageCount.set(id, (totalUsageCount.get(id) || 0) + 1);
            }
          });
        }
      });
    }

    // 过滤数据，只显示在当前页面有使用的零件
    const filteredData = data.filter((item) => {
      const pageCount = pageUsageCount.get(String(item.id)) || 0;
      return pageCount > 0;
    });

    if (filteredData.length === 0) return null;

    return (
      <div className={`table-container hover-lift shadow-sm animate-fade-in-up h-full flex flex-col ${type === 'orders' ? 'bg-[#faf5ff]' : 'bg-[#fffbeb]'}`} style={{ animationDelay: '0.05s' }}>
        <div className="table-title flex items-center gap-2 bg-transparent">
          {type === 'orders' ? (
            <svg className="w-4 h-4 text-[#9333ea]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
          ) : (
            <svg className="w-4 h-4 text-[#d97706]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>
          )}
          {title}
        </div>
        <div className="table-content flex-1 overflow-auto max-h-[400px]">
          <table className="w-full">
            <thead>
              <tr>
                <th className="border p-2">编号</th>
                <th className="border p-2">长度</th>
                <th className="border p-2">宽度</th>
                <th className="border p-2">总数量</th>
                <th className="border p-2">本页数量</th>
                {type === 'others' && <th className="border p-2">客户</th>}
                <th className="border p-2">描述</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((item) => (
                <tr
                  key={item.id}
                  className={
                    type === 'orders'
                      ? 'border-l-2 border-l-accent bg-muted/60'
                      : 'border-l-2 border-l-accent-green bg-muted/60'
                  }
                >
                  <td className="border p-2">{type === 'others' ? `R${item.id}` : item.id}</td>
                  <td className="border p-2">{item.length}</td>
                  <td className="border p-2">{item.width}</td>
                  <td className="border p-2">{type === 'others' ? totalUsageCount.get(String(item.id)) || 0 : item.quantity}</td>
                  <td className="border p-2">{pageUsageCount.get(String(item.id)) || 0}</td>
                  {type === 'others' && <td className="border p-2">{item.client}</td>}
                  <td className="border p-2">{item.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  if (isLoading && !layoutData) return (
    <div className="page-gallery layout-detail-page flex h-screen items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-hairline border-t-accent" />
    </div>
  );

  if (!isLoading && totalPages === 0) {
    return (
      <div className="page-gallery layout-detail-page">
        <div className="page-gallery-inner page-gallery-inner--layout-detail">
          <div className="mb-4 space-y-3 border-b border-hairline pb-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h1 className="flex min-w-0 items-center gap-2 text-xl font-semibold tracking-tight text-ink">
                <span className="truncate">{projectName || '未命名项目'}</span>
              </h1>
              <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
                <span
                  className="inline-flex h-8 items-center rounded-md border border-hairline bg-muted/40 px-2.5 text-xs font-medium text-ink-muted"
                  title="当前所在界面"
                >
                  切板排版
                  <span className="ml-1.5 rounded bg-surface/90 px-1.5 py-0.5 text-[10px] font-semibold text-ink">当前</span>
                </span>
                <ProjectListNavButton size="toolbar" />
              </div>
            </div>
            <div className="flex flex-wrap justify-end">
              <ProjectLayoutNavPills
                projectId={projectId}
                active="layout-detail"
                layoutPageNum={pageNum}
                className="mb-0"
                size="toolbar"
                show={{
                  projectList: false,
                  ...(pageNum === 1 ? { homeLayout: false } : {}),
                }}
                suppressPillCurrentLabel
              />
            </div>
          </div>
          <div className="mx-auto max-w-md rounded-lg border border-hairline bg-surface p-8 text-center shadow-sm">
            <p className="text-lg font-semibold text-ink">暂无切板方案</p>
            <p className="mt-2 text-sm text-ink-muted">
              请先在「项目配置」中执行切板，或从方案总览查看是否已有数据。
            </p>
            <div className="mt-6 flex flex-col flex-wrap items-stretch gap-2 sm:flex-row sm:items-center sm:justify-center">
              <button
                type="button"
                className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-[#3d9eff] bg-[#3d9eff] px-3 text-xs font-medium leading-none text-white shadow-[0_2px_10px_rgba(0,122,255,0.28)] transition-[filter,transform] duration-200 hover:brightness-110 active:scale-[0.99] focus:outline-none focus-visible:ring-1 focus-visible:ring-[#3d9eff] focus-visible:ring-offset-1"
                onClick={() => router.push(`/project/${projectId}`)}
              >
                <IconNavConfig className="h-3.5 w-3.5 shrink-0" />
                项目配置
              </button>
              <button
                type="button"
                className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-teal-500 bg-teal-500 px-3 text-xs font-medium leading-none text-white shadow-[0_2px_10px_rgba(13,148,136,0.26)] transition-[filter,transform] duration-200 hover:brightness-110 active:scale-[0.99] focus:outline-none focus-visible:ring-1 focus-visible:ring-teal-500 focus-visible:ring-offset-1"
                onClick={() => router.push(`/layout/${projectId}/1`)}
              >
                <IconNavHomeLayout className="h-3.5 w-3.5 shrink-0" />
                首页排版
              </button>
              <button
                type="button"
                className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-violet-500 bg-violet-500 px-3 text-xs font-medium leading-none text-white shadow-[0_2px_10px_rgba(109,40,217,0.26)] transition-[filter,transform] duration-200 hover:brightness-110 active:scale-[0.99] focus:outline-none focus-visible:ring-1 focus-visible:ring-violet-500 focus-visible:ring-offset-1"
                onClick={() => router.push(`/layout/${projectId}`)}
              >
                <IconNavOverview className="h-3.5 w-3.5 shrink-0" />
                方案总览
              </button>
              <button
                type="button"
                className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-hairline bg-surface px-3 text-xs font-medium leading-none text-ink shadow-sm transition-[filter,transform] duration-200 hover:border-[#c5c5c7] hover:bg-muted focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-1"
                onClick={() => router.push('/project')}
              >
                <IconNavList className="h-3.5 w-3.5 shrink-0" />
                项目列表
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-gallery layout-detail-page">
      <div className="page-gallery-inner page-gallery-inner--layout-detail">
      {/* 通知 */}
      {notification && (
        <div className="mb-6 animate-fade-out">
          <div className="border border-hairline border-l-2 border-l-accent bg-muted px-4 py-3" style={{ borderRadius: 2 }}>
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 shrink-0 text-accent" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <p className="text-sm text-ink">{notification.message}</p>
            </div>
          </div>
        </div>
      )}

      {/* ① 左标题 | 右：当前页 + 项目列表 ② 左：使用率+页码 | 右：上下页 + 四联导航 */}
      <div className="mb-4 space-y-3 border-b border-hairline pb-3 animate-fade-in-up">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="flex min-w-0 items-center gap-2 text-xl font-semibold tracking-tight text-ink sm:gap-3 sm:text-2xl">
            <svg className="h-5 w-5 shrink-0 text-accent sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
            </svg>
            <span className="truncate">{projectName || '未命名项目'}</span>
          </h1>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            <span
              className="inline-flex h-8 items-center rounded-md border border-hairline bg-muted/40 px-2.5 text-xs font-medium text-ink-muted"
              title="当前所在界面"
            >
              {pageNum === 1 ? '首页排版' : `第 ${pageNum} 张排版`}
              <span className="ml-1.5 rounded bg-surface/90 px-1.5 py-0.5 text-[10px] font-semibold text-ink">当前</span>
            </span>
            <ProjectListNavButton size="toolbar" />
          </div>
        </div>
        <div className="flex min-h-8 flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <div className="flex flex-wrap items-center gap-2">
            {layoutData ? (
              <span
                className="inline-flex h-8 shrink-0 items-center rounded-md border border-[rgba(0,122,255,0.22)] bg-[rgba(0,122,255,0.08)] px-2.5 text-xs font-medium tabular-nums text-[var(--accent)]"
                title="当前页板材利用率"
              >
                使用率 {(layoutData.rate * 100).toFixed(1)}%
              </span>
            ) : null}
            <span
              className="inline-flex h-8 items-center rounded-md border border-hairline bg-surface px-2.5 text-xs font-medium tabular-nums text-ink"
              title="当前张数 / 总张数"
            >
              第 {pageNum} / {totalPages} 张
            </span>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <button
              type="button"
              title="上一页"
              aria-label="上一页"
              className="btn-gallery-primary inline-flex h-8 min-w-9 shrink-0 items-center justify-center gap-1.5 px-2.5 text-xs shadow-sm"
              onClick={() => handlePageChange(pageNum - 1)}
              disabled={pageNum <= 1 || isTransitioning}
            >
              {isTransitioning ? (
                <svg className="h-3.5 w-3.5 animate-spin text-white" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : (
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                </svg>
              )}
            </button>
            <button
              type="button"
              title="下一页"
              aria-label="下一页"
              className="btn-gallery-primary inline-flex h-8 min-w-9 shrink-0 items-center justify-center gap-1.5 px-2.5 text-xs shadow-sm"
              onClick={() => handlePageChange(pageNum + 1)}
              disabled={pageNum >= totalPages || isTransitioning}
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <div className="hidden h-4 w-px bg-border-hairline sm:block" aria-hidden />
            <ProjectLayoutNavPills
              projectId={projectId}
              active="layout-detail"
              layoutPageNum={pageNum}
              className="mb-0"
              size="toolbar"
              show={{
                projectList: false,
                ...(pageNum === 1 ? { homeLayout: false } : {}),
              }}
              suppressPillCurrentLabel
            />
          </div>
        </div>
      </div>

      {/* 零件与常用尺寸列表（在排版图之上） */}
      <div className="mb-4 grid grid-cols-1 gap-4 lg:grid-cols-2 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
        {renderTable('零件信息', orders, 'orders')}
        {renderTable('常用尺寸信息', others, 'others')}
      </div>

      {/* 排版图：上下分隔线、无描边盒子；画布逻辑尺寸 ×1.5；悬停板件显示信息 */}
      <div
        className="relative mx-auto mb-4 w-full min-w-0 max-w-[1320px] border-t border-b border-hairline py-5 animate-fade-in-up"
        style={{ animationDelay: '0.1s' }}
      >
        <canvas
          ref={canvasRef}
          className={`block h-auto w-full max-w-full bg-transparent ${isTransitioning ? 'opacity-50' : ''} ${layoutHover ? 'cursor-help' : 'cursor-default'}`}
          aria-label="切板排版图"
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={handleCanvasLeave}
        />
        {layoutHover && (
          <div
            role="tooltip"
            className="pointer-events-none fixed z-[90] max-w-[min(18rem,calc(100vw-1.5rem))] rounded-lg border border-hairline bg-surface/98 px-3 py-2 text-xs leading-relaxed text-ink shadow-[0_8px_30px_rgba(0,0,0,0.12)]"
            style={{
              left: Math.min(
                layoutHover.clientX + 12,
                typeof window !== 'undefined' ? window.innerWidth - 280 : layoutHover.clientX + 12
              ),
              top: layoutHover.clientY + 12,
            }}
          >
            <div className="whitespace-pre-wrap font-sans">
              {formatLayoutPieceTooltip(layoutHover.piece, orders, others)}
            </div>
          </div>
        )}
      </div>

      {/* 各板排版详情（分页芯片） */}
      {allCutted.length > 0 && (
        <section className="pt-5 pb-2 animate-fade-in-up" style={{ animationDelay: '0.12s' }}>
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold tracking-tight text-ink">
            <svg className="h-4 w-4 text-accent shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
            各板排版详情
          </h2>
          <div className="flex flex-wrap items-center gap-2">
            {cutGroups.map((g, gi) => {
              const page = g.representative;
              const [L, W] = page.plate || [0, 0];
              const r = page.rate != null ? (page.rate * 100).toFixed(1) : '—';
              const multi = g.indices.length > 1;
              const active = g.indices.includes(pageNum - 1);
              const ac = getGroupAccent(gi);

              const pages1 = g.indices.map((i) => i + 1).sort((a, b) => a - b);
              const rangeText =
                pages1.length === 1
                  ? `第 ${pages1[0]} 张`
                  : pages1[pages1.length - 1] - pages1[0] === pages1.length - 1
                    ? `第 ${pages1[0]}–${pages1[pages1.length - 1]} 张`
                    : `共 ${pages1.length} 张`;

              const label = multi
                ? `${rangeText}（同切）· ${L}×${W} mm · ${r}%`
                : `${rangeText} · ${L}×${W} mm · ${r}%`;

              const baseSingle = active
                ? 'border-[var(--accent)] bg-[rgba(0,122,255,0.12)] text-[var(--accent)] shadow-sm ring-1 ring-[var(--accent)]/30'
                : 'border-hairline bg-surface text-ink-muted hover:border-[#c5c5c7] hover:text-ink';

              const baseMulti = active
                ? `border-2 ${ac.border} ${ac.bg} ${ac.text} shadow-md ring-2 ring-offset-1 ${ac.ring}`
                : `border-2 ${ac.border} ${ac.bg} ${ac.text} hover:brightness-[0.98]`;

              return (
                <button
                  key={g.signature}
                  type="button"
                  disabled={isTransitioning}
                  title={multi ? `同切排版，进入该组第 ${pages1[0]} 张` : undefined}
                  onClick={() => void handlePageChange(pages1[0])}
                  className={`rounded-full px-3 py-1.5 text-xs font-medium tabular-nums transition-all ${
                    multi ? baseMulti : baseSingle
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </section>
      )}

      </div>
    </div>
  );
} 