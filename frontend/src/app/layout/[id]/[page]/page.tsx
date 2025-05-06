'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { supabase } from '@/utils/supabaseClient';

interface CuttedItem {
  start_x: number;
  start_y: number;
  length: number;
  width: number;
  type: 0 | 1;  // 0: 零件, 1: 常用尺寸
  id: string;
}

export default function LayoutPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;
  const pageNum = parseInt(params.page as string);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [projectName, setProjectName] = useState('');
  const [layoutData, setLayoutData] = useState<any>(null);
  const [orders, setOrders] = useState<any[]>([]);
  const [others, setOthers] = useState<any[]>([]);
  const [totalPages, setTotalPages] = useState(0);
  const [allCutted, setAllCutted] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      const { data } = await supabase
        .from('Projects')
        .select('name, cutted, orders, others')
        .eq('id', projectId)
        .single();
      
      if (data && data.cutted) {
        setProjectName(data.name);
        setTotalPages(data.cutted.length);
        setAllCutted(data.cutted);
        if (data.cutted[pageNum - 1]) {
          setLayoutData(data.cutted[pageNum - 1]);
          setOrders(data.orders || []);
          setOthers(data.others || []);
        }
      }
    };
    
    if (projectId) fetchData();
  }, [projectId, pageNum]);

  useEffect(() => {
    if (!layoutData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置画布大小
    canvas.width = 800;
    canvas.height = 600;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 计算缩放比例
    const [plateLength, plateWidth] = layoutData.plate;
    const margin = 60; // 留出空间显示标注
    const scale = Math.min(
      (canvas.width - margin * 2) / plateLength,
      (canvas.height - margin * 2) / plateWidth
    );

    // 坐标系变换：原点移到左下角，Y轴向上
    ctx.translate(margin, canvas.height - margin);
    ctx.scale(1, -1);

    // 绘制坐标轴
    ctx.beginPath();
    ctx.strokeStyle = '#000';
    ctx.moveTo(0, 0);
    ctx.lineTo(40, 0);
    ctx.moveTo(0, 0);
    ctx.lineTo(0, 40);
    ctx.stroke();

    // 标注原点
    ctx.scale(1, -1);
    ctx.font = '14px Arial';
    ctx.fillStyle = '#000';
    ctx.fillText('O(0,0)', -20, 20);
    ctx.scale(1, -1);

    // 绘制大板
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, plateLength * scale, plateWidth * scale);
    ctx.strokeStyle = '#000';
    ctx.strokeRect(0, 0, plateLength * scale, plateWidth * scale);

    // 标注大板尺寸
    ctx.scale(1, -1);
    ctx.fillStyle = '#000';
    ctx.font = '14px Arial';
    ctx.fillText(`${plateLength}mm`, plateLength * scale / 2, 20);
    ctx.save();
    ctx.translate(plateLength * scale + 20, -plateWidth * scale / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillText(`${plateWidth}mm`, 0, 0);
    ctx.restore();
    ctx.scale(1, -1);

    // 绘制切割的板件
    if (layoutData.cutted && Array.isArray(layoutData.cutted)) {
      layoutData.cutted.forEach((piece: number[]) => {
        const [start_x, start_y, length, width, type, id] = piece;
        
        // 绘制板件填充
        ctx.fillStyle = type === 0 ? '#93c5fd' : '#fcd34d';
        ctx.fillRect(
          start_x * scale,
          start_y * scale,
          length * scale,
          width * scale
        );
        
        // 绘制板件边框
        ctx.strokeStyle = '#000';
        ctx.strokeRect(
          start_x * scale,
          start_y * scale,
          length * scale,
          width * scale
        );

        // 标注尺寸和ID
        ctx.scale(1, -1);
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        
        // 长度标注 - 固定距离15px
        const lengthText = `${length}mm`;
        const lengthWidth = ctx.measureText(lengthText).width;
        ctx.fillText(
          lengthText,
          (start_x + length/2) * scale - lengthWidth/2,
          -(start_y * scale) - 15  // 固定在上方15px处
        );
        
        // 宽度标注 - 固定距离15px
        const widthText = `${width}mm`;
        const widthWidth = ctx.measureText(widthText).width;
        ctx.save();
        ctx.translate(
          (start_x * scale) + 15,  // 固定在左侧15px处
          -(start_y + width/2) * scale
        );
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(widthText, -widthWidth/2, 0);
        ctx.restore();

        // ID标注 - 保持在中心
        ctx.font = '14px Arial';
        const idText = type === 0 ? `${id}` : `R${id}`;
        const idWidth = ctx.measureText(idText).width;
        ctx.fillText(
          idText,
          (start_x + length/2) * scale - idWidth/2,
          -(start_y + width/2) * scale
        );
        
        ctx.scale(1, -1);
      });
    }
  }, [layoutData]);

  const renderTable = (title: string, data: any[], type: 'orders' | 'others') => {
    // 计算当前页面中每个ID的使用数量
    const pageUsageCount = new Map<number, number>();
    if (layoutData && layoutData.cutted) {
      layoutData.cutted.forEach((piece: any[]) => {
        const [,,,, pieceType, pieceId] = piece;
        // 将pieceId转换为数字并减1以匹配数组索引
        const id = parseInt(pieceId) - 1;
        if ((type === 'orders' && pieceType === 0) || (type === 'others' && pieceType === 1)) {
          pageUsageCount.set(id, (pageUsageCount.get(id) || 0) + 1);
        }
      });
    }

    // 计算所有页面中的使用数量
    const totalUsageCount = new Map<number, number>();
    if (allCutted) {
      allCutted.forEach(page => {
        if (page.cutted) {
          page.cutted.forEach((piece: any[]) => {
            const [,,,, pieceType, pieceId] = piece;
            // 将pieceId转换为数字并减1以匹配数组索引
            const id = parseInt(pieceId) - 1;
            if ((type === 'orders' && pieceType === 0) || (type === 'others' && pieceType === 1)) {
              totalUsageCount.set(id, (totalUsageCount.get(id) || 0) + 1);
            }
          });
        }
      });
    }

    return (
      <div className="mb-4">
        <h3 className="font-semibold mb-2">{title}</h3>
        <table className="w-full border">
          <thead>
            <tr className="bg-gray-50">
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
            {data.map((item, index) => (
              <tr key={index} className={type === 'orders' ? 'bg-blue-200' : 'bg-yellow-200'}>
                <td className="border p-2">{type === 'others' ? `R${index + 1}` : index + 1}</td>
                <td className="border p-2">{item.length}</td>
                <td className="border p-2">{item.width}</td>
                <td className="border p-2">{type === 'others' ? totalUsageCount.get(index) || 0 : item.quantity}</td>
                <td className="border p-2">{pageUsageCount.get(index) || 0}</td>
                {type === 'others' && <td className="border p-2">{item.client}</td>}
                <td className="border p-2">{item.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  if (!layoutData) return <div>加载中...</div>;

  return (
    <div className="relative max-w-7xl mx-auto my-8 rounded-2xl shadow-2xl border bg-white flex flex-col h-[92vh]">
      {/* 导航和标题部分保持不变 */}
      <div className="flex items-center px-6 pt-4">
        <div className="flex gap-2 mb-4">
          <button 
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold"
            onClick={() => router.push(`/project/${projectId}`)}
          >
            项目
          </button>
          <button 
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg font-semibold"
            onClick={() => router.push(`/layout/${projectId}`)}
          >
            切板统计
          </button>
        </div>
      </div>

      <div className="px-6 pb-2 border-b">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">
            {projectName || '未命名项目'} - 第 {pageNum} 页
          </h1>
          <div className="flex gap-2">
            <button
              className={`px-3 py-1 rounded ${
                pageNum > 1
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
              onClick={() => pageNum > 1 && router.push(`/layout/${projectId}/${pageNum - 1}`)}
              disabled={pageNum <= 1}
            >
              上一页
            </button>
            <button
              className={`px-3 py-1 rounded ${
                pageNum < totalPages
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
              onClick={() => pageNum < totalPages && router.push(`/layout/${projectId}/${pageNum + 1}`)}
              disabled={pageNum >= totalPages}
            >
              下一页
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-600">
          使用率: {(layoutData.rate * 100).toFixed(1)}%
        </p>
      </div>

      {/* 内容区：左侧画布，右侧表格 */}
      <div className="flex-1 p-6 flex gap-6">
        {/* 左侧画布 */}
        <div className="flex-1 flex justify-center items-center">
          <canvas
            ref={canvasRef}
            className="border rounded-lg"
          />
        </div>

        {/* 右侧表格 */}
        <div className="w-96 overflow-y-auto">
          {renderTable('零件信息', orders, 'orders')}
          {renderTable('常用尺寸信息', others, 'others')}
        </div>
      </div>
    </div>
  );
} 