/**
 * 切割方案分组：同一组表示板材尺寸、利用率（六位小数）与各件几何一致。
 */

export type CutPlanPieceInput = {
  id?: unknown;
  is_stock?: boolean;
  length?: unknown;
  width?: unknown;
  start_x?: unknown;
  start_y?: unknown;
};

export type CutPlanPageInput = {
  plate?: readonly unknown[];
  cutted?: readonly CutPlanPieceInput[];
  rate?: unknown;
};

type NormalizedPiece = {
  id: unknown;
  is: boolean;
  l: unknown;
  w: unknown;
  sx: unknown;
  sy: unknown;
};

function normalizePiece(p: CutPlanPieceInput): NormalizedPiece {
  return {
    id: p.id,
    is: Boolean(p.is_stock),
    l: p.length,
    w: p.width,
    sx: p.start_x,
    sy: p.start_y,
  };
}

export function cutPlanSignature(page: CutPlanPageInput): string {
  if (!page?.plate || !Array.isArray(page.cutted)) return '';
  const plate = page.plate;
  if (plate.length < 2) return '';
  const [a, b] = plate;
  const pieces = [...page.cutted]
    .map(normalizePiece)
    .sort((x, y) => JSON.stringify(x).localeCompare(JSON.stringify(y)));
  const r = page.rate != null ? Math.round(Number(page.rate) * 1e6) / 1e6 : 0;
  return JSON.stringify({ a, b, r, pieces });
}

export type CutPlanGroup<T extends CutPlanPageInput = CutPlanPageInput> = {
  signature: string;
  /** 在原列表中的 0-based 下标 */
  indices: number[];
  representative: T;
};

/** 按方案顺序：先出现的切法成组，组内页码随列表顺序 */
export function groupCutPlansInOrder<T extends CutPlanPageInput>(
  pages: readonly T[]
): CutPlanGroup<T>[] {
  const bySig = new Map<string, CutPlanGroup<T>>();
  const order: string[] = [];
  for (let i = 0; i < pages.length; i++) {
    const page = pages[i];
    const sig = cutPlanSignature(page);
    let g = bySig.get(sig);
    if (!g) {
      g = { signature: sig, indices: [], representative: page };
      bySig.set(sig, g);
      order.push(sig);
    }
    g.indices.push(i);
  }
  return order.map((s) => bySig.get(s)!);
}

/** 多页组用色（高区分）；单页组在 UI 层用中性样式 */
export const GROUP_ACCENT_STYLES = [
  { border: 'border-amber-500', bg: 'bg-amber-200', text: 'text-amber-950', ring: 'ring-amber-400/50' },
  { border: 'border-sky-600', bg: 'bg-sky-200', text: 'text-sky-950', ring: 'ring-sky-500/40' },
  { border: 'border-violet-600', bg: 'bg-violet-200', text: 'text-violet-950', ring: 'ring-violet-500/40' },
  { border: 'border-emerald-600', bg: 'bg-emerald-200', text: 'text-emerald-950', ring: 'ring-emerald-500/40' },
  { border: 'border-rose-600', bg: 'bg-rose-200', text: 'text-rose-950', ring: 'ring-rose-500/40' },
  { border: 'border-orange-600', bg: 'bg-orange-200', text: 'text-orange-950', ring: 'ring-orange-500/40' },
] as const;

export function getGroupAccent(groupIndex: number) {
  return GROUP_ACCENT_STYLES[groupIndex % GROUP_ACCENT_STYLES.length];
}
