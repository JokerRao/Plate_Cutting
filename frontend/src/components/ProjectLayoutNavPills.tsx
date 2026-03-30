'use client';

import { useRouter } from 'next/navigation';

const pillCore =
  'inline-flex items-center gap-1.5 rounded-md border font-medium transition-[filter,box-shadow,transform] duration-200 focus:outline-none focus-visible:ring-1 focus-visible:ring-offset-1';

/** 全局统一略小于原 h-9，与工具条主按钮对齐 */
const pillSizeDefault = 'h-8 px-2.5 text-xs leading-none shadow-sm';
const pillSizeToolbar = 'h-8 px-3 text-xs leading-none shadow-sm';

const currentMarkSm =
  'ml-0.5 shrink-0 rounded bg-white/25 px-1 text-[9px] font-semibold leading-none tracking-wide';
const currentMarkToolbar =
  'ml-0.5 shrink-0 rounded bg-white/25 px-1.5 text-[10px] font-semibold leading-none tracking-wide';

export function IconNavConfig({ className }: { className?: string }) {
  return (
    <svg className={`shrink-0 ${className ?? 'h-3.5 w-3.5'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}

/** 方案总览：饼图扇区 */
export function IconNavOverview({ className }: { className?: string }) {
  return (
    <svg className={`shrink-0 ${className ?? 'h-3.5 w-3.5'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
    </svg>
  );
}

/** 首页排版：分栏画布 */
export function IconNavHomeLayout({ className }: { className?: string }) {
  return (
    <svg className={`shrink-0 ${className ?? 'h-3.5 w-3.5'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
    </svg>
  );
}

export function IconNavList({ className }: { className?: string }) {
  return (
    <svg className={`shrink-0 ${className ?? 'h-3.5 w-3.5'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h10" />
    </svg>
  );
}

export type ProjectLayoutNavActive = 'config' | 'layout-list' | 'layout-detail';

type ShowPills = Partial<{
  config: boolean;
  homeLayout: boolean;
  schemeOverview: boolean;
  projectList: boolean;
}>;

type Props = {
  projectId: string;
  active: ProjectLayoutNavActive;
  /** 项目配置页：点击「首页排版」时先校验（如是否已切板） */
  onGoLayout?: () => void | Promise<void>;
  /** 点击「项目列表」时（如未保存确认）；不传则直接 /project */
  onProjectListNavigate?: () => void | Promise<void>;
  /** 拆行布局时按需隐藏某一组，未指定的为显示 */
  show?: ShowPills;
  className?: string;
  size?: 'default' | 'toolbar';
  /** @deprecated 请改用 show={{ projectList: true }} */
  showProjectList?: boolean;
  /** 单页排版：当前第几页（为 1 时「首页排版」显示为当前） */
  layoutPageNum?: number;
  /** 顶栏第一行已展示「当前」时设为 true，避免药丸内重复「当前」 */
  suppressPillCurrentLabel?: boolean;
};

/** 顶栏第一行：放在「当前」徽标右侧，与药丸内「项目列表」样式一致 */
export function ProjectListNavButton({
  size = 'default',
  onClick,
}: {
  size?: 'default' | 'toolbar';
  onClick?: () => void | Promise<void>;
}) {
  const router = useRouter();
  const pillBase = `${pillCore} ${size === 'toolbar' ? pillSizeToolbar : pillSizeDefault}`;
  const iconSz = size === 'toolbar' ? 'h-3.5 w-3.5' : 'h-3 w-3';
  return (
    <button
      type="button"
      className={`${pillBase} border-hairline bg-surface text-ink shadow-sm hover:border-[#c5c5c7] hover:bg-muted focus-visible:ring-[var(--accent)]`}
      onClick={() => void (onClick ? onClick() : router.push('/project'))}
    >
      <IconNavList className={iconSz} />
      项目列表
    </button>
  );
}

export default function ProjectLayoutNavPills({
  projectId,
  active,
  onGoLayout,
  onProjectListNavigate,
  show,
  className,
  size = 'default',
  showProjectList,
  layoutPageNum,
  suppressPillCurrentLabel = false,
}: Props) {
  const router = useRouter();

  const pillBase = `${pillCore} ${size === 'toolbar' ? pillSizeToolbar : pillSizeDefault}`;
  const iconSz = size === 'toolbar' ? 'h-3.5 w-3.5' : 'h-3 w-3';
  const currentMark = size === 'toolbar' ? currentMarkToolbar : currentMarkSm;
  const mark = !suppressPillCurrentLabel ? (
    <span className={currentMark}>当前</span>
  ) : null;

  const vis = {
    config: show?.config !== false,
    homeLayout: show?.homeLayout !== false,
    schemeOverview: show?.schemeOverview !== false,
    projectList: show?.projectList !== false || showProjectList === true,
  };

  const configActive = active === 'config';
  const layoutListActive = active === 'layout-list';
  const homeLayoutCurrent = active === 'layout-detail' && layoutPageNum === 1;

  const configShadow = 'shadow-[0_2px_10px_rgba(0,122,255,0.28)]';
  const tealShadow = 'shadow-[0_2px_10px_rgba(13,148,136,0.26)]';
  const violetShadow = 'shadow-[0_2px_10px_rgba(109,40,217,0.26)]';

  const goHomeLayout = () => {
    if (onGoLayout) {
      void onGoLayout();
      return;
    }
    router.push(`/layout/${projectId}/1`);
  };

  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className ?? 'mb-3'}`}>
      {vis.config &&
        (configActive ? (
          <span
            className={`${pillBase} cursor-default border-[var(--accent)] bg-[var(--accent)] text-white ${configShadow} focus-visible:ring-[var(--accent)]`}
            aria-current="page"
          >
            <IconNavConfig className={iconSz} />
            项目配置
            {mark}
          </span>
        ) : (
          <button
            type="button"
            className={`${pillBase} border-[#3d9eff] bg-[#3d9eff] text-white hover:brightness-110 active:scale-[0.99] ${configShadow} focus-visible:ring-[#3d9eff]`}
            onClick={() => router.push(`/project/${projectId}`)}
          >
            <IconNavConfig className={iconSz} />
            项目配置
          </button>
        ))}

      {vis.homeLayout &&
        (homeLayoutCurrent ? (
          <span
            className={`${pillBase} cursor-default border-teal-600 bg-teal-600 text-white ${tealShadow} focus-visible:ring-teal-500`}
            aria-current="page"
            title="第 1 张板材排版"
          >
            <IconNavHomeLayout className={iconSz} />
            首页排版
            {mark}
          </span>
        ) : (
          <button
            type="button"
            title={onGoLayout ? '打开第 1 张板材排版（需已执行切板）' : '第 1 张板材排版'}
            className={`${pillBase} border-teal-500 bg-teal-500 text-white hover:brightness-110 active:scale-[0.99] ${tealShadow} focus-visible:ring-teal-500`}
            onClick={goHomeLayout}
          >
            <IconNavHomeLayout className={iconSz} />
            首页排版
          </button>
        ))}

      {vis.schemeOverview &&
        (layoutListActive ? (
          <span
            className={`${pillBase} cursor-default border-violet-600 bg-violet-600 text-white ${violetShadow} focus-visible:ring-violet-500`}
            aria-current="page"
            title="全部板材方案总览与统计"
          >
            <IconNavOverview className={iconSz} />
            方案总览
            {mark}
          </span>
        ) : (
          <button
            type="button"
            title="全部板材方案总览与统计"
            className={`${pillBase} border-violet-500 bg-violet-500 text-white hover:brightness-110 active:scale-[0.99] ${violetShadow} focus-visible:ring-violet-500`}
            onClick={() => router.push(`/layout/${projectId}`)}
          >
            <IconNavOverview className={iconSz} />
            方案总览
          </button>
        ))}

      {vis.projectList && (
        <button
          type="button"
          className={`${pillBase} border-hairline bg-surface text-ink shadow-sm hover:border-[#c5c5c7] hover:bg-muted focus-visible:ring-[var(--accent)]`}
          onClick={() => void (onProjectListNavigate ? onProjectListNavigate() : router.push('/project'))}
        >
          <IconNavList className={iconSz} />
          项目列表
        </button>
      )}
    </div>
  );
}
