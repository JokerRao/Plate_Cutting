import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL?.trim();
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim();

/**
 * createClient 在 url/key 为空时会直接 throw，导致任意引用本模块的页面整页白屏。
 * 未配置时使用占位值，仅保证应用可挂载；真实请求需在 .env.local 中填写 Supabase 变量。
 */
const resolvedUrl = supabaseUrl || 'https://invalid.invalid';
const resolvedKey =
  supabaseAnonKey ||
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.placeholder';

if (process.env.NODE_ENV === 'development' && (!supabaseUrl || !supabaseAnonKey)) {
  console.warn(
    '[supabase] 未设置 NEXT_PUBLIC_SUPABASE_URL 或 NEXT_PUBLIC_SUPABASE_ANON_KEY，已使用占位客户端；请在 frontend/.env.local 中配置。'
  );
}

/** 按 Supabase 项目主机名区分 localStorage，避免换 URL/密钥后沿用旧 refresh token 触发 Invalid Refresh Token */
function authStorageKey(url: string): string {
  try {
    const host = new URL(url).hostname;
    return `plate-cutting.sb.${host}`;
  } catch {
    return 'plate-cutting.sb.placeholder';
  }
}

export const supabase: SupabaseClient = createClient(resolvedUrl, resolvedKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    storageKey: authStorageKey(resolvedUrl),
  },
});

/**
 * 本地会话损坏（如 Refresh Token Not Found）时清除存储，避免控制台反复报错。
 * 仅在浏览器、且已配置真实 Supabase 时执行。
 */
function recoverFromInvalidRefreshToken(): void {
  if (typeof window === 'undefined') return;
  if (resolvedUrl === 'https://invalid.invalid') return;

  void supabase.auth.getUser().then(({ error }) => {
    const msg = error?.message?.toLowerCase() ?? '';
    if (
      msg.includes('refresh token') ||
      msg.includes('invalid jwt') ||
      msg.includes('jwt expired')
    ) {
      console.warn(
        '[supabase] 会话已失效，已清除本地登录状态，请重新登录。',
        error?.message
      );
      void supabase.auth.signOut({ scope: 'local' });
    }
  });
}

recoverFromInvalidRefreshToken();
