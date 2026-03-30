import { createClient } from '@supabase/supabase-js';

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

export const supabase = createClient(resolvedUrl, resolvedKey);