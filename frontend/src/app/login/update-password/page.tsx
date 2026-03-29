'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/utils/supabaseClient';

export default function UpdatePasswordPage() {
  const router = useRouter();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');

  // 检查用户是否已登录
  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        router.push('/project');
      }
    };
    checkUser();
  }, [router]);

  const handleUpdatePassword = async () => {
    // 验证密码是否一致
    if (password !== confirmPassword) {
      setMessage('两次输入的密码不一致，请重新输入');
      return;
    }

    // 验证密码长度
    if (password.length < 6) {
      setMessage('密码长度至少需要6个字符');
      return;
    }

    const { error } = await supabase.auth.updateUser({
      password,
    });
    if (error) {
      setMessage(error.message);
    } else {
      setMessage('密码重置成功！正在跳转...');
      router.push('/login');
    }
  };

  return (
    <div className="page-gallery flex min-h-screen flex-col items-center justify-center px-4">
      <div className="card-auth">
        <h2 className="mb-8 text-center text-xl text-ink">重置密码</h2>
        <input
          className="field-gallery mb-4"
          type="password"
          placeholder="请输入新密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <input
          className="field-gallery mb-6"
          type="password"
          placeholder="请再次输入新密码"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
        <button
          type="button"
          className="btn-gallery-green w-full py-2.5"
          onClick={handleUpdatePassword}
        >
          确认重置
        </button>
        {message && (
          <p className={`mt-6 text-center text-sm leading-relaxed ${message.includes('成功') ? 'text-accent-green' : 'text-[#b42318]'}`}>
            {message}
          </p>
        )}
      </div>
    </div>
  );
}
