'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/utils/supabaseClient'

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        router.push('/project')
      }
    }
    checkUser()
  }, [router])

  const handleLogin = async () => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })
      if (error) {
        setMessage(error.message)
      } else if (data.user) {
        setMessage('登录成功！')
        router.push('/project')
        router.refresh()
      }
    } catch {
      setMessage('登录过程中出现错误')
    }
  }

  const handleResetPassword = async () => {
    if (!email) {
      setMessage('请输入邮箱地址')
      return
    }
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/login/update-password`,
      })
      if (error) {
        setMessage(error.message)
      } else {
        setMessage('重置密码邮件已发送，请检查邮箱')
      }
    } catch {
      setMessage('发送重置邮件时出现错误')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLogin()
    }
  }

  return (
    <div className="page-gallery flex min-h-screen flex-col items-center justify-center px-4">
      <div className="card-auth">
        <h2 className="mb-8 text-center text-xl text-ink">登录</h2>
        <input
          className="field-gallery mb-4"
          type="email"
          placeholder="邮箱"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          className="field-gallery mb-6"
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          className="btn-gallery-primary w-full py-2.5"
          onClick={handleLogin}
        >
          登录
        </button>
        <button
          type="button"
          className="btn-gallery-link mt-5 w-full text-center text-sm text-ink-muted"
          onClick={handleResetPassword}
        >
          忘记密码？
        </button>
        {message && (
          <p className={`mt-6 text-center text-sm leading-relaxed ${message.includes('成功') ? 'text-accent-green' : 'text-[#b42318]'}`}>
            {message}
          </p>
        )}
      </div>
    </div>
  )
}
