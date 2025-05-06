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
    } catch (err) {
      setMessage('登录过程中出现错误')
    }
  }

  const handleResetPassword = async () => {
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: 'http://localhost:3000/login/update-password', // 修改为你的前端链接
    })
    if (error) {
      setMessage(error.message)
    } else {
      setMessage('重置密码邮件已发送，请检查邮箱')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLogin()
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="w-full max-w-sm p-6 border rounded shadow">
        <h2 className="mb-4 text-xl font-bold text-center">登录</h2>
        <input
          className="w-full p-2 mb-3 border rounded"
          type="email"
          placeholder="邮箱"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          className="w-full p-2 mb-3 border rounded"
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button
          className="w-full p-2 text-white bg-blue-600 rounded hover:bg-blue-700"
          onClick={handleLogin}
        >
          登录
        </button>
        <button
          className="w-full mt-3 text-sm text-blue-500 hover:underline"
          onClick={handleResetPassword}
        >
          忘记密码？
        </button>
        {message && <p className="mt-4 text-center text-red-500">{message}</p>}
      </div>
    </div>
  )
}
