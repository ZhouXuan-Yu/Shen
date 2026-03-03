import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginRequest } from '~/types'

const LOCAL_TOKEN_KEY = 'token'
const LOCAL_USER_KEY = 'auth_user'

export const useAuthStore = defineStore('auth', () => {
  // 状态
  const user = ref<User | null>(null)
  const token = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // 计算属性
  const isAuthenticated = computed(() => !!token.value && !!user.value)
  const userAvatar = computed(() => user.value?.avatar || '')

  // 方法
  function setToken(newToken: string | null) {
    token.value = newToken
    if (newToken) {
      localStorage.setItem(LOCAL_TOKEN_KEY, newToken)
    } else {
      localStorage.removeItem(LOCAL_TOKEN_KEY)
    }
  }

  function setUser(newUser: User | null) {
    user.value = newUser

    // 仅在提供了密码信息时才更新本地缓存（用于注册或修改密码场景）
    const anyUser = newUser as any
    if (newUser && anyUser?.password) {
      const cached = {
        username: anyUser.username,
        email: anyUser.email,
        avatar: anyUser.avatar || '',
        // 仅用于本地 demo，真实项目不要明文保存密码
        password: anyUser.password,
      }
      localStorage.setItem(LOCAL_USER_KEY, JSON.stringify(cached))
    }
  }

  async function login(credentials: LoginRequest): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const raw = localStorage.getItem(LOCAL_USER_KEY)
      if (!raw) {
        error.value = '尚未注册账号，请先完成注册'
        return false
      }

      let saved: any
      try {
        saved = JSON.parse(raw)
      } catch {
        error.value = '本地账户数据异常，请重新注册'
        return false
      }

      if (saved.email !== (credentials as any).email) {
        error.value = '该邮箱尚未注册，请先注册账号'
        return false
      }

      if (saved.password !== (credentials as any).password) {
        error.value = '邮箱或密码不正确'
        return false
      }

      const mockToken = `local-token-${Date.now()}`
      setToken(mockToken)

      // 登录后只挂载运行时 user，不覆盖本地密码信息
      user.value = {
        ...(user.value as any),
        username: saved.username,
        email: saved.email,
        avatar: saved.avatar || '',
      } as User
      return true
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '登录失败，请重试'
      error.value = errorMessage
      return false
    } finally {
      loading.value = false
    }
  }

  async function logout() {
    user.value = null
    setToken(null)
    await navigateTo('/auth/login')
  }

  function initAuth() {
    const savedToken = localStorage.getItem(LOCAL_TOKEN_KEY)
    const rawUser = localStorage.getItem(LOCAL_USER_KEY)

    if (savedToken) {
      token.value = savedToken
    }

    if (rawUser) {
      try {
        const parsed: any = JSON.parse(rawUser)
        user.value = {
          ...(user.value as any),
          username: parsed.username,
          email: parsed.email,
          avatar: parsed.avatar || '',
        } as User
      } catch {
        // ignore parse error
      }
    }
  }

  return {
    // 状态
    user,
    token,
    loading,
    error,
    // 计算属性
    isAuthenticated,
    userAvatar,
    // 方法
    setToken,
    setUser,
    login,
    logout,
    initAuth,
  }
})

