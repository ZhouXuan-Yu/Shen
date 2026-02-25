import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginRequest, LoginResponse } from '~/types'

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
  function setToken(newToken: string) {
    token.value = newToken
    if (newToken) {
      localStorage.setItem('token', newToken)
    } else {
      localStorage.removeItem('token')
    }
  }

  function setUser(newUser: User) {
    user.value = newUser
  }

  async function login(credentials: LoginRequest): Promise<boolean> {
    loading.value = true
    error.value = null

    try {
      const config = useRuntimeConfig()
      const response = await $fetch<{ code: number; message: string; data: LoginResponse }>(
        `${config.public.apiBase}/auth/login`,
        {
          method: 'POST',
          body: credentials,
        }
      )

      if (response.code === 200 && response.data) {
        setToken(response.data.token)
        setUser(response.data.user)
        return true
      }

      error.value = response.message || '登录失败'
      return false
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
    token.value = null
    localStorage.removeItem('token')
    await navigateTo('/auth/login')
  }

  function initAuth() {
    const savedToken = localStorage.getItem('token')
    if (savedToken) {
      token.value = savedToken
      // 可以在这里添加获取用户信息的逻辑
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


