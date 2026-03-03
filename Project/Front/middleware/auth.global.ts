import { useAuthStore } from '~/stores/auth'

export default defineNuxtRouteMiddleware((to) => {
  // 这些路由需要登录才能访问
  const protectedPrefixes = ['/translate', '/video-translate', '/recognize', '/profile']

  const shouldProtect = protectedPrefixes.some((prefix) =>
    to.path === prefix || to.path.startsWith(prefix + '/'),
  )

  // 登录/注册页面本身不拦截
  if (!shouldProtect) return

  const auth = useAuthStore()

  if (!auth.isAuthenticated) {
    return navigateTo({
      path: '/auth/login',
      query: {
        redirect: to.fullPath,
      },
    })
  }
})

