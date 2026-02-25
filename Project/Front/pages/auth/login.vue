<template>
  <div class="min-h-screen flex">
    <!-- 左侧品牌区域 -->
    <div class="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-800 relative overflow-hidden">
      <!-- 背景装饰 -->
      <div class="absolute inset-0">
        <div class="absolute top-20 left-20 w-72 h-72 bg-white/10 rounded-full blur-3xl"></div>
        <div class="absolute bottom-20 right-20 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl"></div>
        <div class="absolute top-1/2 left-1/3 w-64 h-64 bg-pink-400/10 rounded-full blur-3xl"></div>
      </div>
      
      <!-- 内容 -->
      <div class="relative z-10 flex flex-col justify-center px-16 text-white">
        <NuxtLink to="/" class="flex items-center gap-3 mb-16">
          <div class="w-14 h-14 rounded-xl bg-white/20 backdrop-blur-lg flex items-center justify-center">
            <span class="text-2xl">🤟</span>
          </div>
          <div>
            <span class="text-3xl font-bold">译手</span>
            <p class="text-white/60 text-sm">HandTalk AI</p>
          </div>
        </NuxtLink>
        
        <h1 class="text-5xl font-bold mb-6 leading-tight">
          让手语翻译<br />
          <span class="text-transparent bg-clip-text bg-gradient-to-r from-pink-300 to-purple-300">像呼吸一样自然</span>
        </h1>
        
        <p class="text-xl text-white/80 mb-10 max-w-md leading-relaxed">
          面向中国市场的 AI 手语翻译应用，用科技传递人文关怀，消除听障人士与健听人之间的沟通障碍。
        </p>
        
        <!-- 功能特性 -->
        <div class="space-y-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
              <i class="bi bi-lightning-charge"></i>
            </div>
            <span class="text-white/90">0.1 秒实时翻译</span>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
              <i class="bi bi-check-circle"></i>
            </div>
            <span class="text-white/90">98% 超高准确率</span>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
              <i class="bi bi-shield-check"></i>
            </div>
            <span class="text-white/90">安全可靠的数据保护</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 右侧表单区域 -->
    <div class="flex-1 flex items-center justify-center p-8 bg-gradient-to-b from-gray-50 to-white">
      <div class="w-full max-w-md">
        <!-- 移动端 Logo -->
        <div class="lg:hidden text-center mb-8">
          <NuxtLink to="/" class="inline-flex items-center gap-3">
            <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <span class="text-xl">🤟</span>
            </div>
            <span class="text-2xl font-bold text-primary-900">译手</span>
          </NuxtLink>
        </div>

        <!-- 标题 -->
        <div class="text-center mb-10">
          <h2 class="text-3xl font-bold text-primary-900 mb-3">欢迎回来</h2>
          <p class="text-gray-600">登录账号，开始使用手语翻译</p>
        </div>

        <!-- 登录表单 -->
        <div class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 p-8">
          <form @submit.prevent="handleLogin" class="space-y-6">
            <!-- 邮箱输入 -->
            <div class="space-y-2">
              <label class="block text-sm font-medium text-gray-700 ml-1">邮箱地址</label>
              <div class="relative">
                <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <i class="bi bi-envelope text-gray-400 text-lg"></i>
                </div>
                <input
                  type="email"
                  class="input pl-12 py-3 bg-gray-50 border-0 rounded-xl focus:bg-white focus:ring-2 focus:ring-purple-500/20 transition-all"
                  placeholder="请输入邮箱地址"
                  v-model="form.email"
                  required
                />
              </div>
            </div>

            <!-- 密码输入 -->
            <div class="space-y-2">
              <label class="block text-sm font-medium text-gray-700 ml-1">密码</label>
              <div class="relative">
                <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <i class="bi bi-lock text-gray-400 text-lg"></i>
                </div>
                <input
                  :type="showPassword ? 'text' : 'password'"
                  class="input pl-12 pr-12 py-3 bg-gray-50 border-0 rounded-xl focus:bg-white focus:ring-2 focus:ring-purple-500/20 transition-all"
                  placeholder="请输入密码"
                  v-model="form.password"
                  required
                />
                <button
                  type="button"
                  class="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 transition-colors"
                  @click="showPassword = !showPassword"
                >
                  <i :class="showPassword ? 'bi bi-eye-slash' : 'bi bi-eye'" class="text-lg"></i>
                </button>
              </div>
            </div>

            <!-- 选项 -->
            <div class="flex items-center justify-between">
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  class="w-4 h-4 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  v-model="rememberMe"
                />
                <span class="text-sm text-gray-600">记住我</span>
              </label>
              <a href="#" class="text-sm text-purple-600 hover:text-purple-700 font-medium transition-colors">
                忘记密码？
              </a>
            </div>

            <!-- 登录按钮 -->
            <button
              type="submit"
              class="btn w-full py-4 rounded-xl text-lg font-medium bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-purple-500/30 hover:shadow-xl hover:shadow-purple-500/40 hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="authStore.loading"
            >
              <span v-if="authStore.loading" class="spinner-border spinner-border-sm mr-2"></span>
              <i v-else class="bi bi-arrow-right-circle mr-2"></i>
              登录
            </button>

            <!-- 错误提示 -->
            <div
              v-if="authStore.error"
              class="p-4 rounded-xl bg-red-50 border border-red-100 text-red-600 text-sm text-center"
            >
              {{ authStore.error }}
            </div>
          </form>

          <!-- 分割线 -->
          <div class="relative my-8">
            <div class="absolute inset-0 flex items-center">
              <div class="w-full border-t border-gray-200"></div>
            </div>
            <div class="relative flex justify-center">
              <span class="px-4 bg-white text-gray-400 text-sm">或者</span>
            </div>
          </div>

          <!-- 第三方登录 -->
          <div class="grid grid-cols-2 gap-4">
            <button class="flex items-center justify-center gap-2 py-3 rounded-xl border border-gray-200 hover:bg-gray-50 hover:border-gray-300 transition-all">
              <i class="bi bi-wechat text-xl text-green-500"></i>
              <span class="text-sm text-gray-600">微信登录</span>
            </button>
            <button class="flex items-center justify-center gap-2 py-3 rounded-xl border border-gray-200 hover:bg-gray-50 hover:border-gray-300 transition-all">
              <i class="bi bi-github text-xl"></i>
              <span class="text-sm text-gray-600">GitHub</span>
            </button>
          </div>
        </div>

        <!-- 注册链接 -->
        <div class="mt-8 text-center">
          <p class="text-gray-600">
            还没有账号？
            <NuxtLink to="/auth/register" class="text-purple-600 hover:text-purple-700 font-medium transition-colors">
              免费注册
            </NuxtLink>
          </p>
        </div>

        <!-- 版权 -->
        <p class="text-center text-xs text-gray-400 mt-10">
          © {{ new Date().getFullYear() }} HandTalk AI. 让沟通无障碍。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// SEO 元信息
useSeoMeta({
  title: '登录 - 译手 HandTalk AI',
  description: '登录译手账号，开始使用手语翻译功能',
})

// 布局
definePageMeta({
  layout: 'auth',
})

// 表单数据
const form = reactive({
  email: '',
  password: '',
})

const showPassword = ref(false)
const rememberMe = ref(false)

// Composables
const authStore = useAuthStore()
const toast = useToast()

// 处理登录
async function handleLogin() {
  if (!form.email || !form.password) {
    toast.error('请填写完整的登录信息')
    return
  }

  // 邮箱验证
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(form.email)) {
    toast.error('请输入正确的邮箱地址')
    return
  }

  const success = await authStore.login({
    email: form.email,
    password: form.password,
  })

  if (success) {
    toast.success('登录成功！')
    await navigateTo('/profile')
  }
}
</script>
