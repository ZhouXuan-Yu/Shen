<template>
  <div class="min-h-screen flex">
    <!-- 左侧品牌区域 -->
    <div class="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-teal-300 via-sky-300 to-lime-300 relative overflow-hidden">
      <!-- 背景装饰 -->
      <div class="absolute inset-0">
        <div class="absolute top-20 right-20 w-72 h-72 bg-white/30 rounded-full blur-3xl"></div>
        <div class="absolute bottom-20 left-20 w-96 h-96 bg-teal-200/40 rounded-full blur-3xl"></div>
        <div class="absolute top-1/3 right-1/3 w-64 h-64 bg-lime-200/30 rounded-full blur-3xl"></div>
      </div>
      
      <!-- 内容 -->
      <div class="relative z-10 flex flex-col justify-center px-16 text-teal-900">
        <NuxtLink to="/" class="flex items-center gap-3 mb-16">
          <div class="w-14 h-14 rounded-xl bg-white/70 backdrop-blur-lg flex items-center justify-center">
            <span class="text-2xl">🤟</span>
          </div>
          <div>
            <span class="text-3xl font-bold">译手</span>
            <p class="text-white/60 text-sm">HandTalk AI</p>
          </div>
        </NuxtLink>
        
        <h1 class="text-5xl font-bold mb-6 leading-tight">
          开始您的<br />
          <span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-pink-300">手语翻译之旅</span>
        </h1>
        
        <p class="text-xl text-white/80 mb-10 max-w-md leading-relaxed">
          加入译手 AI 大家庭，用科技连接世界，让沟通无障碍，让世界更温暖。
        </p>
        
        <!-- 统计数据 -->
        <!-- <div class="grid grid-cols-3 gap-6">
          <div>
            <div class="text-3xl font-bold text-teal-900">100K+</div>
            <div class="text-teal-900/70 text-sm">翻译次数</div>
          </div>
          <div>
            <div class="text-3xl font-bold text-teal-900">98%</div>
            <div class="text-teal-900/70 text-sm">准确率</div>
          </div>
          <div>
            <div class="text-3xl font-bold text-teal-900">1M+</div>
            <div class="text-teal-900/70 text-sm">用户数</div>
          </div>
        </div> -->
      </div>
    </div>

    <!-- 右侧表单区域 -->
    <div class="flex-1 flex items-center justify-center p-8 bg-gradient-to-b from-gray-50 to-white">
      <div class="w-full max-w-md">
        <!-- 移动端 Logo -->
        <div class="lg:hidden text-center mb-8">
          <NuxtLink to="/" class="inline-flex items-center gap-3">
            <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-teal-400 to-lime-400 flex items-center justify-center shadow-lg">
              <span class="text-xl">🤟</span>
            </div>
            <span class="text-2xl font-bold text-teal-900">译手</span>
          </NuxtLink>
        </div>

        <!-- 标题 -->
        <div class="text-center mb-10">
          <h2 class="text-3xl font-bold text-teal-900 mb-3">创建账号</h2>
          <p class="text-slate-600">注册译手，解锁全功能手语翻译体验</p>
        </div>

        <!-- 注册表单 -->
        <div class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 p-8">
          <form @submit.prevent="handleRegister" class="space-y-6">
            <!-- 用户名 -->
            <div class="space-y-2">
              <label class="block text-sm font-medium text-slate-700 ml-1">用户名</label>
              <div class="relative">
                <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <i class="bi bi-person text-gray-400 text-lg"></i>
                </div>
                <input
                  type="text"
                  class="input pl-12 py-3 bg-gray-50 border-0 rounded-xl focus:bg-white focus:ring-2 focus:ring-purple-500/20 transition-all"
                  placeholder="请输入用户名"
                  v-model="form.username"
                  required
                />
              </div>
            </div>

            <!-- 邮箱 -->
            <div class="space-y-2">
              <label class="block text-sm font-medium text-slate-700 ml-1">邮箱地址</label>
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

            <!-- 密码 -->
            <div class="space-y-2">
              <label class="block text-sm font-medium text-slate-700 ml-1">密码</label>
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
              <!-- 密码强度指示 -->
              <div v-if="form.password" class="ml-1">
                <div class="flex gap-1">
                  <div 
                    class="h-1 flex-1 rounded-full transition-colors"
                    :class="passwordStrength >= 1 ? 'bg-red-500' : 'bg-gray-200'"
                  ></div>
                  <div 
                    class="h-1 flex-1 rounded-full transition-colors"
                    :class="passwordStrength >= 2 ? 'bg-yellow-500' : 'bg-gray-200'"
                  ></div>
                  <div 
                    class="h-1 flex-1 rounded-full transition-colors"
                    :class="passwordStrength >= 3 ? 'bg-green-500' : 'bg-gray-200'"
                  ></div>
                </div>
                <p class="text-xs mt-1" :class="passwordTextColor">
                  {{ passwordStrengthText }}
                </p>
              </div>
            </div>

            <!-- 确认密码 -->
            <div class="space-y-2">
              <label class="block text-sm font-medium text-slate-700 ml-1">确认密码</label>
              <div class="relative">
                <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <i class="bi bi-lock-fill text-gray-400 text-lg"></i>
                </div>
                <input
                  :type="showConfirmPassword ? 'text' : 'password'"
                  class="input pl-12 py-3 bg-gray-50 border-0 rounded-xl focus:bg-white focus:ring-2 focus:ring-purple-500/20 transition-all"
                  placeholder="请再次输入密码"
                  v-model="form.confirmPassword"
                  required
                />
                <button
                  type="button"
                  class="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 transition-colors"
                  @click="showConfirmPassword = !showConfirmPassword"
                >
                  <i :class="showConfirmPassword ? 'bi bi-eye-slash' : 'bi bi-eye'" class="text-lg"></i>
                </button>
              </div>
            </div>

            <!-- 服务条款 -->
            <div class="flex items-start gap-2">
              <input
                type="checkbox"
                class="w-4 h-4 mt-0.5 rounded border-gray-300 text-teal-600 focus:ring-teal-500 mt-1"
                v-model="agreeTerms"
              />
              <span class="text-sm text-slate-600 leading-relaxed">
                我已阅读并同意
                <a href="#" class="text-teal-600 hover:text-teal-700 font-medium">服务条款</a>
                和
                <a href="#" class="text-teal-600 hover:text-teal-700 font-medium">隐私政策</a>
              </span>
            </div>

            <!-- 注册按钮 -->
            <button
              type="submit"
              class="btn w-full py-4 rounded-xl text-lg font-medium bg-gradient-to-r from-teal-400 via-sky-400 to-lime-400 text-white shadow-lg shadow-teal-400/40 hover:shadow-xl hover:shadow-teal-500/50 hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="loading || !agreeTerms"
            >
              <span v-if="loading" class="spinner-border spinner-border-sm mr-2"></span>
              <i v-else class="bi bi-person-plus mr-2"></i>
              免费注册
            </button>

            <!-- 错误提示 -->
            <div
              v-if="error"
              class="p-4 rounded-xl bg-red-50 border border-red-100 text-red-600 text-sm text-center"
            >
              {{ error }}
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

          <!-- 登录链接 -->
          <p class="text-center text-slate-600 mt-8">
            已有账号？
            <NuxtLink to="/auth/login" class="text-purple-600 hover:text-purple-700 font-medium transition-colors">
              立即登录
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
  title: '注册 - 译手 HandTalk AI',
  description: '注册译手账号，开始使用手语翻译功能',
})

// 布局
definePageMeta({
  layout: 'auth',
})

// 表单数据
const form = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
})

const showPassword = ref(false)
const showConfirmPassword = ref(false)
const agreeTerms = ref(false)
const loading = ref(false)
const sendingCaptcha = ref(false)
const error = ref('')

// Composables
const toast = useToast()
const authStore = useAuthStore()

// 密码强度计算
const passwordStrength = computed(() => {
  const password = form.password
  if (!password) return 0
  
  let strength = 0
  if (password.length >= 8) strength++
  if (/[A-Z]/.test(password)) strength++
  if (/[0-9]/.test(password)) strength++
  if (/[^A-Za-z0-9]/.test(password)) strength++
  
  return strength
})

const passwordStrengthText = computed(() => {
  const texts = ['密码太短', '弱', '中', '强']
  return texts[passwordStrength.value]
})

const passwordTextColor = computed(() => {
  const colors = ['text-red-500', 'text-red-500', 'text-yellow-500', 'text-green-500']
  return colors[passwordStrength.value]
})

// 发送验证码（示例，占位逻辑）
async function sendCaptcha() {
  if (!form.email) {
    toast.error('请输入邮箱地址')
    return
  }
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(form.email)) {
    toast.error('请输入正确的邮箱地址')
    return
  }
  
  sendingCaptcha.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    toast.success('验证码已发送至您的邮箱（示例）')
  } catch {
    toast.error('验证码发送失败')
  } finally {
    sendingCaptcha.value = false
  }
}

// 处理注册：本地固化保存账号信息，注册后再去登录
async function handleRegister() {
  error.value = ''
  
  // 验证
  if (!form.username || !form.email || !form.password || !form.confirmPassword) {
    error.value = '请填写完整的注册信息'
    return
  }
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(form.email)) {
    error.value = '请输入正确的邮箱地址'
    return
  }
  
  if (form.password !== form.confirmPassword) {
    error.value = '两次输入的密码不一致'
    return
  }
  
  if (form.password.length < 8) {
    error.value = '密码长度至少为 8 位'
    return
  }
  
  if (!agreeTerms.value) {
    error.value = '请同意服务条款和隐私政策'
    return
  }
  
  loading.value = true
  try {
    const localUser: any = {
      username: form.username,
      email: form.email,
      avatar: '',
      // 仅用于本地示例，真实项目不要明文保存密码
      password: form.password,
    }

    authStore.setUser(localUser)

    toast.success('注册成功！请使用该账户登录')
    await navigateTo('/auth/login')
  } catch {
    error.value = '注册失败，请稍后重试'
  } finally {
    loading.value = false
  }
}
</script>
