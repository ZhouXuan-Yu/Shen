<template>
  <header class="fixed top-0 left-0 right-0 z-50 glass border-b border-gray-100/50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <!-- Logo -->
        <NuxtLink to="/" class="flex items-center gap-3 group">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30 group-hover:shadow-purple-500/40 transition-all">
            <span class="text-white text-lg">🤟</span>
          </div>
          <span class="text-xl font-bold bg-gradient-to-r from-primary-900 to-primary-700 bg-clip-text text-transparent">
            译手
          </span>
        </NuxtLink>

        <!-- 桌面端导航 -->
        <nav class="hidden md:flex items-center gap-2">
          <!-- 受保护功能入口在点击时做登录校验 -->
          <NuxtLink
            to="/translate"
            class="nav-link"
            active-class="nav-link-active"
            @click.prevent="handleProtectedNav('/translate')"
          >
            <i class="bi bi-images mr-1.5"></i>
            图片翻译
          </NuxtLink>
          <NuxtLink
            to="/video-translate"
            class="nav-link"
            active-class="nav-link-active"
            @click.prevent="handleProtectedNav('/video-translate')"
          >
            <i class="bi bi-camera-reels mr-1.5"></i>
            视频翻译
          </NuxtLink>
          <NuxtLink
            to="/dictionary"
            class="nav-link"
            active-class="nav-link-active"
          >
            <i class="bi bi-book mr-1.5"></i>
            手语词典
          </NuxtLink>
          <NuxtLink
            to="/about"
            class="nav-link"
            active-class="nav-link-active"
          >
            <i class="bi bi-info-circle mr-1.5"></i>
            关于我们
          </NuxtLink>
        </nav>

        <!-- 右侧操作区 -->
        <div class="flex items-center gap-2">
          <!-- 主题切换按钮 -->
          <button 
            class="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-gray-600 hover:bg-gray-200 transition-colors"
            @click="toggleTheme"
          >
            <i class="bi bi-moon-stars"></i>
          </button>

          <!-- 用户菜单 -->
          <template v-if="authStore.isAuthenticated">
            <NuxtLink
              to="/profile"
              class="inline-flex items-center justify-center h-10 px-4 rounded-lg border border-gray-200 bg-white/80 text-sm font-medium text-gray-900 hover:bg-gray-900 hover:text-white transition-colors"
            >
              个人
            </NuxtLink>
          </template>
          <template v-else>
            <NuxtLink to="/auth/login" class="btn btn-ghost btn-sm rounded-lg">
              登录
            </NuxtLink>
            <NuxtLink to="/auth/register" class="btn btn-accent btn-sm rounded-lg shadow-lg shadow-accent/30 hover:shadow-xl hover:shadow-accent/40 transition-all">
              免费注册
            </NuxtLink>
          </template>

          <!-- 移动端菜单按钮 -->
          <button
            class="md:hidden w-10 h-10 rounded-lg bg-white/80 border border-gray-200 flex items-center justify-center text-gray-700 hover:bg-gray-100 transition-colors"
            @click="mobileMenuOpen = !mobileMenuOpen"
          >
            <i :class="mobileMenuOpen ? 'bi bi-x-lg' : 'bi bi-list'" class="text-xl"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- 移动端菜单 -->
    <Transition
      enter-active-class="transition-all duration-300 ease-out"
      leave-active-class="transition-all duration-200 ease-in"
      enter-from-class="opacity-0 -translate-y-4"
      enter-to-class="opacity-100 translate-y-0"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 -translate-y-4"
    >
      <div
        v-if="mobileMenuOpen"
        class="md:hidden border-t border-gray-100 bg-white/95 backdrop-blur-lg"
      >
        <nav class="flex flex-col p-4 space-y-1">
          <NuxtLink
            to="/recognize"
            class="mobile-nav-link"
            @click.prevent="handleProtectedNav('/recognize', true)"
          >
            <i class="bi bi-camera-video mr-3"></i>
            <span>实时翻译</span>
            <span class="ml-auto text-xs text-gray-400">摄像头实时识别</span>
          </NuxtLink>
          <NuxtLink
            to="/translate"
            class="mobile-nav-link"
            @click.prevent="handleProtectedNav('/translate', true)"
          >
            <i class="bi bi-images mr-3"></i>
            <span>图片翻译</span>
            <span class="ml-auto text-xs text-gray-400">多图序列分析</span>
          </NuxtLink>
          <NuxtLink
            to="/video-translate"
            class="mobile-nav-link"
            @click.prevent="handleProtectedNav('/video-translate', true)"
          >
            <i class="bi bi-camera-reels mr-3"></i>
            <span>视频翻译</span>
            <span class="ml-auto text-xs text-gray-400">整段视频识别</span>
          </NuxtLink>
          <NuxtLink
            to="/dictionary"
            class="mobile-nav-link"
            @click="mobileMenuOpen = false"
          >
            <i class="bi bi-book mr-3"></i>
            <span>手语词典</span>
            <span class="ml-auto text-xs text-gray-400">词汇学习</span>
          </NuxtLink>
          <NuxtLink
            to="/about"
            class="mobile-nav-link"
            @click="mobileMenuOpen = false"
          >
            <i class="bi bi-info-circle mr-3"></i>
            <span>关于我们</span>
          </NuxtLink>
          
          <hr class="my-3 border-gray-100" />
          
          <template v-if="authStore.isAuthenticated">
            <NuxtLink
              to="/profile"
              class="mobile-nav-link"
              @click="mobileMenuOpen = false"
            >
              <i class="bi bi-person-circle mr-3"></i>
              <span>个人中心</span>
            </NuxtLink>
            <NuxtLink
              to="/profile/history"
              class="mobile-nav-link"
              @click="mobileMenuOpen = false"
            >
              <i class="bi bi-clock-history mr-3"></i>
              <span>历史记录</span>
            </NuxtLink>
            <button
              class="w-full mobile-nav-link text-left text-red-500 hover:text-red-600 hover:bg-red-50"
              @click="handleLogout"
            >
              <i class="bi bi-box-arrow-right mr-3"></i>
              <span>退出登录</span>
            </button>
          </template>
          <template v-else>
            <NuxtLink
              to="/auth/login"
              class="mobile-nav-link"
              @click="mobileMenuOpen = false"
            >
              <i class="bi bi-box-arrow-in-right mr-3"></i>
              <span>登录</span>
            </NuxtLink>
            <NuxtLink
              to="/auth/register"
              class="flex items-center gap-3 px-4 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg shadow-purple-500/30"
              @click="mobileMenuOpen = false"
            >
              <i class="bi bi-person-plus mr-3"></i>
            </NuxtLink>
          </template>
        </nav>
      </div>
    </Transition>
  </header>
</template>

<script setup lang="ts">
const authStore = useAuthStore()
const mobileMenuOpen = ref(false)
const toast = useToast()
const router = useRouter()

async function handleLogout() {
  await authStore.logout()
  mobileMenuOpen.value = false
  toast.success('已退出登录')
}

function toggleTheme() {
  toast.info('主题切换功能开发中')
}

async function handleProtectedNav(path: string, isMobile = false) {
  if (authStore.isAuthenticated) {
    if (isMobile) {
      mobileMenuOpen.value = false
    }
    await router.push(path)
    return
  }

  if (isMobile) {
    mobileMenuOpen.value = false
  }

  toast.info('请先登录后再使用该功能')
  await router.push({
    path: '/auth/login',
    query: {
      redirect: path,
    },
  })
}

// 初始化认证状态
onMounted(() => {
  authStore.initAuth()
})
</script>

<style scoped>
.nav-link {
  @apply flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:text-primary-900 hover:bg-gray-100 transition-all duration-200;
}

.nav-link-active {
  @apply bg-primary-900 text-white hover:bg-primary-800;
}

.mobile-nav-link {
  @apply flex items-center gap-3 px-4 py-3 rounded-xl text-gray-600 hover:text-primary-900 hover:bg-gray-50 transition-all duration-200;
}
</style>
