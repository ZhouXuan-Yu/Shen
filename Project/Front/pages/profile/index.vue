<template>
  <div class="pt-16 min-h-screen bg-gray-50">
    <!-- 顶部导航 -->
    <div class="bg-white border-b border-gray-100">
      <div class="max-w-4xl mx-auto px-4 py-8">
        <div class="flex items-center gap-4">
          <!-- 头像 -->
          <div class="w-20 h-20 rounded-full bg-accent-gradient flex items-center justify-center text-white text-2xl font-bold">
            {{ userStore.user?.username?.charAt(0).toUpperCase() || 'U' }}
          </div>
          <!-- 用户信息 -->
          <div class="flex-1">
            <h1 class="text-xl font-bold text-primary-900">{{ userStore.user?.username || '用户名' }}</h1>
            <p class="text-gray-500 text-sm">{{ userStore.user?.phone || '未登录' }}</p>
          </div>
          <!-- 设置按钮 -->
          <NuxtLink to="/profile/settings" class="btn btn-ghost btn-icon">
            <i class="bi bi-gear text-xl"></i>
          </NuxtLink>
        </div>
      </div>
    </div>

    <!-- 统计卡片 -->
    <div class="max-w-4xl mx-auto px-4 py-6">
      <div class="grid grid-cols-3 gap-4">
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-primary-900">{{ stats.total }}</div>
          <div class="text-sm text-gray-500">翻译次数</div>
        </div>
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-primary-900">{{ stats.today }}</div>
          <div class="text-sm text-gray-500">今日翻译</div>
        </div>
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-success">{{ stats.accuracy }}%</div>
          <div class="text-sm text-gray-500">平均准确率</div>
        </div>
      </div>
    </div>

    <!-- 菜单列表 -->
    <div class="max-w-4xl mx-auto px-4">
      <div class="card">
        <div class="py-2">
          <NuxtLink to="/profile/history" class="flex items-center px-4 py-3 hover:bg-gray-50 transition-colors">
            <div class="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center mr-3">
              <i class="bi bi-clock-history text-blue-600"></i>
            </div>
            <span class="flex-1 text-primary-900">翻译历史</span>
            <i class="bi bi-chevron-right text-gray-400"></i>
          </NuxtLink>
          
          <NuxtLink to="/dictionary" class="flex items-center px-4 py-3 hover:bg-gray-50 transition-colors">
            <div class="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center mr-3">
              <i class="bi bi-book text-green-600"></i>
            </div>
            <span class="flex-1 text-primary-900">我的收藏</span>
            <i class="bi bi-chevron-right text-gray-400"></i>
          </NuxtLink>
          
          <NuxtLink to="/profile/learning" class="flex items-center px-4 py-3 hover:bg-gray-50 transition-colors">
            <div class="w-10 h-10 rounded-lg bg-purple-100 flex items-center justify-center mr-3">
              <i class="bi bi-mortarboard text-purple-600"></i>
            </div>
            <span class="flex-1 text-primary-900">学习记录</span>
            <i class="bi bi-chevron-right text-gray-400"></i>
          </NuxtLink>
          
          <NuxtLink to="/help" class="flex items-center px-4 py-3 hover:bg-gray-50 transition-colors">
            <div class="w-10 h-10 rounded-lg bg-yellow-100 flex items-center justify-center mr-3">
              <i class="bi bi-question-circle text-yellow-600"></i>
            </div>
            <span class="flex-1 text-primary-900">使用帮助</span>
            <i class="bi bi-chevron-right text-gray-400"></i>
          </NuxtLink>
          
          <NuxtLink to="/about" class="flex items-center px-4 py-3 hover:bg-gray-50 transition-colors">
            <div class="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center mr-3">
              <i class="bi bi-info-circle text-gray-600"></i>
            </div>
            <span class="flex-1 text-primary-900">关于我们</span>
            <i class="bi bi-chevron-right text-gray-400"></i>
          </NuxtLink>
        </div>
      </div>

      <!-- 退出登录 -->
      <button class="btn btn-secondary w-full mt-6 rounded-lg" @click="handleLogout">
        <i class="bi bi-box-arrow-right mr-2"></i>退出登录
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({ title: '个人中心 - 译手 HandTalk AI' })

const userStore = useAuthStore()
const toast = useToast()

const stats = reactive({ total: 128, today: 5, accuracy: 94 })

function handleLogout() {
  userStore.logout()
  toast.success('已退出登录')
  navigateTo('/')
}
</script>


