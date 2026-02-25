<template>
  <div class="pt-16 min-h-screen bg-gray-50">
    <!-- 顶部导航 -->
    <div class="bg-white border-b border-gray-100 sticky top-16 z-30">
      <div class="max-w-6xl mx-auto px-4">
        <div class="flex items-center justify-between h-14">
          <NuxtLink to="/profile" class="flex items-center text-gray-600 hover:text-primary-900">
            <i class="bi bi-arrow-left mr-1"></i> 返回
          </NuxtLink>
          <h1 class="font-medium text-primary-900">翻译历史</h1>
          <button v-if="history.length > 0" class="text-sm text-gray-500" @click="clearHistory">
            清除
          </button>
        </div>
      </div>
    </div>

    <div class="max-w-6xl mx-auto px-4 py-6">
      <!-- 筛选标签 -->
      <div class="flex items-center gap-2 overflow-x-auto pb-4 scrollbar-hide">
        <button v-for="filter in filters" :key="filter.value" 
          class="badge whitespace-nowrap px-3 py-1.5 rounded-full transition-all"
          :class="activeFilter === filter.value ? 'bg-primary-900 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'"
          @click="activeFilter = filter.value">
          {{ filter.label }}
        </button>
      </div>

      <!-- 统计概览 -->
      <div class="grid grid-cols-3 gap-3 mb-6">
        <div class="card p-3 text-center">
          <div class="text-lg font-bold text-primary-900">{{ stats.total }}</div>
          <div class="text-xs text-gray-500">总次数</div>
        </div>
        <div class="card p-3 text-center">
          <div class="text-lg font-bold text-primary-900">{{ stats.today }}</div>
          <div class="text-xs text-gray-500">今日</div>
        </div>
        <div class="card p-3 text-center">
          <div class="text-lg font-bold text-success">{{ stats.avgAccuracy }}%</div>
          <div class="text-xs text-gray-500">平均准度</div>
        </div>
      </div>

      <!-- 历史记录网格 -->
      <div v-if="filteredHistory.length > 0" class="row g-3">
        <div v-for="record in filteredHistory" :key="record.id" class="col-6 col-md-4 col-lg-3">
          <div class="card card-hover">
            <!-- 缩略图 -->
            <div class="relative aspect-video bg-gray-100 rounded-t-xl overflow-hidden">
              <img v-if="record.thumbnail" :src="record.thumbnail" class="w-full h-full object-cover" />
              <div v-else class="w-full h-full flex items-center justify-center">
                <i :class="record.type === 'realtime' ? 'bi bi-camera-video' : 'bi bi-image'" class="text-3xl text-gray-300"></i>
              </div>
              <!-- 类型标签 -->
              <span class="absolute bottom-2 right-2 px-2 py-0.5 bg-black/50 rounded text-white text-xs">
                {{ record.type === 'realtime' ? '实时' : record.type === 'upload_video' ? '视频' : '图片' }}
              </span>
            </div>
            <!-- 详情 -->
            <div class="p-3">
              <h3 class="font-bold text-primary-900 truncate">{{ record.result }}</h3>
              <div class="flex items-center justify-between mt-2">
                <span class="text-xs text-gray-500">{{ formatTime(record.createdAt) }}</span>
                <span class="text-xs" :class="record.confidence >= 90 ? 'text-success' : 'text-warning'">
                  {{ Math.round(record.confidence) }}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 空状态 -->
      <div v-else class="text-center py-12">
        <i class="bi bi-inbox text-5xl text-gray-300 mb-4"></i>
        <p class="text-gray-500">暂无翻译记录</p>
        <NuxtLink to="/recognize" class="btn btn-accent mt-4 rounded-full">开始翻译</NuxtLink>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({ title: '翻译历史 - 译手 HandTalk AI' })

const recognitionStore = useRecognitionStore()
const toast = useToast()

const activeFilter = ref('all')
const filters = [
  { label: '全部', value: 'all' },
  { label: '实时', value: 'realtime' },
  { label: '视频', value: 'upload_video' },
  { label: '图片', value: 'upload_image' },
]

const history = computed(() => recognitionStore.history)

const filteredHistory = computed(() => {
  if (activeFilter.value === 'all') return history.value
  return history.value.filter(h => h.type === activeFilter.value)
})

const stats = reactive({ total: 128, today: 5, avgAccuracy: 94 })

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function clearHistory() {
  recognitionStore.clearHistory()
  toast.success('历史记录已清除')
}

onMounted(() => recognitionStore.loadHistory())
</script>


