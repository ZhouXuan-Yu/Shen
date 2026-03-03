<template>
  <div class="pt-16 min-h-screen bg-gray-50">
    <!-- 顶部导航 -->
    <div class="bg-white border-b border-gray-100 sticky top-16 z-30">
      <div class="max-w-6xl mx-auto px-4">
        <div class="flex items-center justify-between h-14">
          <NuxtLink to="/profile" class="flex items-center text-gray-600 hover:text-primary-900">
            <i class="bi bi-arrow-left mr-1"></i> 返回
          </NuxtLink>
          <h1 class="font-medium text-primary-900">我的收藏</h1>
          <span class="text-xs text-gray-400">
            共 {{ filteredFavorites.length }} 条
          </span>
        </div>
      </div>
    </div>

    <div class="max-w-6xl mx-auto px-4 py-6">
      <!-- 筛选标签，与历史记录一致：全部 / 视频 / 图片 -->
      <div class="flex items-center gap-2 overflow-x-auto pb-4 scrollbar-hide">
        <button
          v-for="filter in filters"
          :key="filter.value"
          class="badge whitespace-nowrap px-3 py-1.5 rounded-full transition-all"
          :class="
            activeFilter === filter.value
              ? 'bg-primary-900 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          "
          @click="activeFilter = filter.value"
        >
          {{ filter.label }}
        </button>
      </div>

      <div v-if="groupedFavorites.length > 0" class="space-y-4">
        <div
          v-for="group in groupedFavorites"
          :key="group.date"
          class="space-y-2"
        >
          <div class="text-xs font-medium text-gray-400">
            {{ group.dateLabel }}
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div
              v-for="record in group.items"
              :key="record.id"
            >
              <div
                class="group bg-white rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition cursor-pointer flex flex-col h-full overflow-hidden"
                @click="handleRecordClick(record)"
              >
                <!-- 缩略图 / 占位 -->
                <div
                  class="relative aspect-video bg-gray-50"
                >
                  <img
                    v-if="record.thumbnail"
                    :src="record.thumbnail"
                    class="w-full h-full object-cover"
                  />
                  <div
                    v-else
                    class="absolute inset-0 flex items-center justify-center"
                  >
                    <div
                      class="flex items-center justify-center w-12 h-12 rounded-full bg-gray-900/5"
                    >
                      <i
                        :class="
                          record.type === 'upload_video'
                            ? 'bi bi-camera-video'
                            : 'bi bi-image'
                        "
                        class="text-2xl text-gray-300"
                      ></i>
                    </div>
                  </div>
                  <!-- 类型标签 -->
                  <span
                    class="absolute bottom-2 right-2 px-2 py-0.5 rounded-full bg-black/50 text-white text-[10px] tracking-wide"
                  >
                    {{
                      record.type === 'upload_video'
                        ? '视频'
                        : record.type === 'upload_image'
                          ? '图片'
                          : '其他'
                    }}
                  </span>
                </div>
                <!-- 详情 -->
                <div class="p-3 space-y-2">
                  <h3 class="font-semibold text-primary-900 text-sm line-clamp-2 min-h-[2.75rem]">
                    {{ record.result }}
                  </h3>
                  <div class="flex items-center justify-between text-xs text-gray-500">
                    <span>
                      {{ formatTime(record.createdAt) }}
                    </span>
                    <span class="font-medium text-green-600">
                      {{ Math.round(record.confidence) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div
            v-if="hasMore"
            class="flex justify-center pt-4"
          >
            <button
              class="px-5 py-2 rounded-full border border-gray-200 bg-white text-xs text-gray-700 hover:bg-gray-900 hover:text-white transition-colors"
              @click="loadMore"
            >
              加载更多
            </button>
          </div>
        </div>
      </div>

      <!-- 空状态 -->
      <div v-else class="text-center py-12">
        <i class="bi bi-heart text-5xl text-gray-300 mb-4"></i>
        <p class="text-gray-500">还没有任何翻译被收藏</p>
        <p class="mt-1 text-xs text-gray-400">
          在图片翻译或视频翻译结果处点击小爱心即可收藏
        </p>
        <div class="mt-6 flex items-center justify-center gap-3">
          <NuxtLink
            to="/translate"
            class="px-6 py-2 rounded-full text-sm font-medium border border-gray-300 bg-white text-gray-900 shadow-[0_10px_30px_rgba(0,0,0,0.06)] hover:bg-gray-900 hover:text-white transition-colors"
          >
            去图片翻译
          </NuxtLink>
          <NuxtLink
            to="/video-translate"
            class="px-6 py-2 rounded-full text-sm font-medium border border-gray-300 text-gray-700 bg-transparent hover:bg-gray-100 transition-colors"
          >
            去视频翻译
          </NuxtLink>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({ title: '我的收藏 - 译手 HandTalk AI' })

const recognitionStore = useRecognitionStore()
const router = useRouter()

type FavoriteItem = (typeof recognitionStore.favoriteHistory)[number]

const favoriteHistory = computed(() => recognitionStore.favoriteHistory)

const activeFilter = ref('all')
const filters = [
  { label: '全部', value: 'all' },
  { label: '视频', value: 'upload_video' },
  { label: '图片', value: 'upload_image' },
]

const filteredFavorites = computed(() => {
  if (activeFilter.value === 'all') return favoriteHistory.value
  return favoriteHistory.value.filter((item) => item.type === activeFilter.value)
})

const pageSize = 12
const visibleCount = ref(pageSize)

const groupedFavorites = computed(() => {
  const groups: Record<string, typeof favoriteHistory.value> = {}
  for (const item of filteredFavorites.value) {
    const date = new Date(item.createdAt)
    const key = date.toISOString().slice(0, 10)
    if (!groups[key]) groups[key] = []
    groups[key].push(item)
  }

  const sortedKeys = Object.keys(groups).sort((a, b) =>
    a < b ? 1 : a > b ? -1 : 0,
  )

  const flat: any[] = []
  for (const key of sortedKeys) {
    const [year, month, day] = key.split('-')
    const label = `${Number(month)}月${Number(day)}日`
    const source = groups[key] ?? []
    const items = [...source].sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    )

    for (const item of items) {
      flat.push({ groupDate: key, groupLabel: label, item })
    }
  }

  const slice = flat.slice(0, visibleCount.value)
  const map: Record<string, { date: string; dateLabel: string; items: any[] }> = {}

  for (const row of slice) {
    if (!map[row.groupDate]) {
      map[row.groupDate] = {
        date: row.groupDate,
        dateLabel: row.groupLabel,
        items: [],
      }
    }
    map[row.groupDate]!.items.push(row.item)
  }

  return Object.values(map).sort((a, b) => (a.date < b.date ? 1 : -1))
})

const hasMore = computed(() => filteredFavorites.value.length > visibleCount.value)

function loadMore() {
  visibleCount.value += pageSize
}

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function handleRecordClick(record: FavoriteItem) {
  if (record.type === 'upload_video') {
    router.push({
      path: '/video-translate',
      query: {
        historyId: record.id,
      },
    })
    return
  }

  if (record.type === 'upload_image') {
    router.push({
      path: '/translate',
      query: {
        historyId: record.id,
      },
    })
  }
}

onMounted(() => recognitionStore.loadHistory())
</script>

