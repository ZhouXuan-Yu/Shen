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
            共 {{ favoriteHistory.length }} 条
          </span>
        </div>
      </div>
    </div>

    <div class="max-w-6xl mx-auto px-4 py-6">
      <div v-if="groupedFavorites.length > 0" class="space-y-4">
        <div
          v-for="group in groupedFavorites"
          :key="group.date"
          class="space-y-2"
        >
          <div class="text-xs font-medium text-gray-400">
            {{ group.dateLabel }}
          </div>
          <div class="row g-3">
            <div
              v-for="record in group.items"
              :key="record.id"
              class="col-6 col-md-4 col-lg-3"
            >
              <div class="card card-hover h-100 cursor-pointer">
                <!-- 缩略图 -->
                <div
                  class="relative aspect-video bg-gray-100 rounded-t-xl overflow-hidden"
                >
                  <img
                    v-if="record.thumbnail"
                    :src="record.thumbnail"
                    class="w-full h-full object-cover"
                  />
                  <div
                    v-else
                    class="w-full h-full flex items-center justify-center"
                  >
                    <i
                      :class="
                        record.type === 'upload_video'
                          ? 'bi bi-camera-video'
                          : 'bi bi-image'
                      "
                      class="text-3xl text-gray-300"
                    ></i>
                  </div>

                  <!-- 类型标签 -->
                  <span
                    class="absolute bottom-2 left-2 px-2 py-0.5 bg-black/50 rounded text-white text-xs"
                  >
                    {{
                      record.type === 'upload_video'
                        ? '视频'
                        : record.type === 'upload_image'
                          ? '图片'
                          : '其他'
                    }}
                  </span>

                  <!-- 收藏标记 -->
                  <span
                    class="absolute top-2 right-2 w-7 h-7 rounded-full bg-pink-500 flex items-center justify-center shadow-lg"
                  >
                    <i class="bi bi-heart-fill text-white text-xs"></i>
                  </span>
                </div>

                <!-- 文本信息 -->
                <div class="p-3">
                  <h3
                    class="font-bold text-primary-900 line-clamp-2 min-h-[2.5rem]"
                  >
                    {{ record.result }}
                  </h3>
                  <div
                    class="flex items-center justify-between mt-2 text-xs text-gray-500"
                  >
                    <span>{{ formatTime(record.createdAt) }}</span>
                    <span class="text-green-600 font-medium">
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

const favoriteHistory = computed(() => recognitionStore.favoriteHistory)

const pageSize = 12
const visibleCount = ref(pageSize)

const groupedFavorites = computed(() => {
  const groups: Record<string, typeof favoriteHistory.value> = {}
  for (const item of favoriteHistory.value) {
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
  const map: Record<string, { date: string; dateLabel: string; items: any[] }> =
    {}

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

const hasMore = computed(
  () => favoriteHistory.value.length > visibleCount.value,
)

function loadMore() {
  visibleCount.value += pageSize
}

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

