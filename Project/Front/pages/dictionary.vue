<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <div class="max-w-5xl mx-auto px-4 py-8">
      <!-- 页面标题 -->
      <div class="flex items-start justify-between gap-4 mb-6">
        <div class="flex items-center gap-3">
          <NuxtLink to="/" class="flex items-center gap-2 text-gray-600 hover:text-primary-900 transition-colors pt-1">
            <i class="bi bi-arrow-left text-lg"></i>
            <span class="hidden sm:inline">返回</span>
          </NuxtLink>
          <div class="h-6 w-px bg-gray-200"></div>
          <div>
            <h1 class="text-xl font-semibold text-primary-900">视频检索（CE-CSL）</h1>
            <p class="text-sm text-gray-500 mt-0.5">输入中文句子，检索最接近的手语句子视频</p>
          </div>
        </div>
      </div>

      <!-- 检索卡片 -->
      <div class="p-5 bg-white rounded-2xl shadow-sm border border-gray-100">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-10 h-10 rounded-full bg-accent/10 flex items-center justify-center">
            <i class="bi bi-film text-accent text-lg"></i>
          </div>
          <div class="min-w-0">
            <p class="text-sm font-semibold text-primary-900">句子 → 视频检索</p>
            <p class="text-xs text-gray-400">支持 Enter 快捷检索；结果可点击预览</p>
          </div>
        </div>

          <!-- 句子检索输入框 -->
          <div class="relative">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <i class="bi bi-text-paragraph text-gray-400"></i>
            </div>
            <input
              type="text"
              class="w-full pl-10 pr-24 py-3 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:ring-2 focus:ring-accent/20 focus:bg-white focus:border-accent/40 transition-all"
              placeholder="例如：我今天去学校上课"
              v-model="videoQuery"
              @keyup.enter="handleVideoSearch"
            />
            <div class="absolute inset-y-0 right-0 flex items-center pr-2">
              <button
                v-if="videoQuery"
                class="p-1 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-200 transition-colors mr-1"
                @click="clearVideoQuery"
              >
                <i class="bi bi-x-circle"></i>
              </button>
              <button
                class="btn btn-accent btn-sm rounded-lg px-4"
                :disabled="!videoQuery.trim() || videoRagStore.loading"
                @click="handleVideoSearch"
              >
                <span v-if="videoRagStore.loading" class="spinner-border spinner-border-sm mr-1"></span>
                检索视频
              </button>
            </div>
          </div>

          <!-- 检索统计 / 状态 -->
          <div v-if="videoRagStore.query && videoRagStore.stats.total > 0" class="mt-3 text-xs text-gray-500">
            找到 <span class="font-semibold text-primary-900">{{ videoRagStore.stats.total }}</span> 个视频
            <span class="mx-1">·</span>
            用时 {{ videoRagStore.stats.tookMs }}ms
          </div>
          <div v-else-if="videoRagStore.query && !videoRagStore.loading" class="mt-3 text-xs text-gray-400">
            未找到与 "<span class="font-semibold text-primary-900">{{ videoRagStore.query }}</span>" 相关的视频
          </div>

          <!-- 错误提示 -->
          <div v-if="videoRagStore.error" class="mt-3 text-xs text-red-500">
            {{ videoRagStore.error }}
          </div>

        <!-- 预览 + 列表 -->
        <div class="mt-5 grid lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] gap-5">
          <!-- 左侧：大预览 -->
          <div class="rounded-2xl border border-gray-100 overflow-hidden bg-white">
            <div class="aspect-video bg-gray-900">
              <video
                v-if="activeItem?.videoUrl"
                :src="resolveVideoUrl(activeItem.videoUrl)"
                controls
                preload="metadata"
                class="w-full h-full"
              ></video>
              <div v-else class="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
                <div class="text-center">
                  <i class="bi bi-play-circle text-6xl text-white/70"></i>
                  <p class="text-sm text-white/70 mt-2">选择右侧结果进行预览</p>
                </div>
              </div>
            </div>
            <div class="p-4">
              <div class="flex items-center justify-between gap-3">
                <div class="min-w-0">
                  <p class="text-sm font-semibold text-primary-900 line-clamp-2">
                    {{ activeItem?.sentence || '暂无预览' }}
                  </p>
                  <p v-if="activeItem?.gloss" class="text-xs text-gray-500 line-clamp-1 mt-1">
                    {{ activeItem.gloss }}
                  </p>
                  <p v-if="activeItem?.videoPath" class="text-[11px] text-gray-400 truncate mt-1">
                    {{ activeItem.videoPath }}
                  </p>
                </div>
                <div v-if="activeItem && videoRagStore.hasResults" class="flex-shrink-0 text-right">
                  <p class="text-xs text-gray-400">Top {{ activeItem.rank }} · {{ activeItem.split }}</p>
                  <span
                    class="inline-block mt-1 text-xs font-medium px-2 py-0.5 rounded-full"
                    :class="getSimilarityBadgeClass(activeItem.similarity)"
                  >
                    匹配度 {{ Math.round(activeItem.similarity * 100) }}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- 右侧：列表 -->
          <div class="rounded-2xl border border-gray-100 bg-white overflow-hidden">
            <div class="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <div class="flex items-center gap-2 min-w-0">
                <i :class="videoRagStore.hasResults ? 'bi bi-search' : 'bi bi-stars'" class="text-gray-400"></i>
                <p class="text-sm font-medium text-gray-600 truncate">
                  {{ videoRagStore.hasResults ? '检索结果' : '推荐示例（随机）' }}
                </p>
              </div>
              <button
                v-if="!videoRagStore.hasResults"
                class="text-xs text-gray-400 hover:text-primary-900 flex items-center gap-1"
                @click="refreshRandomVideos"
                :disabled="videoRagStore.randomLoading"
              >
                <i class="bi bi-arrow-clockwise" :class="videoRagStore.randomLoading ? 'animate-spin' : ''"></i>
                换一批
              </button>
            </div>

            <div class="max-h-[520px] overflow-y-auto p-3 space-y-2">
              <div
                v-if="videoRagStore.randomLoading && !videoRagStore.hasResults"
                class="py-10 text-center text-xs text-gray-400"
              >
                加载推荐中...
              </div>

              <button
                v-for="item in listItems"
                :key="item.id + '-' + item.rank"
                type="button"
                class="w-full text-left flex gap-3 p-3 rounded-xl transition-colors"
                :class="activeItem?.id === item.id ? 'bg-accent/10' : 'bg-gray-50 hover:bg-gray-100'"
                @click="setActive(item)"
              >
                <div class="relative w-24 h-14 rounded-lg bg-gray-900 overflow-hidden flex-shrink-0">
                  <video
                    v-if="item.videoUrl"
                    :src="resolveVideoUrl(item.videoUrl)"
                    class="w-full h-full object-cover"
                    preload="metadata"
                    muted
                  ></video>
                  <div v-else class="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-700">
                    <i class="bi bi-play-circle text-xl text-white/80"></i>
                  </div>
                  <div class="absolute inset-0 flex items-center justify-center bg-black/10">
                    <div class="w-8 h-8 rounded-full bg-white/90 flex items-center justify-center">
                      <i class="bi bi-play-fill text-primary-900 ml-0.5"></i>
                    </div>
                  </div>
                </div>

                <div class="flex-1 min-w-0">
                  <div class="flex items-center justify-between gap-2 mb-1">
                    <span class="text-xs text-gray-400">
                      {{ videoRagStore.hasResults ? `Top ${item.rank} · ${item.split}` : item.split }}
                    </span>
                    <span
                      v-if="videoRagStore.hasResults"
                      class="text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0"
                      :class="getSimilarityBadgeClass(item.similarity)"
                    >
                      {{ Math.round(item.similarity * 100) }}%
                    </span>
                  </div>
                  <p class="text-sm text-primary-900 line-clamp-2 mb-1">
                    {{ item.sentence }}
                  </p>
                  <p v-if="item.gloss" class="text-[11px] text-gray-500 line-clamp-1 mb-0.5">
                    {{ item.gloss }}
                  </p>
                  <p class="text-[11px] text-gray-400 truncate">
                    {{ item.videoPath }}
                  </p>
                </div>
              </button>

              <div v-if="listItems.length === 0 && !videoRagStore.randomLoading" class="py-10 text-center text-xs text-gray-400">
                暂无可展示的视频
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { VideoResult } from '~/stores/videoRag'

// SEO 元信息
useSeoMeta({
  title: '视频检索（CE-CSL）- 译手 HandTalk AI',
  description: '输入中文句子，检索最接近的手语句子视频（CE-CSL）',
})

// 统一处理视频 URL，避免走前端 3000 端口导致 404
function resolveVideoUrl(raw: string): string {
  if (!raw) return raw

  // 如果后端已经返回了完整的 http/https URL，直接使用
  if (/^https?:\/\//i.test(raw)) {
    return raw
  }

  // 去掉开头的多余斜杠，避免重复
  const cleaned = raw.replace(/^\/+/, '')

  // 只要包含 cecsl/video，一律走后端 9000 端口（不带 /api/v1 前缀）
  if (cleaned.toLowerCase().startsWith('cecsl/video')) {
    return `http://localhost:9000/${cleaned}`
  }

  // 其他相对路径也按后端 9000 端口处理，防止被 Nuxt 当成页面路由
  return `http://localhost:9000/${cleaned}`
}

// 文本 → 视频 检索
const videoQuery = ref('')
const activeItem = ref<VideoResult | null>(null)

// Composables
const videoRagStore = useVideoRagStore()

const listItems = computed<VideoResult[]>(() => {
  return videoRagStore.hasResults ? videoRagStore.results : videoRagStore.randomSamples
})

function getSimilarityBadgeClass(similarity: number): string {
  if (similarity >= 0.75) return 'bg-emerald-50 text-emerald-700 border border-emerald-100'
  if (similarity >= 0.55) return 'bg-amber-50 text-amber-700 border border-amber-100'
  return 'bg-gray-100 text-gray-600 border border-gray-200'
}

function setActive(item: VideoResult) {
  activeItem.value = item
}

// 文本 → 视频 检索逻辑
async function handleVideoSearch() {
  if (!videoQuery.value.trim()) return
  await videoRagStore.searchVideos(videoQuery.value, 8)
}

function clearVideoQuery() {
  videoQuery.value = ''
  videoRagStore.clearResults()
  const firstRandom = videoRagStore.randomSamples[0]
  activeItem.value = firstRandom ?? null
}

function refreshRandomVideos() {
  videoRagStore.fetchRandomSamples(8)
}

// 初始化
onMounted(() => {
  // 初始化随机视频推荐
  videoRagStore.fetchRandomSamples(8)
})

watch(
  () => [videoRagStore.results.length, videoRagStore.results[0]?.id],
  () => {
    if (videoRagStore.results.length > 0) {
      const first = videoRagStore.results[0]
      activeItem.value = first ?? null
    }
  }
)

watch(
  () => [videoRagStore.hasResults, videoRagStore.randomSamples.length, videoRagStore.randomSamples[0]?.id],
  () => {
    if (!videoRagStore.hasResults && videoRagStore.randomSamples.length > 0 && !activeItem.value) {
      const firstRandom = videoRagStore.randomSamples[0]
      activeItem.value = firstRandom ?? null
    }
  }
)
</script>

<style scoped>
.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
