<template>
  <div
    class="word-card card card-hover cursor-pointer h-full border-0 shadow-sm"
    :class="{ 'list-mode': viewMode === 'list' }"
    @click="$emit('click', word)"
  >
    <!-- 网格模式 -->
    <template v-if="viewMode === 'grid'">
      <!-- 缩略图 -->
      <div class="relative aspect-video bg-gray-100 rounded-t-xl overflow-hidden">
        <img
          v-if="word.thumbnailUrl"
          :src="word.thumbnailUrl"
          :alt="word.chinese"
          class="w-full h-full object-cover transition-transform duration-300 hover:scale-110"
        />
        <div v-else class="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
          <i class="bi bi-play-circle text-5xl text-gray-300"></i>
        </div>
        
        <!-- 收藏标记 -->
        <div 
          v-if="isFavorite"
          class="absolute top-2 left-2 w-7 h-7 rounded-full bg-pink-500 flex items-center justify-center shadow-lg"
        >
          <i class="bi bi-heart-fill text-white text-xs"></i>
        </div>

        <!-- 置信度标签 -->
        <div v-if="word.score" class="absolute top-2 right-2">
          <span class="badge bg-primary-900/80 text-white text-xs backdrop-blur-sm">
            {{ word.score.toFixed(1) }}
          </span>
        </div>

        <!-- 播放图标遮罩 -->
        <div class="absolute inset-0 bg-black/0 hover:bg-black/20 transition-colors flex items-center justify-center">
          <div class="w-12 h-12 rounded-full bg-white/90 flex items-center justify-center opacity-0 hover:opacity-100 transition-all transform scale-90 hover:scale-100 shadow-lg">
            <i class="bi bi-play-fill text-primary-900 text-lg ml-1"></i>
          </div>
        </div>
      </div>

      <!-- 词汇信息 -->
      <div class="p-4">
        <h3 class="font-bold text-primary-900 mb-1 truncate">{{ word.chinese }}</h3>
        <p class="text-sm text-gray-500 mb-2 truncate">{{ word.pinyin }}</p>
        <p v-if="word.meaning" class="text-xs text-gray-400 line-clamp-2">
          {{ word.meaning }}
        </p>
      </div>
    </template>

    <!-- 列表模式 -->
    <template v-else>
      <div class="flex items-center gap-4 p-4">
        <!-- 缩略图 -->
        <div class="relative w-20 h-20 flex-shrink-0 rounded-xl overflow-hidden bg-gray-100">
          <img
            v-if="word.thumbnailUrl"
            :src="word.thumbnailUrl"
            :alt="word.chinese"
            class="w-full h-full object-cover"
          />
          <div v-else class="w-full h-full flex items-center justify-center">
            <i class="bi bi-play-circle text-2xl text-gray-300"></i>
          </div>

          <!-- 收藏标记 -->
          <div 
            v-if="isFavorite"
            class="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-pink-500 flex items-center justify-center"
          >
            <i class="bi bi-heart-fill text-white text-[10px]"></i>
          </div>
        </div>

        <!-- 词汇信息 -->
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 mb-1">
            <h3 class="font-bold text-primary-900 truncate">{{ word.chinese }}</h3>
            <span v-if="word.category" class="badge badge-outline text-xs px-2 py-0.5 rounded-full flex-shrink-0">
              {{ word.category }}
            </span>
          </div>
          <p class="text-sm text-gray-500 mb-1">{{ word.pinyin }}</p>
          <p v-if="word.meaning" class="text-sm text-gray-400 truncate">
            {{ word.meaning }}
          </p>
        </div>

        <!-- 箭头图标 -->
        <div class="flex-shrink-0">
          <i class="bi bi-chevron-right text-gray-300"></i>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { Word } from '~/stores/dictionary'

// 定义 props
interface Props {
  word: Word
  viewMode?: 'grid' | 'list'
}

const props = withDefaults(defineProps<Props>(), {
  viewMode: 'grid'
})

defineEmits<{
  click: [word: Word]
}>()

// 收藏状态
const favorites = ref<string[]>([])

const isFavorite = computed(() => {
  return favorites.value.includes(props.word.id)
})

// 加载收藏状态
onMounted(() => {
  const savedFavorites = localStorage.getItem('wordFavorites')
  if (savedFavorites) {
    try {
      favorites.value = JSON.parse(savedFavorites)
    } catch {
      favorites.value = []
    }
  }
})
</script>

<style scoped>
.word-card {
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}

.word-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.word-card:hover::before {
  transform: scaleX(1);
}

.list-mode {
  transition: all 0.2s ease;
}

.list-mode:hover {
  transform: translateX(4px);
  box-shadow: -4px 0 12px rgba(0, 0, 0, 0.1);
}
</style>

