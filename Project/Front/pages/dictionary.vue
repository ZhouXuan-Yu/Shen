<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <!-- 顶部导航栏 -->
    <div class="sticky top-16 z-30 bg-white/80 backdrop-blur-lg border-b border-gray-100">
      <div class="max-w-7xl mx-auto px-4">
        <!-- 标题栏 -->
        <div class="flex items-center justify-between py-3">
          <div class="flex items-center gap-3">
            <NuxtLink to="/" class="flex items-center gap-2 text-gray-600 hover:text-primary-900 transition-colors">
              <i class="bi bi-arrow-left text-lg"></i>
              <span class="hidden sm:inline">返回</span>
            </NuxtLink>
            <div class="h-6 w-px bg-gray-200"></div>
            <h1 class="text-lg font-semibold text-primary-900">手语词典</h1>
          </div>
          
          <!-- 收藏按钮 -->
          <button 
            class="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-pink-50 text-pink-500 hover:bg-pink-100 transition-colors"
            @click="showFavoritesOnly = !showFavoritesOnly"
          >
            <i class="bi bi-heart-fill"></i>
            <span class="hidden sm:inline text-sm">我的收藏</span>
          </button>
        </div>

        <!-- 搜索栏 -->
        <div class="relative mb-4">
          <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <i class="bi bi-search text-gray-400"></i>
          </div>
          <input
            type="text"
            class="w-full pl-11 pr-24 py-3 bg-gray-100 border-0 rounded-xl text-base focus:ring-2 focus:ring-accent/20 focus:bg-white transition-all"
            placeholder="搜索手语词汇（支持中文、拼音、同义词）..."
            v-model="searchKeyword"
            @keyup.enter="handleSearch"
          />
          <div class="absolute inset-y-0 right-0 flex items-center pr-2">
            <button 
              v-if="searchKeyword" 
              class="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-200 transition-colors mr-1"
              @click="searchKeyword = ''"
            >
              <i class="bi bi-x-circle"></i>
            </button>
            <button 
              class="btn btn-accent btn-sm rounded-lg px-4"
              :disabled="!searchKeyword.trim() || dictionaryStore.searchLoading"
              @click="handleSearch"
            >
              <span v-if="dictionaryStore.searchLoading" class="spinner-border spinner-border-sm mr-1"></span>
              搜索
            </button>
          </div>
        </div>

        <!-- 分类标签 -->
        <div class="flex items-center gap-2 overflow-x-auto pb-4 scrollbar-hide">
          <button
            v-for="category in categories"
            :key="category.id"
            class="category-btn flex-shrink-0 px-4 py-2 rounded-full text-sm font-medium transition-all duration-300"
            :class="selectedCategory === category.id 
              ? 'bg-primary-900 text-white shadow-lg shadow-primary-500/30' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'"
            @click="selectCategory(category.id)"
          >
            <i :class="getCategoryIcon(category.id)" class="mr-1.5"></i>
            {{ category.name }}
            <span class="ml-1.5 opacity-60 text-xs">({{ category.wordCount }})</span>
          </button>
        </div>
      </div>
    </div>

    <div class="max-w-7xl mx-auto px-4 py-8">
      <!-- 搜索状态提示 -->
      <Transition name="slide-down">
        <div v-if="dictionaryStore.searchQuery" class="mb-6">
          <div class="flex items-center justify-between p-4 bg-accent/5 rounded-xl border border-accent/10">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-full bg-accent/20 flex items-center justify-center">
                <i class="bi bi-search text-accent"></i>
              </div>
              <div>
                <p class="text-sm text-gray-500">
                  搜索: <span class="font-semibold text-primary-900">"{{ dictionaryStore.searchQuery }}"</span>
                </p>
                <p v-if="dictionaryStore.searchStats.total > 0" class="text-xs text-gray-400">
                  找到 {{ dictionaryStore.searchStats.total }} 个结果 ({{ dictionaryStore.searchStats.tookMs }}ms)
                </p>
              </div>
            </div>
            <button
              class="text-sm text-gray-500 hover:text-primary-900 px-3 py-1.5 rounded-lg hover:bg-white transition-colors"
              @click="clearSearch"
            >
              清除搜索
            </button>
          </div>
        </div>
      </Transition>

      <!-- 加载状态 -->
      <div v-if="dictionaryStore.loading || dictionaryStore.searchLoading" class="text-center py-16">
        <div class="relative w-20 h-20 mx-auto mb-4">
          <svg class="w-20 h-20 transform -rotate-90">
            <circle class="text-gray-200" stroke-width="6" stroke="currentColor" fill="transparent" :r="34" :cx="40" :cy="40" />
            <circle 
              class="text-accent" 
              stroke-width="6" 
              stroke="currentColor" 
              fill="transparent" 
              :r="34" 
              :cx="40" 
              :cy="40" 
              stroke-dasharray="213.63" 
              stroke-dashoffset="213.63"
              style="animation: dash 1.5s ease-in-out infinite; stroke-dashoffset: 50;"
            />
          </svg>
          <div class="absolute inset-0 flex items-center justify-center">
            <i class="bi bi-book text-2xl text-accent"></i>
          </div>
        </div>
        <p class="text-gray-500">{{ dictionaryStore.searchLoading ? '搜索中...' : '加载中...' }}</p>
      </div>

      <!-- 搜索结果 -->
      <div v-else-if="dictionaryStore.searchResults.length > 0">
        <div class="flex items-center justify-between mb-6">
          <h3 class="font-semibold text-primary-900">
            <i class="bi bi-search mr-2 text-accent"></i>
            搜索结果
          </h3>
          <span class="text-sm text-gray-500">{{ dictionaryStore.searchResults.length }} 个词汇</span>
        </div>
        
        <div class="row g-4">
          <div
            v-for="word in dictionaryStore.searchResults"
            :key="word.id"
            class="col-6 col-md-4 col-lg-3"
          >
            <WordCard :word="word" @click="showWordDetail(word)" />
          </div>
        </div>
      </div>

      <!-- 词汇列表 -->
      <div v-else-if="displayedWords.length > 0">
        <!-- 筛选提示 -->
        <div v-if="showFavoritesOnly" class="mb-6 p-4 bg-pink-50 rounded-xl border border-pink-100">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-full bg-pink-100 flex items-center justify-center">
                <i class="bi bi-heart-fill text-pink-500"></i>
              </div>
              <div>
                <p class="font-medium text-primary-900">收藏词汇</p>
                <p class="text-sm text-gray-500">共 {{ displayedWords.length }} 个</p>
              </div>
            </div>
            <button
              class="text-sm text-gray-500 hover:text-primary-900"
              @click="showFavoritesOnly = false"
            >
              取消筛选
            </button>
          </div>
        </div>

        <!-- 词汇统计 -->
        <div class="flex items-center justify-between mb-6">
          <p class="text-sm text-gray-500">
            共找到 <span class="font-semibold text-primary-900">{{ displayedWords.length }}</span> 个词汇
          </p>
          <div class="flex items-center gap-2">
            <button 
              class="p-2 rounded-lg transition-colors"
              :class="viewMode === 'grid' ? 'bg-primary-900 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'"
              @click="viewMode = 'grid'"
            >
              <i class="bi bi-grid"></i>
            </button>
            <button 
              class="p-2 rounded-lg transition-colors"
              :class="viewMode === 'list' ? 'bg-primary-900 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'"
              @click="viewMode = 'list'"
            >
              <i class="bi bi-list-ul"></i>
            </button>
          </div>
        </div>

        <!-- 词汇卡片网格 -->
        <div 
          :class="viewMode === 'grid' ? 'row g-4' : 'flex flex-col gap-3'"
        >
          <div
            v-for="word in displayedWords"
            :key="word.id"
            :class="viewMode === 'grid' ? 'col-6 col-md-4 col-lg-3' : ''"
          >
            <WordCard 
              :word="word" 
              :view-mode="viewMode"
              @click="showWordDetail(word)" 
            />
          </div>
        </div>
      </div>

      <!-- 空状态 -->
      <div v-else class="text-center py-16">
        <div class="w-24 h-24 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
          <i class="bi bi-search text-4xl text-gray-300"></i>
        </div>
        <h3 class="text-lg font-medium text-primary-900 mb-2">未找到相关词汇</h3>
        <p class="text-gray-500 mb-6">试试其他搜索词或清除筛选条件</p>
        <button class="btn btn-secondary rounded-full" @click="clearFilters">
          <i class="bi bi-arrow-clockwise mr-2"></i>
          清除筛选
        </button>
      </div>
    </div>

    <!-- 词汇详情弹窗 -->
    <Teleport to="body">
      <Transition name="fade">
        <div
          v-if="selectedWord"
          class="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4"
        >
          <!-- 遮罩层 -->
          <div
            class="absolute inset-0 bg-black/60 backdrop-blur-sm"
            @click="closeDetail"
          ></div>
          
          <!-- 弹窗内容 -->
          <div class="relative bg-white rounded-t-3xl sm:rounded-3xl w-full max-w-lg max-h-[90vh] overflow-hidden shadow-2xl animate-slide-up">
            <!-- 视频/图片区域 -->
            <div class="relative aspect-video bg-gray-900">
              <video
                v-if="selectedWord.videoUrl"
                :src="selectedWord.videoUrl"
                controls
                class="w-full h-full"
              ></video>
              <img
                v-else-if="selectedWord.thumbnailUrl"
                :src="selectedWord.thumbnailUrl"
                :alt="selectedWord.chinese"
                class="w-full h-full object-cover"
              />
              <div v-else class="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200">
                <i class="bi bi-play-circle text-6xl text-gray-300"></i>
              </div>
              
              <!-- 关闭按钮 -->
              <button
                class="absolute top-4 right-4 w-10 h-10 rounded-full bg-black/50 text-white flex items-center justify-center hover:bg-black/70 transition-colors backdrop-blur-sm"
                @click="closeDetail"
              >
                <i class="bi bi-x-lg"></i>
              </button>

              <!-- 收藏按钮 -->
              <button
                class="absolute top-4 left-4 w-10 h-10 rounded-full flex items-center justify-center transition-all backdrop-blur-sm"
                :class="isFavorite ? 'bg-pink-500 text-white' : 'bg-white/20 text-white hover:bg-white/30'"
                @click="toggleFavorite"
              >
                <i :class="isFavorite ? 'bi bi-heart-fill' : 'bi bi-heart'" class="text-lg"></i>
              </button>
            </div>
            
            <!-- 详情内容 -->
            <div class="p-6 overflow-y-auto max-h-[50vh]">
              <!-- 词汇标题 -->
              <div class="flex items-start justify-between mb-4">
                <div>
                  <h2 class="text-3xl font-bold text-primary-900 mb-1">{{ selectedWord.chinese }}</h2>
                  <p class="text-lg text-gray-500">{{ selectedWord.pinyin }}</p>
                </div>
                <span v-if="selectedWord.category" class="badge badge-primary rounded-full px-3 py-1">
                  {{ selectedWord.category }}
                </span>
              </div>

              <!-- 释义 -->
              <div class="mb-6">
                <label class="text-xs text-gray-500 mb-2 block uppercase tracking-wide">释义</label>
                <p class="text-gray-700 text-lg leading-relaxed">{{ selectedWord.meaning }}</p>
              </div>

              <!-- 用法示例 -->
              <div v-if="selectedWord.example" class="mb-6">
                <label class="text-xs text-gray-500 mb-2 block uppercase tracking-wide">用法示例</label>
                <div class="p-4 bg-blue-50 rounded-xl border border-blue-100">
                  <p class="text-gray-700">{{ selectedWord.example }}</p>
                </div>
              </div>

              <!-- 手势要点 -->
              <div v-if="selectedWord.gesturePoints" class="mb-6">
                <label class="text-xs text-gray-500 mb-2 block uppercase tracking-wide">手势要点</label>
                <div class="flex flex-wrap gap-2">
                  <span 
                    v-for="(point, idx) in selectedWord.gesturePoints" 
                    :key="idx"
                    class="badge bg-gray-100 text-gray-700 px-3 py-1.5 rounded-full"
                  >
                    {{ point }}
                  </span>
                </div>
              </div>

              <!-- 相关词汇 -->
              <div v-if="selectedWord.relatedWords && selectedWord.relatedWords.length > 0" class="mb-6">
                <label class="text-xs text-gray-500 mb-2 block uppercase tracking-wide">相关词汇</label>
                <div class="flex flex-wrap gap-2">
                  <span
                    v-for="word in selectedWord.relatedWords"
                    :key="word"
                    class="badge badge-outline px-3 py-1.5 rounded-full cursor-pointer hover:bg-accent hover:text-white transition-colors"
                  >
                    {{ word }}
                  </span>
                </div>
              </div>

              <!-- 操作按钮 -->
              <div class="flex gap-3 mt-8 pt-4 border-t border-gray-100">
                <button class="btn btn-secondary flex-1 rounded-xl py-3" @click="playVideo">
                  <i class="bi bi-play-circle mr-2"></i>播放示范
                </button>
                <button class="btn btn-secondary flex-1 rounded-xl py-3" @click="copyWord">
                  <i class="bi bi-clipboard mr-2"></i>复制
                </button>
                <button class="btn btn-accent flex-1 rounded-xl py-3" @click="shareWord">
                  <i class="bi bi-share mr-2"></i>分享
                </button>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import type { Word } from '~/types'

// SEO 元信息
useSeoMeta({
  title: '手语词典 - 译手 HandTalk AI',
  description: '常用手语词汇学习，观看标准示范视频',
})

// 状态
const searchKeyword = ref('')
const selectedCategory = ref('all')
const selectedWord = ref<Word | null>(null)
const isFavorite = ref(false)
const favorites = ref<string[]>([])
const showFavoritesOnly = ref(false)
const viewMode = ref<'grid' | 'list'>('grid')

// Composables
const dictionaryStore = useDictionaryStore()
const toast = useToast()

// 计算属性
const categories = computed(() => [
  { id: 'all', name: '全部', icon: 'grid', wordCount: dictionaryStore.wordCount },
  ...dictionaryStore.categories,
])

const displayedWords = computed(() => {
  if (showFavoritesOnly.value) {
    return dictionaryStore.filteredWords.filter(word => favorites.value.includes(word.id))
  }
  return dictionaryStore.filteredWords
})

// 获取分类图标
function getCategoryIcon(categoryId: string): string {
  const icons: Record<string, string> = {
    all: 'bi-grid',
    greeting: 'bi-hand-wave',
    daily: 'bi-cup-hot',
    emotion: 'bi-heart',
    action: 'bi-body-text',
    number: 'bi-123',
    family: 'bi-people',
    work: 'bi-briefcase',
    travel: 'bi-airplane',
  }
  return icons[categoryId] || 'bi-tag'
}

// 方法
async function handleSearch() {
  if (!searchKeyword.value.trim()) return

  const category = selectedCategory.value !== 'all' ? selectedCategory.value : undefined
  await dictionaryStore.searchWords(searchKeyword.value, 'hybrid', category)
}

function selectCategory(categoryId: string) {
  selectedCategory.value = categoryId
  dictionaryStore.setCategory(categoryId)

  if (searchKeyword.value.trim()) {
    dictionaryStore.searchWords(searchKeyword.value, 'hybrid', categoryId !== 'all' ? categoryId : undefined)
  }
}

function clearSearch() {
  searchKeyword.value = ''
  dictionaryStore.clearFilters()
}

function clearFilters() {
  searchKeyword.value = ''
  selectedCategory.value = 'all'
  showFavoritesOnly.value = false
  dictionaryStore.clearFilters()
}

function showWordDetail(word: Word) {
  selectedWord.value = word
  isFavorite.value = favorites.value.includes(word.id)
}

function closeDetail() {
  selectedWord.value = null
}

function toggleFavorite() {
  if (!selectedWord.value) return

  const index = favorites.value.indexOf(selectedWord.value.id)
  if (index > -1) {
    favorites.value.splice(index, 1)
    isFavorite.value = false
    toast.success('已取消收藏')
  } else {
    favorites.value.push(selectedWord.value.id)
    isFavorite.value = true
    toast.success('已添加收藏')
  }

  localStorage.setItem('wordFavorites', JSON.stringify(favorites.value))
}

function playVideo() {
  toast.info('视频播放功能开发中')
}

function copyWord() {
  if (!selectedWord.value) return
  
  const text = `${selectedWord.value.chinese} (${selectedWord.value.pinyin})\n${selectedWord.value.meaning}`
  navigator.clipboard.writeText(text)
  toast.success('已复制到剪贴板')
}

function shareWord() {
  if (!selectedWord.value) return

  const shareData = {
    title: `译手词典 - ${selectedWord.value.chinese}`,
    text: `${selectedWord.value.chinese} (${selectedWord.value.pinyin}) - ${selectedWord.value.meaning}`,
  }

  if (navigator.share) {
    navigator.share(shareData)
  } else {
    navigator.clipboard.writeText(shareData.text)
    toast.success('已复制分享内容')
  }
}

// 初始化
onMounted(() => {
  const savedFavorites = localStorage.getItem('wordFavorites')
  if (savedFavorites) {
    try {
      favorites.value = JSON.parse(savedFavorites)
    } catch {
      favorites.value = []
    }
  }

  dictionaryStore.fetchCategories()
  dictionaryStore.fetchWords()
})
</script>

<style scoped>
@keyframes dash {
  0% {
    stroke-dasharray: 1, 150;
    stroke-dashoffset: 0;
  }
  50% {
    stroke-dasharray: 90, 150;
    stroke-dashoffset: -35;
  }
  100% {
    stroke-dasharray: 90, 150;
    stroke-dashoffset: -124;
  }
}

.animate-spin {
  animation: dash 1.5s ease-in-out infinite;
}

.category-btn {
  position: relative;
  overflow: hidden;
}

.category-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.2), transparent);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.category-btn:hover::before {
  opacity: 1;
}

/* 过渡动画 */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease-out;
}

.slide-down-enter-from,
.slide-down-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
