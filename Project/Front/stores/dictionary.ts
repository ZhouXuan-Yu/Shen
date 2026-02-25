import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// 类型定义
export interface Word {
  id: string
  chinese: string
  pinyin: string
  meaning: string
  category: string
  videoUrl?: string
  thumbnailUrl?: string
  example?: string
  gesturePoints?: string[]
  relatedWords?: string[]
  score?: number
  searchType?: string
  createdAt?: string
  updatedAt?: string
}

export interface WordCategory {
  id: string
  name: string
  icon: string
  wordCount: number
}

export const useDictionaryStore = defineStore('dictionary', () => {
  // 状态
  const words = ref<Word[]>([])
  const categories = ref<WordCategory[]>([])
  const searchResults = ref<Word[]>([])
  const currentWord = ref<Word | null>(null)
  const searchQuery = ref('')
  const selectedCategory = ref('all')
  const loading = ref(false)
  const searchLoading = ref(false)
  const searchStats = ref({ total: 0, tookMs: 0 })

  // 计算属性
  const filteredWords = computed(() => {
    let result = words.value

    if (selectedCategory.value !== 'all') {
      result = result.filter(w => w.category === selectedCategory.value)
    }

    if (searchQuery.value.trim()) {
      const q = searchQuery.value.toLowerCase()
      result = result.filter(w =>
        w.chinese.includes(q) ||
        w.pinyin.toLowerCase().includes(q) ||
        w.meaning.includes(q)
      )
    }

    return result
  })

  const wordCount = computed(() => words.value.length)

  // API 基础地址
  const apiBase = computed(() => {
    if (typeof window !== 'undefined') {
      return (window as any).ENV?.API_BASE_URL || 'http://localhost:8000'
    }
    return 'http://localhost:8000'
  })

  // 方法
  async function fetchWords() {
    loading.value = true
    try {
      const response = await fetch(`${apiBase.value}/api/v1/dictionary/?page=1&limit=100`)
      const data = await response.json()
      
      if (data.code === 200) {
        words.value = data.data.items
      }
    } catch (error) {
      console.error('获取词汇列表失败:', error)
    } finally {
      loading.value = false
    }
  }

  async function fetchCategories() {
    try {
      const response = await fetch(`${apiBase.value}/api/v1/dictionary/categories/list`)
      const data = await response.json()
      
      if (data.code === 200) {
        categories.value = data.data
      }
    } catch (error) {
      console.error('获取分类列表失败:', error)
    }
  }

  async function searchWords(query: string, searchType = 'hybrid', category?: string) {
    searchLoading.value = true
    searchQuery.value = query
    
    try {
      const params = new URLSearchParams({
        query,
        search_type: searchType,
        top_k: '20'
      })
      if (category && category !== 'all') {
        params.append('category', category)
      }

      const response = await fetch(`${apiBase.value}/api/v1/dictionary/search?${params}`)
      const data = await response.json()
      
      if (data.code === 200) {
        searchResults.value = data.data.results.map((r: any) => ({
          id: r.id,
          chinese: r.chinese,
          pinyin: r.pinyin,
          meaning: r.meaning,
          category: r.category,
          videoUrl: r.video_url,
          thumbnailUrl: r.thumbnail_url,
          example: r.example,
          score: r.score,
          searchType: data.data.search_type
        }))
        searchStats.value = {
          total: data.data.total,
          tookMs: data.data.took_ms
        }
      }
    } catch (error) {
      console.error('搜索失败:', error)
    } finally {
      searchLoading.value = false
    }
  }

  async function getWordById(id: string) {
    try {
      const response = await fetch(`${apiBase.value}/api/v1/dictionary/${id}`)
      const data = await response.json()
      
      if (data.code === 200) {
        const w = data.data
        currentWord.value = {
          id: w.id,
          chinese: w.chinese,
          pinyin: w.pinyin,
          meaning: w.meaning,
          category: w.category,
          videoUrl: w.video_url,
          thumbnailUrl: w.thumbnail_url,
          example: w.example,
          createdAt: w.created_at,
          updatedAt: w.updated_at
        }
        return currentWord.value
      }
      return null
    } catch (error) {
      console.error('获取词汇详情失败:', error)
      return null
    }
  }

  function setSearchQuery(query: string) {
    searchQuery.value = query
  }

  function setCategory(category: string) {
    selectedCategory.value = category
  }

  function clearFilters() {
    searchQuery.value = ''
    selectedCategory.value = 'all'
    searchResults.value = []
  }

  // 模拟数据
  function initMockData() {
    words.value = [
      { id: '1', chinese: '你好', pinyin: 'nǐ hǎo', meaning: '用于问候别人', category: '日常问候', example: '你好，我叫小明。', createdAt: '', updatedAt: '' },
      { id: '2', chinese: '谢谢', pinyin: 'xiè xiè', meaning: '表示感激', category: '日常问候', example: '谢谢你！', createdAt: '', updatedAt: '' },
      { id: '3', chinese: '再见', pinyin: 'zài jiàn', meaning: '告别用语', category: '日常问候', example: '再见！', createdAt: '', updatedAt: '' },
    ]

    categories.value = [
      { id: 'all', name: '全部', icon: 'grid', wordCount: words.value.length },
      { id: 'daily', name: '日常问候', icon: 'chat', wordCount: 3 },
      { id: 'number', name: '数字', icon: 'hash', wordCount: 0 },
      { id: 'family', name: '家庭', icon: 'users', wordCount: 0 },
    ]
  }

  return {
    words,
    categories,
    searchResults,
    currentWord,
    searchQuery,
    selectedCategory,
    loading,
    searchLoading,
    searchStats,
    filteredWords,
    wordCount,
    fetchWords,
    fetchCategories,
    searchWords,
    getWordById,
    setSearchQuery,
    setCategory,
    clearFilters,
    initMockData,
  }
})
