import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface VideoResult {
  id: string
  sentence: string
  gloss?: string
  note?: string
  split: string
  videoPath: string
  videoAbsPath: string
  videoUrl: string
  similarity: number
  rank: number
}

export interface SearchStats {
  total: number
  tookMs: number
}

export const useVideoRagStore = defineStore('videoRag', () => {
  const results = ref<VideoResult[]>([])
  const randomSamples = ref<VideoResult[]>([])
  const query = ref('')
  const loading = ref(false)
  const randomLoading = ref(false)
  const error = ref<string | null>(null)
  const stats = ref<SearchStats>({ total: 0, tookMs: 0 })

  const hasResults = computed(() => results.value.length > 0)
  const hasRandom = computed(() => randomSamples.value.length > 0)

  const apiBase = computed(() => {
    if (typeof window !== 'undefined') {
      const fromWindow = (window as any).ENV?.API_BASE_URL
      if (fromWindow) return fromWindow

      const nuxtConfig = useRuntimeConfig?.()
      if (nuxtConfig?.public?.apiBase) {
        return nuxtConfig.public.apiBase.replace(/\/$/, '')
      }
    }
    return 'http://localhost:9000/api/v1'
  })

  async function searchVideos(text: string, topK = 10) {
    query.value = text
    if (!text.trim()) {
      results.value = []
      stats.value = { total: 0, tookMs: 0 }
      return
    }

    loading.value = true
    error.value = null

    try {
      const params = new URLSearchParams({
        query: text,
        top_k: String(topK),
      })
      const resp = await fetch(`${apiBase.value}/video_rag/search?${params}`)
      const data = await resp.json()

      if (data.code === 200) {
        results.value = (data.data.results || []).map((r: any) => ({
          id: r.id,
          sentence: r.sentence,
          gloss: r.gloss,
          note: r.note,
          split: r.split,
          videoPath: r.videoPath,
          videoAbsPath: r.videoAbsPath,
          videoUrl: r.videoUrl,
          similarity: r.similarity,
          rank: r.rank,
        }))
        stats.value = {
          total: data.data.total,
          tookMs: data.data.took_ms,
        }
      } else {
        error.value = data.message || '检索失败'
      }
    } catch (e: any) {
      console.error('视频检索失败:', e)
      error.value = e?.message ?? String(e)
    } finally {
      loading.value = false
    }
  }

  async function fetchRandomSamples(limit = 8) {
    randomLoading.value = true
    error.value = null

    try {
      const params = new URLSearchParams({
        limit: String(limit),
      })
      const resp = await fetch(`${apiBase.value}/video_rag/random?${params}`)
      const data = await resp.json()

      if (data.code === 200) {
        randomSamples.value = (data.data.results || []).map((r: any) => ({
          id: r.id,
          sentence: r.sentence,
          gloss: r.gloss,
          note: r.note,
          split: r.split,
          videoPath: r.videoPath,
          videoAbsPath: r.videoAbsPath,
          videoUrl: r.videoUrl,
          similarity: r.similarity,
          rank: r.rank,
        }))
      } else {
        error.value = data.message || '获取随机推荐失败'
      }
    } catch (e: any) {
      console.error('获取随机视频失败:', e)
      error.value = e?.message ?? String(e)
    } finally {
      randomLoading.value = false
    }
  }

  function clearResults() {
    results.value = []
    stats.value = { total: 0, tookMs: 0 }
  }

  return {
    results,
    randomSamples,
    query,
    loading,
    randomLoading,
    error,
    stats,
    hasResults,
    hasRandom,
    apiBase,
    searchVideos,
    fetchRandomSamples,
    clearResults,
  }
})

