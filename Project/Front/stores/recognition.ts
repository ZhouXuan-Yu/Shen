import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { RecognitionResult, HistoryRecord } from '~/types'

export const useRecognitionStore = defineStore('recognition', () => {
  // 状态
  const currentResult = ref<RecognitionResult | null>(null)
  const results = ref<RecognitionResult[]>([])
  const history = ref<HistoryRecord[]>([])
  const isRecording = ref(false)
  const confidence = ref(0)
  const loading = ref(false)

  // 计算属性
  const hasResult = computed(() => !!currentResult.value)
  const latestResult = computed(() => results.value[results.value.length - 1] || null)

  // 方法
  function setResult(result: RecognitionResult) {
    currentResult.value = result
    results.value.push(result)

    // 限制保存的结果数量
    if (results.value.length > 100) {
      results.value.shift()
    }
  }

  function clearResults() {
    currentResult.value = null
    results.value = []
  }

  function setRecording(recording: boolean) {
    isRecording.value = recording
  }

  function setConfidence(value: number) {
    confidence.value = Math.max(0, Math.min(100, value))
  }

  function addToHistory(record: HistoryRecord) {
    history.value.unshift(record)

    // 限制保存的历史数量
    if (history.value.length > 50) {
      history.value.pop()
    }

    // 保存到本地存储
    localStorage.setItem('recognitionHistory', JSON.stringify(history.value))
  }

  function loadHistory() {
    const saved = localStorage.getItem('recognitionHistory')
    if (saved) {
      try {
        history.value = JSON.parse(saved)
      } catch {
        history.value = []
      }
    }
  }

  function clearHistory() {
    history.value = []
    localStorage.removeItem('recognitionHistory')
  }

  function setLoading(value: boolean) {
    loading.value = value
  }

  return {
    // 状态
    currentResult,
    results,
    history,
    isRecording,
    confidence,
    loading,
    // 计算属性
    hasResult,
    latestResult,
    // 方法
    setResult,
    clearResults,
    setRecording,
    setConfidence,
    addToHistory,
    loadHistory,
    clearHistory,
    setLoading,
  }
})


