import { ref } from 'vue'

/**
 * 语音播报 hook
 */
export function useSpeech() {
  const isSpeaking = ref(false)
  const isSupported = ref(false)
  const error = ref<string | null>(null)

  let synthesis: SpeechSynthesis | null = null
  let currentUtterance: SpeechSynthesisUtterance | null = null

  onMounted(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      synthesis = window.speechSynthesis
      isSupported.value = true
    }
  })

  function speak(text: string, lang = 'zh-CN') {
    if (!synthesis) {
      error.value = '浏览器不支持语音合成'
      return
    }

    // 停止当前播放
    synthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    currentUtterance = utterance
    
    utterance.lang = lang
    utterance.rate = 0.9 // 稍微放慢语速
    utterance.pitch = 1

    // 选择中文语音
    const voices = synthesis.getVoices()
    const chineseVoice = voices.find(voice => voice.lang.includes('zh'))
    if (chineseVoice) {
      utterance.voice = chineseVoice
    }

    utterance.onstart = () => {
      isSpeaking.value = true
      error.value = null
    }

    utterance.onerror = () => {
      isSpeaking.value = false
      error.value = '语音播放失败'
    }

    utterance.onend = () => {
      isSpeaking.value = false
    }

    synthesis.speak(utterance)
  }

  function stop() {
    if (synthesis) {
      synthesis.cancel()
      isSpeaking.value = false
    }
  }

  function pause() {
    if (synthesis) {
      synthesis.pause()
      isSpeaking.value = false
    }
  }

  function resume() {
    if (synthesis) {
      synthesis.resume()
      isSpeaking.value = true
    }
  }

  return {
    isSpeaking,
    isSupported,
    error,
    speak,
    stop,
    pause,
    resume,
  }
}
