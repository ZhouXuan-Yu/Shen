import { ref, onMounted, onUnmounted } from 'vue'

/**
 * WebSocket 实时识别 hook
 */
export function useRecognitionWebSocket() {
  const config = useRuntimeConfig()
  const recognitionStore = useRecognitionStore()

  const ws = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const error = ref<string | null>(null)

  // 连接 WebSocket
  function connect() {
    if (ws.value?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      ws.value = new WebSocket(`${config.public.wsUrl}/recognize`)

      ws.value.onopen = () => {
        isConnected.value = true
        error.value = null
        console.log('WebSocket 连接成功')
      }

      ws.value.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'result') {
            const result = {
              text: data.data.text,
              pinyin: data.data.pinyin,
              meaning: data.data.meaning,
              confidence: data.data.confidence,
              timestamp: new Date().toISOString(),
            }

            recognitionStore.setResult(result)
            recognitionStore.setConfidence(data.data.confidence)
          }
        } catch (e) {
          console.error('解析 WebSocket 消息失败:', e)
        }
      }

      ws.value.onerror = (e) => {
        error.value = 'WebSocket 连接错误'
        console.error('WebSocket 错误:', e)
      }

      ws.value.onclose = () => {
        isConnected.value = false
        console.log('WebSocket 连接关闭')
      }
    } catch (e) {
      error.value = '连接失败'
      console.error('创建 WebSocket 失败:', e)
    }
  }

  // 发送帧数据
  function sendFrame(imageData: string) {
    if (ws.value?.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify({
        type: 'frame',
        data: {
          image: imageData,
          timestamp: Date.now(),
        },
      }))
    }
  }

  // 断开连接
  function disconnect() {
    ws.value?.close()
    ws.value = null
    isConnected.value = false
  }

  // 清理
  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected,
    error,
    connect,
    disconnect,
    sendFrame,
  }
}


