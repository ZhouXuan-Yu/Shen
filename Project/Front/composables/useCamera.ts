import { ref, onMounted, onUnmounted } from 'vue'

/**
 * 摄像头控制 hook
 */
export function useCamera() {
  const videoStream = ref<MediaStream | null>(null)
  const videoRef = ref<HTMLVideoElement | null>(null)
  const canvasRef = ref<HTMLCanvasElement | null>(null)

  const isMirrored = ref(true)
  const hasPermission = ref(false)
  const error = ref<string | null>(null)
  const isStreaming = ref(false)

  // 启动摄像头
  async function startCamera(videoElement?: HTMLVideoElement) {
    if (videoStream.value) {
      return videoStream.value
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })

      videoStream.value = stream
      hasPermission.value = true
      error.value = null

      // 如果传入了 videoElement，绑定流
      if (videoElement) {
        videoRef.value = videoElement
        videoElement.srcObject = stream
        await videoElement.play()
        isStreaming.value = true
      }

      return stream
    } catch (e) {
      const err = e as Error
      error.value = err.message || '无法访问摄像头'
      hasPermission.value = false
      console.error('启动摄像头失败:', e)
      return null
    }
  }

  // 停止摄像头
  function stopCamera() {
    if (videoStream.value) {
      videoStream.value.getTracks().forEach(track => track.stop())
      videoStream.value = null
      isStreaming.value = false
    }
  }

  // 切换镜像
  function toggleMirror() {
    isMirrored.value = !isMirrored.value
  }

  // 捕获当前帧
  function captureFrame(): string | null {
    if (!videoRef.value || !canvasRef.value) {
      return null
    }

    const video = videoRef.value
    const canvas = canvasRef.value
    const ctx = canvas.getContext('2d')

    if (!ctx) {
      return null
    }

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // 如果镜像，水平翻转绘制
    if (isMirrored.value) {
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)
    }

    ctx.drawImage(video, 0, 0)

    return canvas.toDataURL('image/jpeg', 0.8)
  }

  // 获取视频元素引用
  function setVideoRef(el: HTMLVideoElement | null) {
    if (el && videoStream.value) {
      videoRef.value = el
      el.srcObject = videoStream.value
    }
  }

  // 清理
  onUnmounted(() => {
    stopCamera()
  })

  return {
    videoStream,
    videoRef,
    canvasRef,
    isMirrored,
    hasPermission,
    error,
    isStreaming,
    startCamera,
    stopCamera,
    toggleMirror,
    captureFrame,
    setVideoRef,
  }
}


