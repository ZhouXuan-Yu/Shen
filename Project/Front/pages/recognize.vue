<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-900 to-gray-800">
    <!-- 顶部导航 -->
    <div class="fixed top-0 left-0 right-0 z-40 bg-gray-900/80 backdrop-blur-lg border-b border-gray-800">
      <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <NuxtLink to="/" class="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
            <i class="bi bi-arrow-left text-lg"></i>
            <span class="hidden sm:inline">返回</span>
          </NuxtLink>
          <div class="h-6 w-px bg-gray-700"></div>
          <h1 class="text-white font-semibold">实时手语翻译</h1>
        </div>
        <div class="flex items-center gap-2">
          <button 
            class="p-2 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-gray-800"
            @click="toggleFullscreen"
            title="全屏"
          >
            <i class="bi bi-fullscreen"></i>
          </button>
          <button 
            class="p-2 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-gray-800"
            @click="shareResult"
            title="分享"
          >
            <i class="bi bi-share"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- 摄像头区域 -->
    <div class="relative h-[calc(100vh-8rem)] bg-gray-900 overflow-hidden">
      <!-- 视频层 -->
      <video
        ref="videoRef"
        class="absolute inset-0 w-full h-full object-cover"
        :class="{ 'scale-x-[-1]': camera.isMirrored.value }"
        autoplay
        playsinline
        muted
      ></video>

      <!-- 骨架叠加层（Canvas） -->
      <canvas
        ref="skeletonCanvasRef"
        class="absolute inset-0 w-full h-full pointer-events-none"
      ></canvas>

      <!-- 背景装饰 -->
      <div class="absolute inset-0 pointer-events-none">
        <!-- 角落装饰 -->
        <div class="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-white/30 rounded-tl-lg"></div>
        <div class="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-white/30 rounded-tr-lg"></div>
        <div class="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-white/30 rounded-bl-lg"></div>
        <div class="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-white/30 rounded-br-lg"></div>
        
        <!-- 渐变遮罩 -->
        <div class="absolute inset-0 bg-gradient-to-t from-gray-900/80 via-transparent to-gray-900/30"></div>
      </div>

      <!-- 状态指示器 -->
      <div class="absolute top-20 left-4 right-4">
        <!-- 录制状态 -->
        <div v-if="recognitionStore.isRecording" class="flex items-center gap-3 mb-4">
          <div class="flex items-center gap-2 px-3 py-1.5 bg-red-500/20 rounded-full">
            <span class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            <span class="text-red-400 text-sm font-medium">实时识别中</span>
          </div>
          <div class="flex items-center gap-2 px-3 py-1.5 bg-white/10 rounded-full">
            <i class="bi bi-clock text-white/60"></i>
            <span class="text-white/80 text-sm">{{ recordingDuration }}</span>
          </div>
        </div>

        <!-- 置信度指示器 -->
        <div class="bg-gray-900/60 backdrop-blur-lg rounded-2xl p-4">
          <div class="flex items-center gap-3">
            <div class="flex-1">
              <div class="flex items-center justify-between mb-2">
                <span class="text-white/60 text-xs">置信度</span>
                <span class="text-white font-medium">{{ Math.round(recognitionStore.confidence) }}%</span>
              </div>
              <div class="h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  class="h-full rounded-full transition-all duration-300"
                  :class="confidenceClass"
                  :style="{ width: `${recognitionStore.confidence}%` }"
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 加载状态 -->
      <div
        v-if="loading"
        class="absolute inset-0 flex items-center justify-center bg-gray-900/90 backdrop-blur-sm"
      >
        <div class="text-center">
          <div class="relative w-20 h-20 mx-auto mb-4">
            <svg class="w-20 h-20 transform -rotate-90">
              <circle
                class="text-gray-700"
                stroke-width="4"
                stroke="currentColor"
                fill="transparent"
                :r="36"
                :cx="40"
                :cy="40"
              />
              <circle 
                class="text-accent animate-spin"
                stroke-width="4"
                stroke="currentColor"
                fill="transparent"
                :r="36"
                :cx="40"
                :cy="40"
                stroke-dasharray="226.19"
                stroke-dashoffset="50"
                style="animation: dash 1.5s ease-in-out infinite;"
              />
            </svg>
            <div class="absolute inset-0 flex items-center justify-center">
              <i class="bi bi-camera-video text-2xl text-accent"></i>
            </div>
          </div>
          <p class="text-white text-lg mb-2">正在启动摄像头</p>
          <p class="text-gray-400 text-sm">请确保摄像头权限已开启</p>
        </div>
      </div>

      <!-- 错误提示 -->
      <div
        v-if="camera.error.value"
        class="absolute inset-0 flex items-center justify-center bg-gray-900/95 backdrop-blur-sm"
      >
        <div class="text-center px-4 max-w-sm">
          <div class="w-20 h-20 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <i class="bi bi-camera-video-off text-4xl text-red-500"></i>
          </div>
          <p class="text-white text-lg mb-2">摄像头访问失败</p>
          <p class="text-gray-400 text-sm mb-6">{{ camera.error.value }}</p>
          <button class="btn btn-secondary rounded-full" @click="initCamera">
            <i class="bi bi-arrow-clockwise mr-2"></i>
            重新尝试
          </button>
        </div>
      </div>

      <!-- 操作提示 -->
      <div 
        v-if="!recognitionStore.isRecording && !loading && !camera.error.value"
        class="absolute inset-0 flex items-center justify-center"
      >
        <div class="text-center">
          <div class="w-24 h-24 mx-auto mb-4 rounded-full bg-white/10 backdrop-blur-lg flex items-center justify-center">
            <i class="bi bi-play-fill text-4xl text-white"></i>
          </div>
          <p class="text-white/80 text-lg mb-2">点击下方按钮开始识别</p>
          <p class="text-gray-500 text-sm">将手语动作对准摄像头</p>
        </div>
      </div>
    </div>

    <!-- 控制栏 -->
    <div class="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-gray-900 to-gray-900/80 backdrop-blur-xl pb-8 pt-4 px-4">
      <div class="max-w-md mx-auto">
        <div class="flex items-center justify-center gap-4">
          <!-- 镜像切换 -->
          <button
            class="w-12 h-12 rounded-full flex items-center justify-center transition-all"
            :class="camera.isMirrored.value ? 'bg-white text-gray-900' : 'bg-white/10 text-white hover:bg-white/20'"
            @click="camera.toggleMirror()"
            title="镜像切换"
          >
            <i class="bi bi-arrow-left-right"></i>
          </button>

          <!-- 主录制按钮 -->
          <button
            class="w-16 h-16 rounded-full flex items-center justify-center shadow-2xl transition-all transform hover:scale-105"
            :class="recognitionStore.isRecording 
              ? 'bg-red-500 text-white shadow-red-500/30 animate-pulse' 
              : 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-blue-500/30'"
            @click="toggleRecording"
          >
            <i :class="recognitionStore.isRecording ? 'bi bi-stop-fill' : 'bi bi-play-fill'" class="text-2xl"></i>
          </button>

          <!-- 拍照按钮 -->
          <button
            class="w-12 h-12 rounded-full flex items-center justify-center transition-all bg-white/10 text-white hover:bg-white/20"
            @click="captureAndSave"
            title="拍照"
            :disabled="!recognitionStore.isRecording"
          >
            <i class="bi bi-camera"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- 识别结果面板 -->
    <div 
      class="fixed bottom-0 left-0 right-0 bg-white rounded-t-3xl shadow-2xl z-20 transition-transform duration-300"
      :class="showResultPanel ? 'translate-y-0' : 'translate-y-[calc(100%-120px)]'"
    >
      <!-- 拖拽指示器 -->
      <div 
        class="flex justify-center py-3 cursor-pointer"
        @click="toggleResultPanel"
      >
        <div class="w-16 h-1.5 bg-gray-300 rounded-full"></div>
      </div>

      <div class="max-w-2xl mx-auto px-4 pb-8">
        <div v-if="recognitionStore.hasResult" class="animate-fade-in">
          <!-- 头部信息 -->
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center gap-3">
              <span class="text-sm text-gray-500">{{ currentTime }}</span>
              <span 
                class="badge rounded-full px-3 py-1"
                :class="confidenceBadgeClass"
              >
                {{ Math.round(recognitionStore.confidence) }}% 置信度
              </span>
            </div>
            <span class="badge badge-success rounded-full">已识别</span>
          </div>

          <!-- 识别结果 -->
          <div class="mb-6">
            <h3 class="text-3xl font-bold text-primary-900 mb-2">
              {{ recognitionStore.latestResult?.text }}
            </h3>
            <p class="text-lg text-gray-500">{{ recognitionStore.latestResult?.pinyin }}</p>
            <p class="text-gray-600 mt-2">{{ recognitionStore.latestResult?.meaning }}</p>
          </div>

          <!-- 操作按钮 -->
          <div class="grid grid-cols-4 gap-3">
            <button
              class="flex flex-col items-center gap-1 p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
              @click="playVoice"
            >
              <i class="bi bi-volume-up text-xl text-blue-500"></i>
              <span class="text-xs text-gray-600">播放</span>
            </button>
            <button
              class="flex flex-col items-center gap-1 p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
              @click="copyResult"
            >
              <i class="bi bi-clipboard text-xl text-green-500"></i>
              <span class="text-xs text-gray-600">复制</span>
            </button>
            <button
              class="flex flex-col items-center gap-1 p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
              @click="saveToCollection"
            >
              <i class="bi bi-bookmark text-xl text-purple-500"></i>
              <span class="text-xs text-gray-600">收藏</span>
            </button>
            <button
              class="flex flex-col items-center gap-1 p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
              @click="shareResult"
            >
              <i class="bi bi-share text-xl text-orange-500"></i>
              <span class="text-xs text-gray-600">分享</span>
            </button>
          </div>
        </div>

        <div v-else class="text-center py-8">
          <i class="bi bi-hand-index text-5xl text-gray-200 mb-3"></i>
          <p class="text-gray-500">
            {{ recognitionStore.isRecording ? '正在识别手语...' : '开始识别后显示结果' }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useSpeech } from '~/composables/useSpeech'
import type { ApiResponse, UploadRecognitionResponse } from '~/types'

// SEO 元信息
useSeoMeta({
  title: '实时翻译 - 译手 HandTalk AI',
  description: '摄像头实时捕获手语动作，即时转换为文字',
})

// 组件引用
const videoRef = ref<HTMLVideoElement | null>(null)
const skeletonCanvasRef = ref<HTMLCanvasElement | null>(null)

// Composables
const config = useRuntimeConfig()

const camera = useCamera()
const recognitionStore = useRecognitionStore()
const toast = useToast()
const speech = useSpeech()

// 状态
const loading = ref(true)
const showResultPanel = ref(true)
const recordingDuration = ref('00:00')
const isRequesting = ref(false)
let lastCaptureTime = 0
let animationFrameId: number | null = null
let durationInterval: ReturnType<typeof setInterval> | null = null

// 计算属性
const confidenceClass = computed(() => {
  const confidence = recognitionStore.confidence
  if (confidence >= 80) return 'bg-gradient-to-r from-green-500 to-emerald-500'
  if (confidence >= 60) return 'bg-gradient-to-r from-yellow-500 to-orange-500'
  return 'bg-gradient-to-r from-red-500 to-pink-500'
})

const confidenceBadgeClass = computed(() => {
  const confidence = recognitionStore.confidence
  if (confidence >= 80) return 'bg-green-100 text-green-700'
  if (confidence >= 60) return 'bg-yellow-100 text-yellow-700'
  return 'bg-red-100 text-red-700'
})

// 当前时间
const currentTime = ref(new Date().toLocaleTimeString())
setInterval(() => {
  currentTime.value = new Date().toLocaleTimeString()
}, 1000)

// 格式化时长
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// 切换结果面板
function toggleResultPanel() {
  showResultPanel.value = !showResultPanel.value
}

// 切换全屏
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen()
  } else {
    document.exitFullscreen()
  }
}

// 初始化摄像头
async function initCamera() {
  loading.value = true
  try {
    await camera.startCamera(videoRef.value || undefined)
  } catch (error) {
    console.error('Camera initialization failed:', error)
  }
  loading.value = false
}

// 切换录制状态
async function toggleRecording() {
  if (recognitionStore.isRecording) {
    recognitionStore.setRecording(false)
    camera.stopCamera()

    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId)
    }
    if (durationInterval) {
      clearInterval(durationInterval)
    }
    recordingDuration.value = '00:00'
  } else {
    await initCamera()
    recognitionStore.setRecording(true)
    startCaptureLoop()
    
    // 开始计时
    let seconds = 0
    durationInterval = setInterval(() => {
      seconds++
      recordingDuration.value = formatDuration(seconds)
    }, 1000)
  }
}

// 开始捕获循环
function startCaptureLoop() {
  function capture() {
    if (recognitionStore.isRecording && videoRef.value) {
      const now = Date.now()
      const intervalMs = 1000

      if (now - lastCaptureTime >= intervalMs && !isRequesting.value) {
      const frameData = camera.captureFrame()
      if (frameData) {
          lastCaptureTime = now
          recognizeFrame(frameData)
        }
      }

      animationFrameId = requestAnimationFrame(capture)
    }
  }

  capture()
}

function dataUrlToFile(dataUrl: string, filename: string): File {
  const arr = dataUrl.split(',')

  // 基本校验，避免 undefined 传入 match / atob
  const header = arr[0]
  const base64 = arr[1]
  if (!header || !base64) {
    throw new Error('Invalid data URL')
  }

  const mimeMatch = header.match(/:(.*?);/)
  const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg'
  const bstr = atob(base64)
  let n = bstr.length
  const u8arr = new Uint8Array(n)
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n)
  }
  return new File([u8arr], filename, { type: mime })
}

async function recognizeFrame(frameData: string) {
  if (isRequesting.value) return

  isRequesting.value = true
  try {
    console.log('[recognize.vue] 即将发送实时识别请求', {
      apiBase: config.public.apiBase,
      url: `${config.public.apiBase}/recognize/upload`,
      frameLength: frameData.length,
      isRecording: recognitionStore.isRecording,
    })

    const file = dataUrlToFile(frameData, `frame_${Date.now()}.jpg`)
    const formData = new FormData()
    formData.append('file', file)

    const response = await $fetch<ApiResponse<UploadRecognitionResponse>>(
      `${config.public.apiBase}/recognize/upload`,
      {
        method: 'POST',
        body: formData,
      },
    )

    console.log('[recognize.vue] 收到后端响应', response)

    const data = response.data
    const top = data.results[0]
    if (!top) return

      recognitionStore.setResult({
      text: top.text,
      pinyin: top.pinyin,
      meaning: top.meaning,
      confidence: top.confidence,
        timestamp: new Date().toISOString(),
      })
    recognitionStore.setConfidence(top.confidence)
      showResultPanel.value = true
  } catch (error) {
    console.error('实时识别失败:', error)
  } finally {
    isRequesting.value = false
  }
}

// 播放语音
function playVoice() {
  const text = recognitionStore.latestResult?.text
  if (text) {
    speech.speak(text)
  }
}

// 复制结果
function copyResult() {
  const text = recognitionStore.latestResult?.text
  if (text) {
    navigator.clipboard.writeText(text)
    toast.success('已复制到剪贴板')
  }
}

// 分享结果
function shareResult() {
  const text = recognitionStore.latestResult?.text
  if (text) {
    const shareData = {
      title: '译手 HandTalk AI - 实时翻译结果',
      text: `手语识别结果：${text}`,
    }

    if (navigator.share) {
      navigator.share(shareData)
    } else {
      navigator.clipboard.writeText(shareData.text)
      toast.success('分享内容已复制')
    }
  }
}

// 保存到收藏
function saveToCollection() {
  toast.success('已添加到收藏')
}

// 拍照保存
function captureAndSave() {
  if (!recognitionStore.isRecording) {
    toast.warning('请先开始识别')
    return
  }
  
  const frameData = camera.captureFrame()
  if (frameData) {
    recognitionStore.addToHistory({
      id: Date.now().toString(),
      type: 'realtime',
      thumbnail: frameData,
      result: recognitionStore.latestResult?.text || '',
      confidence: recognitionStore.latestResult?.confidence || 0,
      createdAt: new Date().toISOString(),
    })

    toast.success('已保存到历史记录')
  }
}

// 页面卸载时清理
onUnmounted(() => {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
  }
  if (durationInterval) {
    clearInterval(durationInterval)
  }
  camera.stopCamera()
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
</style>
