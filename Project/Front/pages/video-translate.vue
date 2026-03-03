<template>
  <div class="relative pt-20 min-h-screen overflow-hidden bg-stone-50">

    <!-- 顶部返回栏 -->
    <div class="sticky top-0 z-30 border-b border-white/60 bg-white/60 backdrop-blur-xl">
      <div class="max-w-6xl mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center gap-3">
              <NuxtLink
              to="/"
              class="flex items-center gap-2 text-slate-700 hover:text-emerald-700 transition-colors"
            >
              <i class="bi bi-arrow-left text-lg"></i>
              <span class="hidden sm:inline">返回</span>
            </NuxtLink>
            <div class="h-6 w-px bg-slate-200/60"></div>
            <span class="text-sm font-medium tracking-wide text-slate-500">
              视频手语翻译
            </span>
          </div>
        </div>
      </div>
    </div>

    <div class="relative max-w-6xl mx-auto px-4 py-10">
      <!-- 顶部标题：极简大字号 + 信息层级 + 不对称布局 -->
      <div class="mb-12 animate-fade-up md:flex md:items-end md:justify-between">
        <div class="max-w-xl">
          <p class="mb-3 text-[11px] font-semibold tracking-[0.22em] text-emerald-600 uppercase">
            VIDEO · SIGN LANGUAGE · CTC
          </p>
          <h1
            class="font-semibold tracking-tight text-slate-900 text-3xl sm:text-4xl md:text-5xl leading-tight"
          >
            上传手语视频，一次性完成整段翻译
          </h1>
          <p class="mt-4 text-base md:text-lg text-slate-600">
            上传一段手语视频，系统将利用 CTC 模型对整段视频进行识别，并生成流畅自然的中文翻译。
          </p>
        </div>
        <div class="mt-6 md:mt-0 md:ml-10">
          <div
            class="inline-flex items-center gap-3 rounded-full border border-stone-200 bg-white/80 px-5 py-2 text-[11px] font-medium text-slate-500 shadow-sm rotate-[-2deg]"
          >
            <span
              class="inline-flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500 text-[10px] font-semibold text-white"
            >
              CTC
            </span>
            <span class="tracking-[0.16em] uppercase">Batch video translate</span>
          </div>
        </div>
      </div>
      <div class="grid gap-8 md:grid-cols-12 md:items-start">
        <!-- 左：上传与预览（更大面积） -->
        <div
          class="relative rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] md:col-span-7 animate-fade-up animation-delay-100"
        >
          <div
            class="relative border border-dashed rounded-2xl p-8 text-center transition-all duration-300 bg-stone-50"
            :class="[
              isDragging
                ? 'border-emerald-500 bg-emerald-50'
                : 'border-stone-300 hover:bg-white hover:border-emerald-500'
            ]"
            @dragenter.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @dragover.prevent
            @drop.prevent="handleDrop"
          >
            <div v-if="!videoFile" class="space-y-4">
              <div
                class="w-20 h-20 mx-auto mb-3 rounded-2xl bg-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-300/40"
              >
                <i class="bi bi-camera-video text-3xl text-white"></i>
              </div>
              <h3 class="text-xl font-semibold text-slate-900">
                拖拽或点击上传手语视频
              </h3>
              <p class="text-slate-500 text-sm">
                支持 MP4、MOV 等常见视频格式，建议时长 3-15 秒
              </p>
              <!-- 选择文件按钮 -->
              <label
                class="group relative inline-flex items-center justify-center gap-2 rounded-xl border border-dashed border-stone-300 bg-white px-5 py-3 text-sm font-medium text-slate-800 shadow-sm cursor-pointer transition-colors hover:bg-stone-50"
              >
                <i class="bi bi-cloud-upload text-emerald-500 group-hover:text-emerald-600"></i>
                选择视频文件
                <input
                  type="file"
                  class="d-none"
                  accept="video/*"
                  @change="handleFileSelect"
                />
              </label>
              <p class="text-slate-400 text-xs">
                视频将在本地与服务器安全处理，仅用于本次翻译任务。
              </p>
            </div>

            <div v-else class="space-y-4">
              <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                  <div class="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center">
                    <i class="bi bi-film text-emerald-600"></i>
                  </div>
                  <div class="text-left">
                    <p class="font-medium text-slate-800 truncate max-w-[180px]">
                      {{ videoFile.name }}
                    </p>
                    <p class="text-xs text-slate-500">
                      {{ formatSize(videoFile.size) }}
                    </p>
                  </div>
                </div>
                <button
                  class="inline-flex items-center rounded-full px-3 py-1 text-xs text-slate-500 hover:bg-slate-100"
                  @click="clearVideo"
                >
                  <i class="bi bi-x mr-1"></i>移除
                </button>
              </div>

              <div class="rounded-xl overflow-hidden bg-slate-900/5 min-h-[200px] flex items-center justify-center">
                <video
                  v-if="videoPreviewUrl"
                  :src="videoPreviewUrl"
                  controls
                  class="w-full rounded-xl"
                ></video>
                <div
                  v-else-if="result"
                  class="flex flex-col items-center justify-center gap-2 text-xs text-slate-400 px-6 py-10 text-center"
                >
                  <i class="bi bi-film text-2xl text-slate-300"></i>
                  <span>历史记录仅保存翻译结果，原始视频未保存，因此无法在此回放。</span>
                </div>
                <div
                  v-else
                  class="text-xs text-slate-400 px-6 py-10 text-center"
                >
                  上传视频后，这里将展示预览并支持播放。
                </div>
              </div>
            </div>
          </div>

          <!-- 上传进度 -->
          <div v-if="uploading" class="mt-6 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-600">{{ uploadStatus }}</span>
              <span class="text-xs text-slate-500">{{ uploadProgress }}%</span>
            </div>
            <div class="h-2 bg-slate-100/80 rounded-full overflow-hidden">
              <div
                class="h-full bg-emerald-500 transition-all duration-300"
                :style="{ width: `${uploadProgress}%` }"
              ></div>
            </div>
          </div>

          <!-- 开始翻译按钮 -->
          <button
            class="mt-6 inline-flex w-full items-center justify-center rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-white shadow-[0_16px_32px_rgba(16,185,129,0.5)] transition-transform transition-shadow duration-200 hover:shadow-[0_20px_40px_rgba(16,185,129,0.6)] hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
            :disabled="!videoFile || uploading"
            @click="startVideoTranslation"
          >
            <i class="bi bi-lightning-charge mr-2"></i>
            开始视频翻译
          </button>
        </div>

        <!-- 右：结果展示（相对更窄） -->
        <div class="space-y-6 md:col-span-5">
          <!-- 翻译结果 -->
          <div
            class="min-h-[220px] rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] animate-fade-up animation-delay-200"
          >
            <div class="mb-4 flex items-center justify-between gap-2">
              <h3 class="flex items-center gap-2 font-semibold text-slate-900">
                <i class="bi bi-translate text-emerald-500"></i>
                翻译结果
              </h3>
              <button
                v-if="result"
                class="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium hover:bg-emerald-100 transition-colors"
                :class="isCurrentFavorite ? 'text-pink-500 border-pink-200 bg-pink-50' : 'text-emerald-700'"
                @click.stop="toggleCurrentFavorite"
              >
                <span>{{ isCurrentFavorite ? '已收藏' : '收藏' }}</span>
              </button>
            </div>

            <div v-if="result" class="space-y-4">
              <div
                class="result-fade-in rounded-2xl border border-emerald-100 bg-emerald-50/70 p-4"
              >
                <p class="mb-1 text-xs text-slate-500">识别文本</p>
                <p class="break-words text-2xl font-semibold leading-snug text-slate-900">
                  {{ result.text }}
                </p>
                <p class="mt-2 text-xs text-slate-500">
                  置信度：<span class="font-semibold text-emerald-600">{{ Math.round(result.confidence) }}%</span>
                  <span v-if="result.videoDuration"> ｜ 视频时长约 {{ result.videoDuration.toFixed(1) }} 秒</span>
                </p>
              </div>

              <div class="flex flex-wrap gap-3">
                <button
                  class="inline-flex items-center rounded-full bg-emerald-50 px-4 py-1.5 text-xs font-medium text-emerald-700 shadow-sm hover:bg-emerald-100"
                  @click="playVoice"
                  :disabled="!result"
                >
                  <i class="bi bi-volume-up mr-1"></i>播放语音
                </button>
                <button
                  class="inline-flex items-center rounded-full bg-stone-100 px-4 py-1.5 text-xs font-medium text-slate-800 shadow-sm hover:bg-stone-200"
                  @click="copyResult"
                  :disabled="!result"
                >
                  <i class="bi bi-clipboard mr-1"></i>复制结果
                </button>
              </div>
            </div>

            <div v-else class="flex h-full items-center justify-center text-sm text-slate-400">
              上传视频并点击“开始视频翻译”后，这里会呈现识别结果。
            </div>
          </div>

          <!-- 使用建议 -->
          <div
            class="rounded-3xl border border-stone-200 bg-white p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] animate-fade-up animation-delay-300"
          >
            <h3 class="mb-3 flex items-center gap-2 font-semibold text-slate-900">
              <i class="bi bi-lightbulb text-amber-400"></i>
              使用建议
            </h3>
            <ul class="space-y-2 text-sm text-slate-600">
              <li class="flex items-start gap-2">
                <span class="mt-1 inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400"></span>
                <span>视频时长建议控制在 3-15 秒，动作完整但不过长</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="mt-1 inline-flex h-2.5 w-2.5 rounded-full bg-amber-400"></span>
                <span>保持镜头稳定，手势尽量在画面中心位置</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="mt-1 inline-flex h-2.5 w-2.5 rounded-full bg-emerald-300"></span>
                <span>光线充足、背景简洁有助于提升识别准确率</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({
  title: '视频翻译 - 译手 HandTalk AI',
  description: '上传手语视频，通过 CTC 模型对整段视频进行识别与翻译。',
})

import type { ApiResponse, UploadRecognitionResponse, HistoryRecord } from '~/types'

const config = useRuntimeConfig()
const toast = useToast()
const speech = useSpeech()
const recognitionStore = useRecognitionStore()
const route = useRoute()

interface VideoResult {
  text: string
  confidence: number
  videoDuration: number
}

const isDragging = ref(false)
const videoFile = ref<File | null>(null)
const videoPreviewUrl = ref<string | null>(null)
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadStatus = ref('')
const result = ref<VideoResult | null>(null)
const currentHistoryId = ref<string | null>(null)

const isCurrentFavorite = computed(() => {
  if (!currentHistoryId.value) return false
  return recognitionStore.history.find(h => h.id === currentHistoryId.value)?.favorite === true
})

function toggleCurrentFavorite() {
  if (!currentHistoryId.value) return
  recognitionStore.toggleFavorite(currentHistoryId.value)
}

function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) {
    setVideoFile(file)
  }
  input.value = ''
}

function handleDrop(event: DragEvent) {
  isDragging.value = false
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  if (!file.type.startsWith('video/')) {
    toast.warning('请上传视频文件')
    return
  }
  setVideoFile(file)
}

function setVideoFile(file: File) {
  if (!file.type.startsWith('video/')) {
    toast.warning('请选择视频文件')
    return
  }
  if (file.size > 200 * 1024 * 1024) {
    toast.warning('视频过大，请控制在 200MB 以内')
    return
  }
  if (videoPreviewUrl.value) {
    URL.revokeObjectURL(videoPreviewUrl.value)
  }
  videoFile.value = file
  videoPreviewUrl.value = URL.createObjectURL(file)
  result.value = null
  currentHistoryId.value = null
}

function clearVideo() {
  if (videoPreviewUrl.value) {
    URL.revokeObjectURL(videoPreviewUrl.value)
  }
  videoFile.value = null
  videoPreviewUrl.value = null
  result.value = null
}

function formatSize(size: number): string {
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

async function startVideoTranslation() {
  if (!videoFile.value) return

  uploading.value = true
  uploadProgress.value = 0
  uploadStatus.value = '正在上传视频...'

  try {
    const formData = new FormData()
    formData.append('file', videoFile.value)

    console.log('[video-translate.vue] 即将发送视频翻译请求', {
      apiBase: config.public.apiBase,
      url: `${config.public.apiBase}/recognize/upload`,
      fileName: videoFile.value.name,
      fileSize: videoFile.value.size,
    })

    uploadProgress.value = 30
    uploadStatus.value = '正在调用识别服务...'

    const response = await $fetch<ApiResponse<UploadRecognitionResponse>>(
      `${config.public.uploadBase}/recognize/upload`,
      {
        method: 'POST',
        body: formData,
      },
    )

    console.log('[video-translate.vue] 收到后端响应', response)

    uploadProgress.value = 80

    const data = response.data
    const top = data.results[0]
    if (!top) {
      throw new Error('识别结果为空')
    }

    result.value = {
      text: top.text,
      confidence: top.confidence,
      videoDuration: data.videoDuration || 0,
    }

    const historyId = data.id || Date.now().toString()

    // 为历史记录生成一张视频缩略图（首帧），用于“翻译历史 / 我的收藏”展示
    let thumbnail: string | undefined
    try {
      thumbnail = await generateVideoThumbnail(videoFile.value)
    } catch (e) {
      console.warn('生成视频缩略图失败，将使用占位图:', e)
    }

    recognitionStore.addToHistory({
      id: historyId,
      type: 'upload_video',
      result: top.text,
      confidence: top.confidence,
      duration: data.videoDuration || 0,
      thumbnail,
      // 将后端返回的相对 videoUrl 拼接为完整可访问地址，便于历史记录中直接回放
      videoUrl: data.videoUrl ? `${config.public.uploadBase}${data.videoUrl}` : undefined,
      createdAt: data.createdAt || new Date().toISOString(),
      favorite: false,
    })

    currentHistoryId.value = historyId

    uploadStatus.value = '翻译完成'
    uploadProgress.value = 100
    toast.success('视频翻译完成')
  } catch (error: any) {
    console.error('视频翻译失败:', error)
    uploadStatus.value = '翻译失败'
    toast.error(error?.message || '视频翻译失败，请稍后重试')
  } finally {
    await new Promise(resolve => setTimeout(resolve, 500))
    uploading.value = false
  }
}

// 生成视频缩略图（首帧），以 base64 data URL 形式返回
async function generateVideoThumbnail(file: File): Promise<string> {
  if (typeof document === 'undefined') {
    throw new Error('当前环境不支持生成视频缩略图')
  }

  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    const canvas = document.createElement('canvas')
    const url = URL.createObjectURL(file)

    let handled = false

    const cleanup = () => {
      if (handled) return
      handled = true
      URL.revokeObjectURL(url)
    }

    video.preload = 'metadata'
    video.src = url
    video.muted = true
    video.playsInline = true

    video.onloadeddata = () => {
      // 取 0.1 秒或中间位置，避免黑屏
      const targetTime =
        isFinite(video.duration) && video.duration > 0
          ? Math.min(video.duration / 2, 1)
          : 0.1
      video.currentTime = targetTime
    }

    video.onseeked = () => {
      try {
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 360
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          throw new Error('无法获取 Canvas 上下文')
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
        cleanup()
        resolve(dataUrl)
      } catch (error) {
        cleanup()
        reject(error)
      }
    }

    video.onerror = () => {
      cleanup()
      reject(new Error('视频加载失败，无法生成缩略图'))
    }
  })
}

function playVoice() {
  if (result.value) {
    speech.speak(result.value.text)
  }
}

function copyResult() {
  if (result.value) {
    navigator.clipboard.writeText(result.value.text)
    toast.success('已复制到剪贴板')
  }
}

onUnmounted(() => {
  if (videoPreviewUrl.value) {
    URL.revokeObjectURL(videoPreviewUrl.value)
  }
})

onMounted(() => {
  const historyId = route.query.historyId as string | undefined
  if (!historyId) return

  if (process.client) {
    recognitionStore.loadHistory()
  }

  const record = (recognitionStore.history as HistoryRecord[]).find(
    (h) => h.id === historyId && h.type === 'upload_video',
  )

  if (!record) return

  result.value = {
    text: record.result,
    confidence: record.confidence,
    videoDuration: record.duration || 0,
  }
  currentHistoryId.value = record.id

  // 如果历史记录中保存了视频地址，则直接用于预览与回放
  if (record.videoUrl) {
    videoPreviewUrl.value = record.videoUrl
  }
})
</script>

<style scoped>
@keyframes fade-up-soft {
  0% {
    opacity: 0;
    transform: translateY(16px) scale(0.98);
  }
  100% {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.animate-fade-up {
  animation: fade-up-soft 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.animation-delay-100 {
  animation-delay: 0.1s;
}

.animation-delay-200 {
  animation-delay: 0.2s;
}

.animation-delay-300 {
  animation-delay: 0.3s;
}

.result-fade-in {
  animation: fade-up-soft 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
</style>

