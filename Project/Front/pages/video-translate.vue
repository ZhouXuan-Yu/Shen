<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <div class="sticky top-16 z-30 bg-white/80 backdrop-blur-lg border-b border-gray-100">
      <div class="max-w-7xl mx-auto px-4">
        <div class="flex items-center justify-between py-3">
          <div class="flex items-center gap-3">
            <NuxtLink to="/" class="flex items-center gap-2 text-gray-600 hover:text-primary-900 transition-colors">
              <i class="bi bi-arrow-left text-lg"></i>
              <span class="hidden sm:inline">返回</span>
            </NuxtLink>
            <div class="h-6 w-px bg-gray-200"></div>
            <h1 class="text-lg font-semibold text-primary-900">视频翻译</h1>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-5xl mx-auto px-4 py-10">
      <div class="text-center mb-10">
        <h1 class="text-4xl font-bold text-primary-900 mb-4">
          <span class="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            上传视频手语翻译
          </span>
        </h1>
        <p class="text-gray-600 text-lg max-w-2xl mx-auto">
          上传一段手语视频，系统将利用 CTC 模型对整段视频进行识别并生成中文翻译。
        </p>
      </div>

      <div class="grid md:grid-cols-2 gap-8">
        <!-- 左：上传与预览 -->
        <div class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 p-6">
          <div
            class="relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300"
            :class="[
              isDragging
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            ]"
            @dragenter.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @dragover.prevent
            @drop.prevent="handleDrop"
          >
            <div v-if="!videoFile" class="space-y-4">
              <div class="w-20 h-20 mx-auto mb-2 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                <i class="bi bi-camera-video text-3xl text-white"></i>
              </div>
              <h3 class="text-xl font-bold text-primary-900">
                拖拽或点击上传视频
              </h3>
              <p class="text-gray-500 text-sm">
                支持 MP4、MOV 等常见视频格式，建议时长 3-15 秒
              </p>
              <label class="btn btn-primary rounded-xl cursor-pointer inline-flex items-center gap-2">
                <i class="bi bi-cloud-upload"></i>
                选择视频文件
                <input
                  type="file"
                  class="d-none"
                  accept="video/*"
                  @change="handleFileSelect"
                />
              </label>
              <p class="text-gray-400 text-xs">
                视频将在本地与服务器安全处理，不会用于其他用途
              </p>
            </div>

            <div v-else class="space-y-4">
              <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                  <div class="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center">
                    <i class="bi bi-film text-blue-600"></i>
                  </div>
                  <div class="text-left">
                    <p class="font-medium text-primary-900 truncate max-w-[180px]">
                      {{ videoFile.name }}
                    </p>
                    <p class="text-xs text-gray-500">
                      {{ formatSize(videoFile.size) }}
                    </p>
                  </div>
                </div>
                <button class="btn btn-ghost btn-xs" @click="clearVideo">
                  <i class="bi bi-x mr-1"></i>移除
                </button>
              </div>

              <div class="rounded-xl overflow-hidden bg-black/5">
                <video
                  v-if="videoPreviewUrl"
                  :src="videoPreviewUrl"
                  controls
                  class="w-full rounded-xl"
                ></video>
              </div>
            </div>
          </div>

          <!-- 上传进度 -->
          <div v-if="uploading" class="mt-6 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600">{{ uploadStatus }}</span>
              <span class="text-xs text-gray-500">{{ uploadProgress }}%</span>
            </div>
            <div class="h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                class="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                :style="{ width: `${uploadProgress}%` }"
              ></div>
            </div>
          </div>

          <!-- 开始翻译按钮 -->
          <button
            class="btn btn-primary w-full mt-6 rounded-xl py-3"
            :disabled="!videoFile || uploading"
            @click="startVideoTranslation"
          >
            <i class="bi bi-lightning-charge mr-2"></i>
            开始视频翻译
          </button>
        </div>

        <!-- 右：结果展示 -->
        <div class="space-y-6">
          <div class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 p-6 min-h-[220px]">
            <h3 class="font-semibold text-primary-900 mb-4 flex items-center gap-2">
              <i class="bi bi-translate text-blue-500"></i>
              翻译结果
            </h3>

            <div v-if="result" class="space-y-4">
              <div class="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-4 border border-blue-100">
                <p class="text-xs text-gray-500 mb-1">识别文本</p>
                <p class="text-2xl font-bold text-primary-900 break-words">
                  {{ result.text }}
                </p>
                <p class="text-xs text-gray-500 mt-2">
                  置信度：<span class="font-semibold text-blue-600">{{ Math.round(result.confidence) }}%</span>
                  <span v-if="result.videoDuration"> ｜ 视频时长约 {{ result.videoDuration.toFixed(1) }} 秒</span>
                </p>
              </div>

              <div class="flex flex-wrap gap-3">
                <button class="btn btn-secondary btn-sm rounded-full" @click="playVoice" :disabled="!result">
                  <i class="bi bi-volume-up mr-1"></i>播放语音
                </button>
                <button class="btn btn-secondary btn-sm rounded-full" @click="copyResult" :disabled="!result">
                  <i class="bi bi-clipboard mr-1"></i>复制结果
                </button>
              </div>
            </div>

            <div v-else class="flex items-center justify-center h-full text-gray-400 text-sm">
              上传视频并点击“开始视频翻译”后，这里会显示识别结果
            </div>
          </div>

          <div class="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl p-6">
            <h3 class="font-semibold text-primary-900 mb-3 flex items-center gap-2">
              <i class="bi bi-lightbulb text-yellow-500"></i>
              使用建议
            </h3>
            <ul class="space-y-2 text-sm text-gray-600">
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
                <span>视频时长建议控制在 3-15 秒，动作完整但不过长</span>
              </li>
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
                <span>保持镜头稳定，手势尽量在画面中心位置</span>
              </li>
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
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

import type { ApiResponse, UploadRecognitionResponse } from '~/types'

const config = useRuntimeConfig()
const toast = useToast()
const speech = useSpeech()

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
</script>

<style scoped>
</style>

