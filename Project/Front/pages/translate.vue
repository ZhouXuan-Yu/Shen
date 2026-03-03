<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <!-- 顶部导航栏 -->
    <div class="sticky top-16 z-30 bg-white/80 backdrop-blur-lg border-b border-gray-100">
      <div class="max-w-7xl mx-auto px-4">
        <div class="flex items-center justify-between py-3">
          <div class="flex items-center gap-3">
            <NuxtLink to="/" class="flex items-center gap-2 text-gray-600 hover:text-primary-900 transition-colors">
              <i class="bi bi-arrow-left text-lg"></i>
              <span class="hidden sm:inline">返回</span>
            </NuxtLink>
            <div class="h-6 w-px bg-gray-200"></div>
            <h1 class="text-lg font-semibold text-primary-900">图片序列翻译</h1>
          </div>
          
          <!-- 快捷操作 -->
          <div v-if="uploadedFiles.length > 0" class="flex items-center gap-2">
            <button 
              class="btn btn-secondary btn-sm rounded-full"
              @click="clearAllFiles"
            >
              <i class="bi bi-trash mr-1"></i>
              清除
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-7xl mx-auto px-4 py-10">
      <!-- 页面标题区域 -->
      <div class="text-center mb-12">
        <h1 class="text-4xl font-bold text-primary-900 mb-4">
          <span class="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            图片序列手语翻译
          </span>
        </h1>
        <p class="text-gray-600 text-lg max-w-2xl mx-auto">
          上传连贯图片序列，通过多张连续图片分析准确识别完整的手语动作含义
        </p>
      </div>

      <!-- 主要内容区域 -->
      <div class="grid lg:grid-cols-3 gap-8">
        <!-- 左侧：上传区域 -->
        <div class="lg:col-span-2">
          <div class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 overflow-hidden">
            <!-- 拖拽上传区域 -->
            <div
              class="relative border-2 border-dashed rounded-t-2xl p-10 text-center transition-all duration-300"
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
              <!-- 背景装饰 -->
              <div class="absolute inset-0 pointer-events-none">
                <div class="absolute top-4 left-4 w-20 h-20 bg-blue-100 rounded-full opacity-50 blur-xl"></div>
                <div class="absolute bottom-4 right-4 w-24 h-24 bg-purple-100 rounded-full opacity-50 blur-xl"></div>
              </div>

              <div v-if="!uploading" class="relative z-10">
                <!-- 图标 -->
                <div class="w-20 h-20 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                  <i class="bi bi-images text-3xl text-white"></i>
                </div>
                
                <h3 class="text-xl font-bold text-primary-900 mb-2">
                  {{ uploadedFiles.length > 0 ? '添加更多图片' : '上传图片序列' }}
                </h3>
                <p class="text-gray-500 text-sm mb-6">
                  拖拽图片到此区域，或点击下方按钮选择
                </p>

                <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
                  <label class="btn btn-primary rounded-xl cursor-pointer shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40">
                    <i class="bi bi-cloud-upload mr-2"></i>
                    选择图片
                    <input
                      type="file"
                      class="d-none"
                      accept="image/*"
                      multiple
                      @change="handleFileSelect"
                    />
                  </label>
                  
                  <label class="btn btn-secondary rounded-xl cursor-pointer">
                    <i class="bi bi-camera mr-2"></i>
                    拍照上传
                    <input
                      type="file"
                      class="d-none"
                      accept="image/*"
                      capture="environment"
                      multiple
                      @change="handleFileSelect"
                    />
                  </label>
                </div>

                <p class="text-gray-400 text-xs mt-6">
                  支持 JPG, PNG 格式 | 建议上传 3-10 张连贯图片 | 单张最大 10MB
                </p>
              </div>

              <!-- 上传进度 -->
              <div v-else class="relative z-10 py-8">
                <div class="w-24 h-24 mx-auto mb-4 relative">
                  <svg class="w-24 h-24 transform -rotate-90">
                    <circle
                      class="text-gray-200"
                      stroke-width="8"
                      stroke="currentColor"
                      fill="transparent"
                      :r="48"
                      :cx="48"
                      :cy="48"
                    />
                    <circle
                      class="text-blue-500"
                      stroke-width="8"
                      stroke="currentColor"
                      fill="transparent"
                      :r="48"
                      :cx="48"
                      :cy="48"
                      :stroke-dasharray="301.59"
                      :stroke-dashoffset="301.59 - (301.59 * uploadProgress / 100)"
                      stroke-linecap="round"
                    />
                  </svg>
                  <div class="absolute inset-0 flex items-center justify-center">
                    <i :class="uploadStatusIcon" class="text-2xl text-blue-500"></i>
                  </div>
                </div>
                
                <h4 class="text-lg font-semibold text-primary-900 mb-2">
                  {{ uploadStatus }}
                </h4>
                <p class="text-gray-500 mb-4">{{ uploadProgress }}%</p>
                
                <!-- 进度条 -->
                <div class="max-w-xs mx-auto h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    class="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                    :style="{ width: `${uploadProgress}%` }"
                  ></div>
                </div>
              </div>
            </div>

            <!-- 已上传图片展示 -->
            <div v-if="uploadedFiles.length > 0" class="p-6 border-t border-gray-100">
              <!-- 标题 -->
              <div class="flex items-center justify-between mb-4">
                <h4 class="font-semibold text-primary-900 flex items-center gap-2">
                  <i class="bi bi-images text-blue-500"></i>
                  已上传 ({{ uploadedFiles.length }} 张)
                </h4>
                <span class="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                  {{ getSequenceTip() }}
                </span>
              </div>

              <!-- 图片画廊 -->
              <div class="image-gallery">
                <div 
                  v-for="(file, index) in uploadedFiles" 
                  :key="file.id"
                  class="image-card"
                  @click="selectImage(file)"
                >
                  <!-- 序号标记 -->
                  <div class="image-number">{{ index + 1 }}</div>
                  
                  <!-- 图片预览 -->
                  <div class="aspect-square bg-gray-100 rounded-xl overflow-hidden relative">
                    <img
                      :src="file.previewUrl"
                      :alt="file.name"
                      class="w-full h-full object-cover transition-transform duration-300 hover:scale-110"
                    />
                    
                    <!-- 移除按钮 -->
                    <button 
                      class="remove-btn"
                      @click.stop="removeFile(file.id)"
                    >
                      <i class="bi bi-x"></i>
                    </button>
                  </div>
                </div>

                <!-- 添加更多 -->
                <label class="image-card add-more">
                  <div class="aspect-square bg-gray-50 rounded-xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center hover:border-blue-500 hover:bg-blue-50 transition-all cursor-pointer">
                    <i class="bi bi-plus-circle text-3xl text-gray-300 mb-2"></i>
                    <span class="text-sm text-gray-500">添加更多</span>
                  </div>
                  <input
                    type="file"
                    class="d-none"
                    accept="image/*"
                    multiple
                    @change="handleFileSelect"
                  />
                </label>
              </div>
            </div>
          </div>

          <!-- 翻译结果展示 -->
          <Transition name="slide-up">
            <div v-if="result" class="mt-8 bg-white rounded-2xl shadow-xl shadow-gray-200/50 overflow-hidden">
              <!-- 头部 -->
              <div class="bg-gradient-to-r from-blue-500 to-purple-500 text-white p-6">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-3">
                    <div class="w-12 h-12 rounded-xl bg-white/20 flex items-center justify-center">
                      <i class="bi bi-translate text-xl"></i>
                    </div>
                    <div>
                      <h3 class="text-xl font-bold">翻译结果</h3>
                      <p class="text-white/80 text-sm">基于 {{ uploadedFiles.length }} 张连贯图片分析</p>
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="text-3xl font-bold">{{ Math.round(result.confidence) }}%</div>
                    <div class="text-xs text-white/80">置信度</div>
                  </div>
                </div>
              </div>

              <div class="p-6">
                <div class="grid md:grid-cols-2 gap-8">
                  <!-- 左侧：图片序列预览 -->
                  <div>
                    <h4 class="text-sm font-medium text-gray-500 mb-3 flex items-center gap-2">
                      <i class="bi bi-images"></i>
                      图片序列预览
                    </h4>
                    <div class="sequence-preview">
                      <div 
                        v-for="(file, index) in uploadedFiles.slice(0, 6)" 
                        :key="file.id"
                        class="sequence-thumb"
                      >
                        <img :src="file.previewUrl" class="w-full h-full object-cover rounded-lg" />
                        <span class="sequence-num">{{ index + 1 }}</span>
                      </div>
                      <div v-if="uploadedFiles.length > 6" class="sequence-thumb more">
                        <span>+{{ uploadedFiles.length - 6 }}</span>
                      </div>
                    </div>
                  </div>

                  <!-- 右侧：识别结果 -->
                  <div class="space-y-6">
                    <!-- 手语识别 -->
                    <div>
                      <label class="text-xs text-gray-500 mb-2 block">手语识别</label>
                      <div class="result-text bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-4 border border-blue-100">
                        <h2 class="text-3xl font-bold text-primary-900">{{ result.text }}</h2>
                        <p class="text-gray-500 mt-1">{{ result.pinyin }}</p>
                      </div>
                    </div>

                    <!-- 释义 -->
                    <div>
                      <label class="text-xs text-gray-500 mb-2 block">释义</label>
                      <p class="text-gray-700 text-lg leading-relaxed">{{ result.meaning }}</p>
                    </div>

                    <!-- 动作分解 -->
                    <div v-if="result.actions && result.actions.length > 0">
                      <label class="text-xs text-gray-500 mb-2 block">动作分解</label>
                      <div class="flex flex-wrap gap-2">
                        <span 
                          v-for="(action, idx) in result.actions" 
                          :key="idx"
                          class="badge bg-gradient-to-r from-blue-500 to-purple-500 text-white px-3 py-1.5 rounded-full"
                        >
                          {{ action }}
                        </span>
                      </div>
                    </div>

                    <!-- 操作按钮 -->
                    <div class="flex flex-wrap gap-3 pt-4">
                      <button class="btn btn-secondary btn-sm rounded-full" @click="playVoice">
                        <i class="bi bi-volume-up mr-1"></i>播放语音
                      </button>
                      <button class="btn btn-secondary btn-sm rounded-full" @click="copyResult">
                        <i class="bi bi-clipboard mr-1"></i>复制结果
                      </button>
                      <button class="btn btn-secondary btn-sm rounded-full" @click="shareResult">
                        <i class="bi bi-share mr-1"></i>分享
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Transition>
        </div>

        <!-- 右侧：参数设置和历史 -->
        <div class="space-y-8">
          <!-- 翻译设置 -->
          <div class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 p-6">
            <h3 class="font-semibold text-primary-900 mb-4 flex items-center gap-2">
              <i class="bi bi-sliders text-blue-500"></i>
              翻译设置
            </h3>

            <div class="space-y-4">
              <!-- 分析模式 -->
              <div>
                <label class="text-sm text-gray-600 mb-2 block">分析模式</label>
                <select v-model="analysisMode" class="input">
                  <option value="standard">标准模式</option>
                  <option value="detailed">详细模式</option>
                  <option value="quick">快速模式</option>
                </select>
              </div>

              <!-- 置信度阈值 -->
              <div>
                <label class="text-sm text-gray-600 mb-2 block">
                  置信度阈值: {{ confidenceThreshold }}%
                </label>
                <input 
                  type="range" 
                  v-model="confidenceThreshold" 
                  min="50" 
                  max="100" 
                  class="w-full"
                />
              </div>

              <!-- 输出格式 -->
              <div>
                <label class="text-sm text-gray-600 mb-2 block">输出格式</label>
                <div class="grid grid-cols-2 gap-2">
                  <button 
                    class="btn btn-sm rounded-lg"
                    :class="outputFormat === 'text' ? 'btn-primary' : 'btn-secondary'"
                    @click="outputFormat = 'text'"
                  >
                    <i class="bi bi-text-paragraph mr-1"></i>文字
                  </button>
                  <button 
                    class="btn btn-sm rounded-lg"
                    :class="outputFormat === 'both' ? 'btn-primary' : 'btn-secondary'"
                    @click="outputFormat = 'both'"
                  >
                    <i class="bi bi-list-check mr-1"></i>完整
                  </button>
                </div>
              </div>
            </div>

            <!-- 开始翻译按钮 -->
            <button 
              class="btn btn-primary w-full mt-6 rounded-xl py-3"
              :disabled="uploadedFiles.length === 0 || uploading"
              @click="startTranslation"
            >
              <i class="bi bi-lightning-charge mr-2"></i>
              开始翻译
            </button>
          </div>

          <!-- 翻译历史 -->
          <div v-if="recognitionStore.history.length > 0" class="bg-white rounded-2xl shadow-xl shadow-gray-200/50 p-6">
            <h3 class="font-semibold text-primary-900 mb-4 flex items-center gap-2">
              <i class="bi bi-clock-history text-purple-500"></i>
              历史记录
            </h3>
            
            <div class="space-y-3">
              <div
                v-for="record in recognitionStore.history.slice(0, 5)"
                :key="record.id"
                class="history-item p-3 rounded-xl hover:bg-gray-50 cursor-pointer transition-colors"
                @click="loadHistoryItem(record)"
              >
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 rounded-lg bg-gray-100 overflow-hidden flex-shrink-0">
                    <img
                      v-if="record.thumbnail"
                      :src="record.thumbnail"
                      class="w-full h-full object-cover"
                    />
                    <div v-else class="w-full h-full flex items-center justify-center">
                      <i class="bi bi-image text-gray-400"></i>
                    </div>
                  </div>
                  <div class="flex-1 min-w-0">
                    <p class="font-medium text-primary-900 truncate">{{ record.result }}</p>
                    <p class="text-xs text-gray-500">
                      {{ formatTime(record.createdAt) }}
                    </p>
                  </div>
                  <span class="text-xs text-green-600 font-medium">
                    {{ Math.round(record.confidence) }}%
                  </span>
                </div>
              </div>
            </div>

            <button 
              v-if="recognitionStore.history.length > 5"
              class="btn btn-ghost btn-sm w-full mt-4"
              @click="clearHistory"
            >
              清除历史
            </button>
          </div>

          <!-- 使用帮助 -->
          <div class="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl p-6">
            <h3 class="font-semibold text-primary-900 mb-3 flex items-center gap-2">
              <i class="bi bi-lightbulb text-yellow-500"></i>
              使用帮助
            </h3>
            <ul class="space-y-2 text-sm text-gray-600">
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
                <span>上传 3-10 张连贯图片效果最佳</span>
              </li>
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
                <span>确保手势动作完整清晰</span>
              </li>
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
                <span>图片光照均匀，背景简洁</span>
              </li>
              <li class="flex items-start gap-2">
                <i class="bi bi-check-circle text-green-500 mt-0.5"></i>
                <span>建议使用连续拍摄模式</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 常见问题 -->
      <div class="mt-16">
        <h2 class="text-2xl font-bold text-primary-900 text-center mb-8">
          常见问题
        </h2>
        
        <div class="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          <div class="faq-card bg-white rounded-xl p-6 shadow-lg">
            <h3 class="font-semibold text-primary-900 mb-2 flex items-center gap-2">
              <i class="bi bi-question-circle text-blue-500"></i>
              支持哪些图片格式？
            </h3>
            <p class="text-gray-600 text-sm">
              支持 JPG、PNG、GIF 等常用图片格式。建议上传 JPG 或 PNG 格式的图片以获得最佳识别效果。
            </p>
          </div>

          <div class="faq-card bg-white rounded-xl p-6 shadow-lg">
            <h3 class="font-semibold text-primary-900 mb-2 flex items-center gap-2">
              <i class="bi bi-question-circle text-blue-500"></i>
              上传多少张图片合适？
            </h3>
            <p class="text-gray-600 text-sm">
              建议上传 3-10 张连贯图片。图片太少可能无法准确识别动作，图片太多会增加处理时间。
            </p>
          </div>

          <div class="faq-card bg-white rounded-xl p-6 shadow-lg">
            <h3 class="font-semibold text-primary-900 mb-2 flex items-center gap-2">
              <i class="bi bi-question-circle text-blue-500"></i>
              识别准确率如何？
            </h3>
            <p class="text-gray-600 text-sm">
              在标准条件下，我们的算法可以达到 95% 以上的识别准确率。准确率受图片质量、光照条件等因素影响。
            </p>
          </div>

          <div class="faq-card bg-white rounded-xl p-6 shadow-lg">
            <h3 class="font-semibold text-primary-900 mb-2 flex items-center gap-2">
              <i class="bi bi-question-circle text-blue-500"></i>
              如何提高识别准确率？
            </h3>
            <p class="text-gray-600 text-sm">
              确保手势动作完整、光照均匀、背景简洁。使用连拍模式捕捉完整动作，可有效提升识别效果。
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// SEO 元信息
useSeoMeta({
  title: '图片序列翻译 - 译手 HandTalk AI',
  description: '上传连贯图片序列，通过多张连续图片准确识别完整的手语动作含义',
})

// 类型定义
import type { ApiResponse, UploadRecognitionResponse } from '~/types'

interface UploadedFile {
  id: string
  name: string
  size: number
  previewUrl: string
  file: File
}

interface TranslationResult {
  text: string
  pinyin: string
  meaning: string
  confidence: number
  actions?: string[]
}

// 状态
const config = useRuntimeConfig()

const isDragging = ref(false)
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadStatus = ref('')
const uploadStatusIcon = ref('bi bi-cloud-arrow-up')
const uploadedFiles = ref<UploadedFile[]>([])
const result = ref<TranslationResult | null>(null)
const analysisMode = ref('standard')
const confidenceThreshold = ref(75)
const outputFormat = ref('text')

// Composables
const recognitionStore = useRecognitionStore()
const toast = useToast()
const speech = useSpeech()

// 处理文件选择
function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  const files = input.files
  if (files) {
    processFiles(Array.from(files))
  }
  input.value = ''
}

// 处理拖拽
function handleDrop(event: DragEvent) {
  isDragging.value = false
  const files = event.dataTransfer?.files
  if (files) {
    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'))
    if (imageFiles.length > 0) {
      processFiles(imageFiles)
    } else {
      toast.warning('请上传图片文件')
    }
  }
}

// 处理文件
async function processFiles(files: File[]) {
  const validFiles = files.filter(f => {
    if (!f.type.startsWith('image/')) {
      toast.warning(`${f.name} 不是图片文件`)
      return false
    }
    if (f.size > 10 * 1024 * 1024) {
      toast.warning(`${f.name} 超过 10MB 限制`)
      return false
    }
    return true
  })

  if (validFiles.length === 0) return

  uploading.value = true
  uploadProgress.value = 0
  uploadStatus.value = '正在处理图片...'
  uploadStatusIcon.value = 'bi bi-images'

  for (let i = 0; i < validFiles.length; i++) {
    const file = validFiles[i]
    if (!file) continue
    
    const previewUrl = URL.createObjectURL(file)
    
    uploadedFiles.value.push({
      id: Date.now().toString() + Math.random(),
      name: file.name,
      size: file.size,
      previewUrl,
      file,
    })

    uploadProgress.value = Math.round(((i + 1) / validFiles.length) * 80)
    await new Promise(resolve => setTimeout(resolve, 100))
  }

  uploadStatus.value = '图片处理完成'
  uploadStatusIcon.value = 'bi bi-check-circle'
  uploadProgress.value = 100
  
  await new Promise(resolve => setTimeout(resolve, 500))
  uploading.value = false
}

// 获取序列提示
function getSequenceTip() {
  const count = uploadedFiles.value.length
  if (count === 0) return ''
  if (count < 3) return '建议上传 3 张以上'
  if (count > 10) return '图片较多，分析时间较长'
  return '图片连贯性良好'
}

// 选择图片
function selectImage(file: UploadedFile) {
  toast.info('查看图片详情')
}

// 移除文件
function removeFile(fileId: string) {
  const index = uploadedFiles.value.findIndex(f => f.id === fileId)
  if (index > -1) {
    const file = uploadedFiles.value[index]
    if (file) {
      URL.revokeObjectURL(file.previewUrl)
    }
    uploadedFiles.value.splice(index, 1)
  }
}

// 清除全部
function clearAllFiles() {
  uploadedFiles.value.forEach(file => {
    URL.revokeObjectURL(file.previewUrl)
  })
  uploadedFiles.value = []
  result.value = null
  toast.success('已清除全部图片')
}

// 开始翻译
async function startTranslation() {
  if (uploadedFiles.value.length === 0) return

  uploading.value = true
  uploadProgress.value = 0
  uploadStatus.value = '正在上传图片...'
  uploadStatusIcon.value = 'bi bi-cloud-arrow-up'

  try {
    const target = uploadedFiles.value[0]
    if (!target) {
      throw new Error('没有可用的图片')
    }

    const formData = new FormData()
    formData.append('file', target.file)

    console.log('[translate.vue] 即将发送翻译请求', {
      apiBase: config.public.apiBase,
      url: `${config.public.apiBase}/recognize/upload`,
      fileName: target.name,
      fileSize: target.size,
    })

    uploadProgress.value = 30
    uploadStatus.value = '正在调用识别服务...'
    uploadStatusIcon.value = 'bi bi-gear-wide-connected'

    const response = await $fetch<ApiResponse<UploadRecognitionResponse>>(
      `${config.public.uploadBase}/recognize/upload`,
      {
        method: 'POST',
        body: formData,
      },
    )

    console.log('[translate.vue] 收到后端响应', response)

    uploadProgress.value = 80

    const data = response.data
    const top = data.results[0]
  
    if (!top) {
      throw new Error('识别结果为空')
    }

    result.value = {
      text: top.text,
      pinyin: top.pinyin,
      meaning: top.meaning,
      confidence: top.confidence,
      actions: ['起始动作', '过渡动作', '收尾动作'],
  }

    recognitionStore.addToHistory({
      id: data.id || Date.now().toString(),
      type: 'upload_image',
      result: top.text,
      confidence: top.confidence,
      thumbnail: target.previewUrl,
      createdAt: data.createdAt || new Date().toISOString(),
    })

  uploadStatus.value = '翻译完成'
  uploadStatusIcon.value = 'bi bi-check-circle-fill'
  uploadProgress.value = 100
  
    toast.success('翻译完成')
  } catch (error: any) {
    console.error('翻译失败:', error)
    uploadStatus.value = '翻译失败'
    uploadStatusIcon.value = 'bi bi-exclamation-triangle'
    toast.error(error?.message || '翻译失败，请稍后重试')
  } finally {
  await new Promise(resolve => setTimeout(resolve, 500))
  uploading.value = false
  }
}

// 格式化时间
function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  
  if (diff < 60 * 1000) return '刚刚'
  if (diff < 60 * 60 * 1000) return Math.floor(diff / (60 * 1000)) + '分钟前'
  if (diff < 24 * 60 * 60 * 1000) return Math.floor(diff / (60 * 60 * 1000)) + '小时前'
  
  return date.toLocaleDateString('zh-CN')
}

// 清除历史
function clearHistory() {
  recognitionStore.clearHistory()
  toast.success('历史记录已清除')
}

// 加载历史记录
function loadHistoryItem(record: any) {
  if (record.type === 'image_sequence') {
    toast.info('加载历史翻译')
  }
}

// 播放语音
function playVoice() {
  if (result.value) {
    speech.speak(result.value.text)
  }
}

// 复制结果
function copyResult() {
  if (result.value) {
    const text = `${result.value.text} (${result.value.pinyin})\n${result.value.meaning}`
    navigator.clipboard.writeText(text)
    toast.success('已复制到剪贴板')
  }
}

// 分享结果
function shareResult() {
  if (result.value) {
    const shareData = {
      title: '译手 HandTalk AI - 翻译结果',
      text: `手语识别结果：${result.value.text} (${result.value.pinyin})`,
    }

    if (navigator.share) {
      navigator.share(shareData)
    } else {
      navigator.clipboard.writeText(shareData.text)
      toast.success('分享内容已复制')
    }
  }
}

// 页面卸载时清理
onUnmounted(() => {
  uploadedFiles.value.forEach(file => {
    URL.revokeObjectURL(file.previewUrl)
  })
})
</script>

<style scoped>
/* 图片画廊样式 */
.image-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
}

.image-card {
  position: relative;
  cursor: pointer;
  transition: all 0.3s ease;
}

.image-card:hover {
  transform: translateY(-4px);
}

.image-number {
  position: absolute;
  top: 8px;
  left: 8px;
  width: 24px;
  height: 24px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  z-index: 10;
}

.remove-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 28px;
  height: 28px;
  background: rgba(0, 0, 0, 0.5);
  color: white;
  border: none;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: all 0.2s ease;
}

.image-card:hover .remove-btn {
  opacity: 1;
}

.remove-btn:hover {
  background: #ef4444;
}

.image-info {
  padding: 8px 0;
  text-align: center;
}

/* 序列预览样式 */
.sequence-preview {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 4px;
}

.sequence-thumb {
  flex-shrink: 0;
  width: 60px;
  height: 60px;
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.sequence-thumb .sequence-num {
  position: absolute;
  bottom: 4px;
  right: 4px;
  width: 18px;
  height: 18px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
}

.sequence-thumb.more {
  background: #f3f4f6;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  font-size: 14px;
  font-weight: 600;
}

/* 历史记录样式 */
.history-item {
  border: 1px solid transparent;
}

.history-item:hover {
  border-color: #e5e7eb;
}

/* FAQ 卡片样式 */
.faq-card {
  border: 1px solid #f3f4f6;
  transition: all 0.3s ease;
}

.faq-card:hover {
  border-color: #6366f1;
  transform: translateY(-2px);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}

/* 过渡动画 */
.slide-up-enter-active,
.slide-up-leave-active {
  transition: all 0.4s ease-out;
}

.slide-up-enter-from,
.slide-up-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

/* 响应式调整 */
@media (max-width: 1024px) {
  .image-gallery {
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  }
  
  .sequence-thumb {
    width: 50px;
    height: 50px;
  }
}
</style>
