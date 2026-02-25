<template>
  <div class="pt-16 min-h-screen bg-gray-50">
    <!-- Hero 区域 -->
    <section class="py-20">
      <div class="max-w-4xl mx-auto px-4 text-center">
        <div class="w-20 h-20 rounded-2xl bg-accent-gradient flex items-center justify-center mx-auto mb-6">
          <span class="text-white text-4xl">🤟</span>
        </div>
        <h1 class="text-4xl font-bold text-primary-900 mb-4">关于译手 HandTalk AI</h1>
        <p class="text-xl text-gray-600 mb-8">
          用 AI 技术传递人文关怀，让手语翻译像呼吸一样自然
        </p>
      </div>
    </section>

    <!-- 产品愿景 -->
    <section class="py-16 bg-white">
      <div class="max-w-4xl mx-auto px-4">
        <h2 class="text-2xl font-bold text-primary-900 mb-8 text-center">产品愿景</h2>
        <div class="grid md:grid-cols-3 gap-8">
          <div class="text-center p-6">
            <div class="w-14 h-14 rounded-xl bg-blue-100 flex items-center justify-center mx-auto mb-4">
              <i class="bi bi-heart text-2xl text-blue-600"></i>
            </div>
            <h3 class="font-semibold text-primary-900 mb-2">科技温度</h3>
            <p class="text-gray-600 text-sm">用 AI 技术传递人文关怀</p>
          </div>
          <div class="text-center p-6">
            <div class="w-14 h-14 rounded-xl bg-green-100 flex items-center justify-center mx-auto mb-4">
              <i class="bi bi-lightning text-2xl text-green-600"></i>
            </div>
            <h3 class="font-semibold text-primary-900 mb-2">极简高效</h3>
            <p class="text-gray-600 text-sm">让翻译像呼吸一样自然</p>
          </div>
          <div class="text-center p-6">
            <div class="w-14 h-14 rounded-xl bg-purple-100 flex items-center justify-center mx-auto mb-4">
              <i class="bi bi-universal-access text-2xl text-purple-600"></i>
            </div>
            <h3 class="font-semibold text-primary-900 mb-2">普惠无障碍</h3>
            <p class="text-gray-600 text-sm">人人可用的翻译工具</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 核心功能 -->
    <section class="py-16 bg-gray-50">
      <div class="max-w-4xl mx-auto px-4">
        <h2 class="text-2xl font-bold text-primary-900 mb-8 text-center">核心功能</h2>
        <div class="space-y-6">
          <div class="card p-6 flex items-start gap-4">
            <div class="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
              <i class="bi bi-camera-video text-xl text-accent"></i>
            </div>
            <div>
              <h3 class="font-semibold text-primary-900 mb-1">实时翻译</h3>
              <p class="text-gray-600 text-sm">摄像头实时捕获手语动作，即时转换为文字</p>
            </div>
          </div>
          <div class="card p-6 flex items-start gap-4">
            <div class="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
              <i class="bi bi-cloud-upload text-xl text-accent"></i>
            </div>
            <div>
              <h3 class="font-semibold text-primary-900 mb-1">上传翻译</h3>
              <p class="text-gray-600 text-sm">上传本地视频或图片进行异步翻译</p>
            </div>
          </div>
          <div class="card p-6 flex items-start gap-4">
            <div class="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
              <i class="bi bi-book text-xl text-accent"></i>
            </div>
            <div>
              <h3 class="font-semibold text-primary-900 mb-1">手语词典</h3>
              <p class="text-gray-600 text-sm">常用手语词汇学习，观看标准示范</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 技术栈 -->
    <section class="py-16 bg-white">
      <div class="max-w-4xl mx-auto px-4 text-center">
        <h2 class="text-2xl font-bold text-primary-900 mb-8">技术栈</h2>
        <div class="flex flex-wrap justify-center gap-3">
          <span class="badge badge-outline px-4 py-2">Vue 3</span>
          <span class="badge badge-outline px-4 py-2">Nuxt 3</span>
          <span class="badge badge-outline px-4 py-2">TypeScript</span>
          <span class="badge badge-outline px-4 py-2">Tailwind CSS</span>
          <span class="badge badge-outline px-4 py-2">Pinia</span>
          <span class="badge badge-outline px-4 py-2">WebSocket</span>
          <span class="badge badge-outline px-4 py-2">Python</span>
          <span class="badge badge-outline px-4 py-2">FastAPI</span>
          <span class="badge badge-outline px-4 py-2">MediaPipe</span>
        </div>
      </div>
    </section>

    <!-- 联系我们 -->
    <section class="py-16 bg-gray-50">
      <div class="max-w-4xl mx-auto px-4 text-center">
        <h2 class="text-2xl font-bold text-primary-900 mb-4">联系我们</h2>
        <p class="text-gray-600 mb-6">如果您有任何建议或合作意向，欢迎联系我们</p>
        <div class="flex justify-center gap-4">
          <a href="#" class="btn btn-secondary rounded-full">
            <i class="bi bi-envelope mr-2"></i>联系邮箱
          </a>
          <a href="#" class="btn btn-ghost rounded-full">
            <i class="bi bi-github mr-2"></i>GitHub
          </a>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({
  title: '关于我们 - 译手 HandTalk AI',
  description: '关于译手 HandTalk AI - 用AI技术消除沟通障碍',
})
</script>


