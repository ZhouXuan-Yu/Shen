<template>
  <div class="pt-16">
    <!-- Hero 区域 -->
    <section class="hero-section relative overflow-hidden">
      <!-- 背景装饰 -->
      <div class="absolute inset-0 overflow-hidden">
        <div class="absolute -top-40 -right-40 w-96 h-96 bg-purple-200 rounded-full opacity-30 blur-3xl"></div>
        <div class="absolute top-1/2 -left-40 w-72 h-72 bg-blue-200 rounded-full opacity-30 blur-3xl"></div>
        <div class="absolute bottom-0 right-1/4 w-64 h-64 bg-pink-200 rounded-full opacity-30 blur-3xl"></div>
      </div>
      
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 relative z-10">
        <div class="text-center">
          <!-- 标签 -->
          <div
            class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 text-accent text-sm font-medium mb-8"
          >
            <i class="bi bi-stars"></i>
            HandTalk AI 2.0 发布
          </div>

          <!-- 主标题 -->
          <h1
            class="text-4xl md:text-5xl lg:text-7xl font-bold text-primary-900 mb-6 leading-tight"
          >
            让手语翻译
            <span class="text-gradient">像呼吸一样自然</span>
          </h1>

          <!-- 副标题 -->
          <p
            class="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            面向中国市场的 AI 手语翻译应用，用科技传递人文关怀，
            消除听障人士与健听人之间的沟通障碍。
          </p>

          <!-- CTA 按钮 -->
          <div
            class="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <NuxtLink to="/recognize" class="btn btn-accent btn-lg btn-pill shadow-lg shadow-accent/30 hover:shadow-xl hover:shadow-accent/40 hover:-translate-y-0.5 transition-all">
              <i class="bi bi-camera-video mr-2"></i>
              开始实时翻译
            </NuxtLink>
            <NuxtLink to="/translate" class="btn btn-secondary btn-lg btn-pill hover:bg-gray-50 transition-all">
              <i class="bi bi-images mr-2"></i>
              图片序列翻译
            </NuxtLink>
            <NuxtLink to="/about" class="btn btn-ghost btn-lg btn-pill">
              了解更多
            </NuxtLink>
          </div>
        </div>
      </div>
    </section>

    <!-- 公告栏 -->
    <section class="bg-gradient-to-r from-primary-900 to-primary-800 text-white py-4">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-center gap-3">
          <i class="bi bi-megaphone-fill text-accent"></i>
          <span class="text-sm font-medium">HandTalk AI 2.0 发布：识别准确率提升至 98%，新增图片序列翻译功能！</span>
        </div>
      </div>
    </section>

    <!-- 功能卡片 -->
    <section class="py-20 bg-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-12">
          <h2 class="text-3xl md:text-4xl font-bold text-primary-900 mb-4">核心功能</h2>
          <p class="text-gray-600 max-w-xl mx-auto text-lg">
            三大核心功能满足不同场景需求，让手语翻译随时随地可用
          </p>
        </div>

        <div class="row g-4">
          <!-- 实时翻译卡片 -->
          <div class="col-12 col-md-4">
            <div class="feature-card card card-hover h-100 border-0 shadow-sm">
              <div class="card-body text-center p-8">
                <div class="feature-icon-wrapper mb-6">
                  <div class="feature-icon-bg"></div>
                  <div class="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
                    <i class="bi bi-camera-video text-3xl text-white"></i>
                  </div>
                </div>
                <h3 class="text-xl font-bold text-primary-900 mb-3">实时翻译</h3>
                <p class="text-gray-600 mb-6">
                  摄像头实时捕获手语动作，即时转换为文字
                </p>
                <div class="flex items-center justify-center gap-4 text-sm text-gray-500 mb-6">
                  <span class="flex items-center gap-1">
                    <i class="bi bi-lightning-charge text-yellow-500"></i>
                    0.1s 延迟
                  </span>
                  <span class="flex items-center gap-1">
                    <i class="bi bi-check-circle text-green-500"></i>
                    99% 准度
                  </span>
                </div>
                <NuxtLink to="/recognize" class="btn btn-primary w-100 rounded-pill">
                  立即开启
                </NuxtLink>
              </div>
            </div>
          </div>

          <!-- 图片序列翻译卡片 -->
          <div class="col-12 col-md-4">
            <div class="feature-card card card-hover h-100 border-0 shadow-sm">
              <div class="card-body text-center p-8">
                <div class="feature-icon-wrapper mb-6">
                  <div class="feature-icon-bg bg-purple-100"></div>
                  <div class="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
                    <i class="bi bi-images text-3xl text-white"></i>
                  </div>
                </div>
                <h3 class="text-xl font-bold text-primary-900 mb-3">图片序列翻译</h3>
                <p class="text-gray-600 mb-6">
                  上传多张连贯图片，通过图片序列分析准确识别完整手语含义
                </p>
                <div class="flex items-center justify-center gap-4 text-sm text-gray-500 mb-6">
                  <span class="flex items-center gap-1">
                    <i class="bi bi-collection text-purple-500"></i>
                    多图支持
                  </span>
                  <span class="flex items-center gap-1">
                    <i class="bi bi-graph-up text-purple-500"></i>
                    98% 准确
                  </span>
                </div>
                <NuxtLink to="/translate" class="btn btn-primary w-100 rounded-pill">
                  立即体验
                </NuxtLink>
              </div>
            </div>
          </div>

          <!-- 手语词典卡片 -->
          <div class="col-12 col-md-4">
            <div class="feature-card card card-hover h-100 border-0 shadow-sm">
              <div class="card-body text-center p-8">
                <div class="feature-icon-wrapper mb-6">
                  <div class="feature-icon-bg bg-green-100"></div>
                  <div class="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center shadow-lg shadow-green-500/30">
                    <i class="bi bi-book text-3xl text-white"></i>
                  </div>
                </div>
                <h3 class="text-xl font-bold text-primary-900 mb-3">手语词典</h3>
                <p class="text-gray-600 mb-6">
                  常用手语词汇学习，观看标准示范视频
                </p>
                <div class="flex items-center justify-center gap-4 text-sm text-gray-500 mb-6">
                  <span class="flex items-center gap-1">
                    <i class="bi bi-collection text-yellow-500"></i>
                    持续更新
                  </span>
                  <span class="flex items-center gap-1">
                    <i class="bi bi-play-circle text-blue-500"></i>
                    视频示范
                  </span>
                </div>
                <NuxtLink to="/dictionary" class="btn btn-primary w-100 rounded-pill">
                  开始学习
                </NuxtLink>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 统计数据 -->
    <section class="py-16 bg-gradient-to-br from-primary-900 via-primary-800 to-purple-900 text-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div class="text-center">
            <div class="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">100K+</div>
            <div class="text-white/60 text-sm">翻译次数</div>
          </div>
          <div class="text-center">
            <div class="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">98%</div>
            <div class="text-white/60 text-sm">准确率</div>
          </div>
          <div class="text-center">
            <div class="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-white to-green-200 bg-clip-text text-transparent">500+</div>
            <div class="text-white/60 text-sm">词条量</div>
          </div>
          <div class="text-center">
            <div class="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-white to-pink-200 bg-clip-text text-transparent">1M+</div>
            <div class="text-white/60 text-sm">用户数</div>
          </div>
        </div>
      </div>
    </section>

    <!-- 使用场景 -->
    <section class="py-20 bg-gray-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-12">
          <h2 class="text-3xl md:text-4xl font-bold text-primary-900 mb-4">适用场景</h2>
          <p class="text-gray-600 max-w-xl mx-auto text-lg">
            无论是在医院、银行还是日常交流，译手随时为您服务
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="scene-card card p-8 bg-white hover:shadow-xl transition-all duration-300">
            <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center mb-6 shadow-lg shadow-blue-500/30">
              <i class="bi bi-hospital text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-primary-900 mb-3">医疗服务</h3>
            <p class="text-gray-600 leading-relaxed">
              听障患者与医生沟通病情，获取准确的医疗服务，保障医患沟通顺畅
            </p>
          </div>

          <div class="scene-card card p-8 bg-white hover:shadow-xl transition-all duration-300 delay-100">
            <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center mb-6 shadow-lg shadow-green-500/30">
              <i class="bi bi-bank text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-primary-900 mb-3">金融服务</h3>
            <p class="text-gray-600 leading-relaxed">
              在银行、邮局等窗口办理业务时顺畅交流，不再因沟通障碍而困扰
            </p>
          </div>

          <div class="scene-card card p-8 bg-white hover:shadow-xl transition-all duration-300 delay-200">
            <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center mb-6 shadow-lg shadow-purple-500/30">
              <i class="bi bi-house-heart text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-primary-900 mb-3">家庭沟通</h3>
            <p class="text-gray-600 leading-relaxed">
              帮助听障家人与健听家人建立更好的沟通桥梁，增进家庭情感交流
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA 区域 -->
    <section class="py-20 bg-gradient-to-br from-purple-600 via-purple-700 to-blue-800 text-white">
      <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 class="text-3xl md:text-4xl font-bold mb-6">准备好开始使用了吗？</h2>
        <p class="text-white/80 text-lg mb-8 max-w-2xl mx-auto">
          立即体验 AI 手语翻译，让沟通无障碍，让世界更温暖
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
          <NuxtLink to="/recognize" class="btn btn-lg bg-white text-purple-700 hover:bg-gray-100 rounded-full px-8 shadow-xl">
            <i class="bi bi-camera-video mr-2"></i>
            立即体验
          </NuxtLink>
          <NuxtLink to="/dictionary" class="btn btn-lg bg-transparent border-2 border-white text-white hover:bg-white/10 rounded-full px-8">
            <i class="bi bi-book mr-2"></i>
            学习手语
          </NuxtLink>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
// SEO 元信息
useSeoMeta({
  title: '译手 HandTalk AI - 让沟通无障碍',
  description: '面向中国市场的 AI 手语翻译应用，用科技传递人文关怀，消除听障人士与健听人之间的沟通障碍。',
  ogTitle: '译手 HandTalk AI - 让沟通无障碍',
  ogDescription: '面向中国市场的 AI 手语翻译应用',
})
</script>

<style scoped>
.hero-section {
  background: linear-gradient(180deg, #f9fafb 0%, #fff 50%, #f0f9ff 100%);
  min-height: 80vh;
  display: flex;
  align-items: center;
}

.feature-card {
  position: relative;
  overflow: hidden;
}

.feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.feature-card:hover::before {
  transform: scaleX(1);
}

.feature-icon-wrapper {
  position: relative;
  display: inline-block;
}

.feature-icon-bg {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100px;
  height: 100px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.5;
  }
  50% {
    transform: translate(-50%, -50%) scale(1.1);
    opacity: 0.8;
  }
}

.scene-card {
  position: relative;
  overflow: hidden;
}

.scene-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.03), rgba(139, 92, 246, 0.03));
  opacity: 0;
  transition: opacity 0.3s ease;
}

.scene-card:hover::before {
  opacity: 1;
}

/* 动画延迟 */
.delay-100 {
  transition-delay: 100ms;
}

.delay-200 {
  transition-delay: 200ms;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .hero-section {
    min-height: auto;
    padding: 60px 0;
  }
}
</style>
