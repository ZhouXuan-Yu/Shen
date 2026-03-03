<template>
  <footer class="bg-gradient-to-b from-white to-gray-50 border-t border-gray-100">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-12">
        <!-- 品牌信息 -->
        <div class="col-span-1 md:col-span-2">
          <div class="flex items-center gap-3 mb-6">
            <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
              <span class="text-white text-xl">🤟</span>
            </div>
            <div>
              <span class="text-2xl font-bold bg-gradient-to-r from-primary-900 to-primary-700 bg-clip-text text-transparent">译手</span>
              <p class="text-xs text-gray-500">HandTalk AI</p>
            </div>
          </div>
          
          <p class="text-gray-600 text-sm leading-relaxed mb-6 max-w-md">
            让手语翻译像呼吸一样自然。我们致力于消除听障人士与健听人之间的沟通障碍，
            用 AI 技术传递人文关怀，让世界更温暖。
          </p>
          
          <!-- 社交媒体 -->
          <div class="flex items-center gap-3">
            <a href="#" class="social-btn">
              <i class="bi bi-wechat"></i>
            </a>
            <a href="#" class="social-btn">
              <i class="bi bi-weibo"></i>
            </a>
            <a href="#" class="social-btn">
              <i class="bi bi-github"></i>
            </a>
            <a href="#" class="social-btn">
              <i class="bi bi-twitter-x"></i>
            </a>
            <a href="#" class="social-btn">
              <i class="bi bi-envelope"></i>
            </a>
          </div>
        </div>

        <!-- 快速链接 -->
        <div>
          <h4 class="font-semibold text-primary-900 mb-6 flex items-center gap-2">
            <i class="bi bi-lightning-charge text-accent"></i>
            快速入口
          </h4>
          <ul class="space-y-3">
            <li>
              <NuxtLink to="/recognize" class="footer-link">
                <i class="bi bi-camera-video"></i>
                实时翻译
              </NuxtLink>
            </li>
            <li>
              <NuxtLink to="/translate" class="footer-link">
                <i class="bi bi-images"></i>
                图片翻译
              </NuxtLink>
            </li>
            <li>
              <NuxtLink to="/dictionary" class="footer-link">
                <i class="bi bi-book"></i>
                手语词典
              </NuxtLink>
            </li>
            <li>
              <NuxtLink to="/about" class="footer-link">
                <i class="bi bi-info-circle"></i>
                关于我们
              </NuxtLink>
            </li>
          </ul>
        </div>

        <!-- 帮助与支持 -->
        <div>
          <h4 class="font-semibold text-primary-900 mb-6 flex items-center gap-2">
            <i class="bi bi-headset text-accent"></i>
            帮助支持
          </h4>
          <ul class="space-y-3">
            <li>
              <NuxtLink to="/help" class="footer-link">
                <i class="bi bi-question-circle"></i>
                使用帮助
              </NuxtLink>
            </li>
            <li>
              <NuxtLink to="/faq" class="footer-link">
                <i class="bi bi-chat-dots"></i>
                常见问题
              </NuxtLink>
            </li>
            <li>
              <NuxtLink to="/contact" class="footer-link">
                <i class="bi bi-telephone"></i>
                联系我们
              </NuxtLink>
            </li>
            <li>
              <NuxtLink to="/privacy" class="footer-link">
                <i class="bi bi-shield-check"></i>
                隐私政策
              </NuxtLink>
            </li>
          </ul>
        </div>
      </div>

      <!-- 统计数据条 -->
      <!-- <div class="mt-12 pt-8 border-t border-gray-200">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div class="text-center">
            <div class="text-3xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">100K+</div>
            <div class="text-sm text-gray-500 mt-1">翻译次数</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold bg-gradient-to-r from-green-500 to-emerald-600 bg-clip-text text-transparent">98%</div>
            <div class="text-sm text-gray-500 mt-1">准确率</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold bg-gradient-to-r from-orange-500 to-amber-600 bg-clip-text text-transparent">500+</div>
            <div class="text-sm text-gray-500 mt-1">词条量</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold bg-gradient-to-r from-pink-500 to-rose-600 bg-clip-text text-transparent">1M+</div>
            <div class="text-sm text-gray-500 mt-1">用户数</div>
          </div>
        </div>
      </div> -->

      <!-- 底部版权 -->
      <div class="mt-12 pt-8 border-t border-gray-200">
        <div class="flex flex-col md:flex-row items-center justify-between gap-4">
          <div class="flex items-center gap-2">
            <span class="text-gray-500 text-sm">© {{ new Date().getFullYear() }} HandTalk AI. 让沟通无障碍。</span>
          </div>
          
          <div class="flex items-center gap-6">
            <div class="flex items-center gap-2 text-gray-400 text-sm">
              <i class="bi bi-heart-fill text-red-500"></i>
              <span>用爱创造价值</span>
            </div>
            <span class="text-gray-300">|</span>
            <span class="text-gray-400 text-sm">AI 手语翻译</span>
          </div>
        </div>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
// Footer 组件
</script>

<style scoped>
.social-btn {
  @apply w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-gray-600 hover:bg-gradient-to-br hover:from-blue-500 hover:to-purple-600 hover:text-white transition-all duration-300;
}

.footer-link {
  @apply flex items-center gap-2 text-gray-600 hover:text-primary-900 text-sm transition-colors duration-200;
}

.footer-link:hover {
  @apply translate-x-1;
}
</style>
