<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <div class="max-w-4xl mx-auto px-4 py-10">
      <h1 class="text-3xl font-bold text-primary-900 mb-4">常见问题</h1>
      <p class="text-gray-600 mb-6">
        这是一个占位页面，用于消除路由告警。后续可以在这里整理更详细的 FAQ。
      </p>
      <ul class="space-y-3 text-gray-700 text-sm">
        <li>
          <span class="font-semibold">Q: 翻译接口 404 怎么办？</span><br>
          <span>A: 请确认后端已在本地运行，且前端环境变量中的 API 地址与后端端口一致。</span>
        </li>
        <li>
          <span class="font-semibold">Q: 视频翻译页面打不开？</span><br>
          <span>A: 请访问地址栏输入 <code>/video-translate</code>，或从首页「图片序列翻译」卡片下方的「视频翻译」按钮进入。</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({
  title: '常见问题 - 译手 HandTalk AI',
})
</script>

