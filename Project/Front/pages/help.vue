<template>
  <div class="pt-16 min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <div class="max-w-4xl mx-auto px-4 py-10">
      <h1 class="text-3xl font-bold text-primary-900 mb-4">使用帮助</h1>
      <p class="text-gray-600 mb-6">
        这里将逐步补充完整的使用指南。目前可以通过顶部导航和首页卡片进入各个功能页面：
        实时翻译、图片序列翻译、视频翻译和手语词典。
      </p>
      <ul class="list-disc pl-6 space-y-2 text-gray-700 text-sm">
        <li>如果看到 Vue Router 的路由警告，一般是某些占位页面尚未实现，不影响主要功能使用。</li>
        <li>遇到问题可以返回首页，重新进入对应功能页面再尝试。</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({
  title: '使用帮助 - 译手 HandTalk AI',
})
</script>

