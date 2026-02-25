<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>

<script setup lang="ts">
// 全局应用设置
useHead({
  htmlAttrs: {
    lang: 'zh-CN',
  },
})
</script>

<style>
/* 全局过渡动画 */
.page-enter-active,
.page-leave-active {
  transition: all 0.3s ease-out;
}

.page-enter-from,
.page-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>
