<template>
  <div class="min-h-screen flex flex-col">
    <!-- 顶部导航 -->
    <TheHeader />

    <!-- 主内容区 -->
    <main class="flex-1">
      <slot />
    </main>

    <!-- 底部 -->
    <TheFooter />

    <!-- Toast 提示 -->
    <Toast />
  </div>
</template>

<script setup lang="ts">
// 默认布局
</script>


