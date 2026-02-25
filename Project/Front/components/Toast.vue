<template>
  <Teleport to="body">
    <div class="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
      <TransitionGroup
        enter-active-class="transition-all duration-300 ease-out"
        leave-active-class="transition-all duration-200 ease-in"
        enter-from-class="opacity-0 translate-x-8"
        enter-to-class="opacity-100 translate-x-0"
        leave-from-class="opacity-100 translate-x-0"
        leave-to-class="opacity-0 translate-x-8"
      >
        <div
          v-for="toast in toasts"
          :key="toast.id"
          class="flex items-center gap-3 px-4 py-3 rounded-xl shadow-lg max-w-sm"
          :class="toastClasses[toast.type]"
        >
          <i :class="iconClasses[toast.type]"></i>
          <span class="text-sm font-medium flex-1">{{ toast.message }}</span>
          <button
            class="opacity-60 hover:opacity-100 transition-opacity"
            @click="removeToast(toast.id)"
          >
            <i class="bi bi-x-lg"></i>
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
// Toast 组件直接使用 composables 中的方法，避免 storeToRefs 类型问题
const { toasts, remove: removeToast } = useToast()

const toastClasses: Record<string, string> = {
  success: 'bg-success text-white',
  error: 'bg-error text-white',
  warning: 'bg-warning text-white',
  info: 'bg-info text-white',
}

const iconClasses: Record<string, string> = {
  success: 'bi bi-check-circle text-lg',
  error: 'bi bi-x-circle text-lg',
  warning: 'bi bi-exclamation-circle text-lg',
  info: 'bi bi-info-circle text-lg',
}
</script>
