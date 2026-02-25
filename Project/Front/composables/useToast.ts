import { ref } from 'vue'

/**
 * Toast 提示 hook
 */
export function useToast() {
  const toasts = ref<Array<{
    id: string
    type: 'success' | 'error' | 'warning' | 'info'
    message: string
    duration: number
  }>>([])

  function show(message: string, type: 'success' | 'error' | 'warning' | 'info' = 'success', duration = 2500) {
    const id = Date.now().toString()
    const toast = { id, type, message, duration }
    toasts.value.push(toast)

    // 自动移除
    setTimeout(() => {
      remove(id)
    }, duration)

    return id
  }

  function success(message: string, duration?: number) {
    return show(message, 'success', duration)
  }

  function error(message: string, duration?: number) {
    return show(message, 'error', duration)
  }

  function warning(message: string, duration?: number) {
    return show(message, 'warning', duration)
  }

  function info(message: string, duration?: number) {
    return show(message, 'info', duration)
  }

  function remove(id: string) {
    const index = toasts.value.findIndex(t => t.id === id)
    if (index > -1) {
      toasts.value.splice(index, 1)
    }
  }

  function clear() {
    toasts.value = []
  }

  return {
    toasts,
    show,
    success,
    error,
    warning,
    info,
    remove,
    clear,
  }
}


