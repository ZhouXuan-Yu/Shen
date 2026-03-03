// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },

  modules: [
    '@nuxtjs/tailwindcss',
    '@pinia/nuxt',
    '@vueuse/nuxt',
  ],

  css: ['~/assets/css/main.css'],

  app: {
    head: {
      title: '译手 HandTalk AI - 让沟通无障碍',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: '译手 (HandTalk AI) - 中国手语实时翻译应用，让沟通无障碍' },
        { name: 'theme-color', content: '#1A1A1A' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap' },
      ],
    },
  },

  runtimeConfig: {
    public: {
      // 默认指向手语识别 FastAPI 服务端口（避免和其他 8000 端口服务冲突）
      apiBase: process.env.API_BASE_URL || 'http://localhost:9000/api/v1',
      // 上传图片 / 视频识别接口单独配置基础地址，避免重复拼接 /api/v1 前缀导致 404
      uploadBase: process.env.UPLOAD_BASE_URL || 'http://localhost:9000',
      wsUrl: process.env.WS_URL || 'ws://localhost:9000',
    },
  },

  typescript: {
    strict: true,
    typeCheck: true,
  },

  compatibilityDate: '2025-01-15',
})


