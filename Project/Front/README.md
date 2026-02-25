# 译手 HandTalk AI - 前端项目

> 面向中国市场的 AI 手语翻译应用前端项目

## 技术栈

- **框架**: Nuxt 3 + Vue 3 Composition API
- **语言**: TypeScript
- **样式**: Tailwind CSS
- **状态管理**: Pinia
- **工具库**: VueUse
- **图表**: ECharts

## 项目结构

```
Front/
├── assets/
│   └── css/
│       └── main.css          # 全局样式
├── components/               # 组件
│   ├── TheHeader.vue         # 顶部导航
│   ├── TheFooter.vue         # 底部
│   └── Toast.vue             # Toast 提示
├── composables/              # 组合式函数
│   ├── useCamera.ts          # 摄像头控制
│   ├── useWebSocket.ts       # WebSocket
│   ├── useToast.ts           # Toast
│   └── useSpeech.ts          # 语音播报
├── layouts/                  # 布局
│   ├── default.vue           # 默认布局
│   ├── auth.vue              # 认证布局
│   └── admin.vue             # 管理后台布局
├── pages/                    # 页面
│   ├── index.vue             # 首页
│   ├── recognize.vue         # 实时翻译
│   ├── translate.vue         # 上传翻译
│   ├── dictionary.vue        # 手语词典
│   └── auth/
│       ├── login.vue         # 登录
│       └── register.vue      # 注册
├── stores/                   # Pinia 状态管理
│   ├── auth.ts               # 认证状态
│   ├── recognition.ts        # 识别状态
│   └── dictionary.ts         # 词典状态
├── types/                    # TypeScript 类型
│   └── index.ts              # 类型定义
├── nuxt.config.ts            # Nuxt 配置
├── tailwind.config.ts        # Tailwind 配置
├── tsconfig.json             # TypeScript 配置
└── package.json              # 依赖配置
```

## 快速开始

### 环境要求

- Node.js 18+ 
- npm / yarn / pnpm

### 安装依赖

```bash
cd Project\Front

# 使用 npm
npm install

# 或使用 pnpm（推荐）
pnpm install

# 或使用 yarn
yarn install
```

### 启动开发服务器

```bash
# 启动开发服务器


# 或
pnpm dev
```

开发服务器运行在 `http://localhost:3000`

### 构建生产版本

```bash
# 构建
npm run build

# 预览生产版本
npm run preview
```

## 核心功能

### 1. 实时翻译 (recognize)

- 摄像头实时捕获手语动作
- WebSocket 实时通信
- 骨架关键点可视化
- 置信度实时显示
- 语音播报

### 2. 上传翻译 (translate)

- 拖拽/点击上传
- 视频/图片支持
- 进度显示
- 历史记录

### 3. 手语词典 (dictionary)

- 词汇搜索
- 分类筛选
- 视频示范
- 收藏功能

### 4. 用户系统 (auth)

- 手机号登录
- 用户注册
- 个人中心

## 配置

### 环境变量

创建 `.env` 文件：

```env
# API 基础地址
API_BASE_URL=http://localhost:8000/api/v1

# WebSocket 地址
WS_URL=ws://localhost:8000
```

### Nuxt 配置

在 `nuxt.config.ts` 中配置：

```typescript
export default defineNuxtConfig({
  modules: [
    '@nuxtjs/tailwindcss',
    '@pinia/nuxt',
    '@vueuse/nuxt',
  ],
  runtimeConfig: {
    public: {
      apiBase: process.env.API_BASE_URL || 'http://localhost:8000/api/v1',
      wsUrl: process.env.WS_URL || 'ws://localhost:8000',
    },
  },
})
```

## 开发规范

### 代码风格

- 使用 TypeScript 严格模式
- 组合式 API (Composition API)
- 组件使用 PascalCase
- 文件命名使用 kebab-case

### 提交规范

```
feat: 新功能
fix: Bug 修复
docs: 文档更新
style: 代码格式
refactor: 重构
perf: 性能优化
test: 测试相关
chore: 构建/工具相关
```

## 相关链接

- [项目设计文档](./front_project.md)
- [后端项目](../Back/)
- [主项目README](../../ShenCode/README.md)

## License

MIT


