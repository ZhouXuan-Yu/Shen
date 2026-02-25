# 「译手 (HandTalk AI)」前端开发文档

> 版本：v1.0  
> 更新日期：2026年2月  
> 项目定位：中国手语实时翻译商业级 Web 应用

---

## 目录

1. [项目概述](#1-项目概述)
2. [设计规范](#2-设计规范)
3. [项目结构](#3-项目结构)
4. [页面设计](#4-页面设计)
5. [组件规范](#5-组件规范)
6. [交互规范](#6-交互规范)
7. [API 接口](#7-api-接口)
8. [开发规范](#8-开发规范)

---

## 1. 项目概述

### 1.1 产品愿景

「译手 (HandTalk AI)」是一款面向中国市场的 AI 手语翻译应用，致力于消除听障人士与健听人之间的沟通障碍。产品核心理念：

- **科技温度**：用 AI 技术传递人文关怀
- **极简高效**：让翻译像呼吸一样自然
- **普惠无障碍**：人人可用的翻译工具

### 1.2 目标用户

| 用户群体 | 核心需求 | 使用场景 |
|---------|---------|---------|
| 听障人士 | 与健听人顺畅沟通 | 医院、银行、学校、日常交流 |
| 健听人 | 学习基础手语、关爱特殊群体 | 家庭沟通、公共服务窗口 |
| 手语学习者 | 系统学习手语词汇和表达 | 自学、课堂教学 |

### 1.3 核心功能

| 功能模块 | 功能描述 | 优先级 |
|---------|---------|:------:|
| 实时翻译 | 摄像头实时捕获手语动作，即时输出文字 | P0 |
| 视频/图片翻译 | 上传本地视频或图片进行异步翻译 | P0 |
| 手语词典 | 查询常用手语词汇，观看示范视频 | P0 |
| 翻译历史 | 保存翻译记录，支持回顾和导出 | P1 |
| 用户系统 | 注册登录、个人中心、设置 | P1 |
| 数据后台 | 管理员查看运营数据、用户管理 | P2 |

---

## 2. 设计规范

### 2.1 设计理念

采用「小红书风格的极简高商业感」设计语言，融合 AI 科技感与无障碍友好性。

**核心设计原则**：
- **高对比度**：确保视觉清晰度，便于特殊人群阅读
- **轻量化卡片**：信息模块化，降低认知负担
- **微阴影**：营造层次感，不失轻盈
- **极致圆角**：亲和力与温柔感

### 2.2 色彩系统

```css
/* ==================== 主色系 ==================== */
:root {
  /* 纯白背景 */
  --color-bg: #FFFFFF;
  
  /* 极简黑（文字、主要元素） */
  --color-primary: #1A1A1A;
  --color-primary-light: #333333;
  
  /* 浅靛蓝渐变（品牌辅助色） */
  --color-accent: #6366F1;        /* 靛蓝 */
  --color-accent-light: #A5B4FC;  /* 浅紫蓝 */
  --color-accent-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
  
  /* 功能色 */
  --color-success: #10B981;       /* 成功 - 薄荷绿 */
  --color-warning: #F59E0B;       /* 警告 - 温和橙 */
  --color-error: #EF4444;         /* 错误 - 柔和红 */
  --color-info: #3B82F6;          /* 信息 - 科技蓝 */
  
  /* 背景灰度 */
  --color-gray-50: #F9FAFB;
  --color-gray-100: #F3F4F6;
  --color-gray-200: #E5E7EB;
  --color-gray-300: #D1D5DB;
  --color-gray-400: #9CA3AF;
  --color-gray-500: #6B7280;
  --color-gray-600: #4B5563;
  --color-gray-700: #374151;
  --color-gray-800: #1F2937;
  --color-gray-900: #111827;
  
  /* 边框颜色 */
  --color-border: #E5E7EB;
  --color-border-light: #F3F4F6;
}
```

### 2.3 字体系统

```css
:root {
  /* 字体族 */
  --font-family-sans: 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-family-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
  
  /* 字号体系 */
  --font-size-xs: 12px;    /* 辅助说明 */
  --font-size-sm: 14px;    /* 次要文字 */
  --font-size-base: 16px;  /* 正文（WCAG 最低要求） */
  --font-size-lg: 18px;    /* 强调文字 */
  --font-size-xl: 20px;    /* 小标题 */
  --font-size-2xl: 24px;   /* 标题 */
  --font-size-3xl: 30px;   /* 大标题 */
  --font-size-4xl: 36px;   /* 特大标题 */
  
  /* 字重 */
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  --font-weight-bold: 700;
}
```

### 2.4 圆角与阴影

```css
:root {
  /* 圆角体系 */
  --radius-sm: 8px;      /* 小按钮、标签 */
  --radius-md: 12px;     /* 中卡片、小组件 */
  --radius-lg: 16px;     /* 大卡片、弹窗 */
  --radius-xl: 24px;     /* 特大卡片 */
  --radius-full: 9999px; /* 胶囊按钮、徽章 */
  
  /* 微阴影体系 */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.02);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.04);
  --shadow-lg: 0 10px 30px rgba(0, 0, 0, 0.05);  /* 核心阴影 */
  --shadow-xl: 0 20px 40px rgba(0, 0, 0, 0.08);
  
  /* 悬浮阴影（交互用） */
  --shadow-hover: 0 20px 40px rgba(0, 0, 0, 0.1);
}
```

### 2.5 间距体系

```css
:root {
  /* 8px 基础单位 */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;
  --space-16: 64px;
  --space-20: 80px;
}
```

### 2.6 响应式断点

```css
:root {
  /* 移动优先断点 */
  --breakpoint-xs: 375px;   /* 小手机 */
  --breakpoint-sm: 640px;   /* 大手机 */
  --breakpoint-md: 768px;   /* 平板 */
  --breakpoint-lg: 1024px;  /* 笔记本 */
  --breakpoint-xl: 1280px;  /* 桌面 */
  --breakpoint-2xl: 1536px; /* 大屏 */
}

/* Bootstrap 5 断点对应 */
@media (min-width: 576px) { /* sm */ }
@media (min-width: 768px) { /* md */ }
@media (min-width: 992px) { /* lg */ }
@media (min-width: 1200px) { /* xl */ }
@media (min-width: 1400px) { /* xxl */ }
```

---

## 3. 项目结构

### 3.1 目录架构

```
Project/Front/
│
├── css/                          # 样式文件
│   ├── style.css                 # 主样式文件（包含所有自定义样式）
│   ├── animations.css            # 动画效果
│   └── vendor/                   # 第三方 CSS
│       ├── bootstrap.min.css     # Bootstrap 5
│       ├── bootstrap-icons.css   # Bootstrap Icons
│       ├── aos.css               # AOS 滚动动画
│       └── sweetalert2.min.css   # 弹窗样式
│
├── js/                           # JavaScript 文件
│   ├── main.js                   # 核心逻辑（页面通用功能）
│   ├── camera.js                 # 摄像头与实时识别相关
│   ├── upload.js                 # 文件上传处理
│   ├── dashboard.js              # 数据后台图表
│   ├── data.js                   # Mock 数据与 LocalStorage
│   └── vendor/                   # 第三方 JS
│       ├── jquery.min.js         # jQuery 3.x
│       ├── bootstrap.bundle.min.js # Bootstrap 5
│       ├── aos.js                # AOS 滚动动画
│       ├── echarts.min.js        # ECharts 图表
│       └── sweetalert2.all.min.js # 弹窗
│
├── assets/                       # 静态资源
│   ├── images/                   # 图片资源
│   │   ├── icons/                # SVG 图标
│   │   ├── placeholders/         # 占位图
│   │   └── logos/                # Logo 文件
│   └── videos/                   # 示例视频
│
├── plugins/                      # 自定义插件
│   └── handtalk-voice.js         # 语音播报插件
│
├── index.html                    # 首页（功能入口）
├── live.html                     # 实时手语翻译页
├── upload.html                   # 视频/图片上传翻译页
├── dictionary.html               # 手语词典页
├── orders.html                   # 翻译记录页
├── profile.html                  # 个人中心
├── admin.html                    # 管理后台
│
└── README.md                     # 本文档
```

### 3.2 第三方依赖清单

| 依赖 | 版本 | 用途 | CDN 链接 |
|------|------|------|----------|
| Bootstrap | 5.3.x | CSS 框架、响应式布局 | `https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css` |
| Bootstrap Icons | 1.11.x | 图标库 | `https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css` |
| jQuery | 3.7.x | DOM 操作 | `https://code.jquery.com/jquery-3.7.1.min.js` |
| AOS | 2.3.x | 滚动淡入动画 | `https://unpkg.com/aos@2.3.1/dist/aos.css` |
| ECharts | 5.4.x | 数据可视化 | `https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js` |
| SweetAlert2 | 11.x | 美化弹窗 | `https://cdn.jsdelivr.net/npm/sweetalert2@11.10.0/dist/sweetalert2.min.css` |

### 3.3 引入方式

所有页面统一使用以下基础模板：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="译手 (HandTalk AI) - 中国手语实时翻译应用">
    <title>译手 HandTalk AI - 让沟通无障碍</title>
    
    <!-- CSS -->
    <link rel="stylesheet" href="css/vendor/bootstrap.min.css">
    <link rel="stylesheet" href="css/vendor/bootstrap-icons.css">
    <link rel="stylesheet" href="css/vendor/aos.css">
    <link rel="stylesheet" href="css/vendor/sweetalert2.min.css">
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <!-- 页面内容 -->
    
    <!-- JS -->
    <script src="js/vendor/jquery.min.js"></script>
    <script src="js/vendor/bootstrap.bundle.min.js"></script>
    <script src="js/vendor/aos.js"></script>
    <script src="js/vendor/echarts.min.js"></script>
    <script src="js/vendor/sweetalert2.all.min.js"></script>
    <script src="js/data.js"></script>
    <script src="js/main.js"></script>
    <script src="js/camera.js"></script>
    <script src="js/upload.js"></script>
    <script src="js/dashboard.js"></script>
</body>
</html>
```

---

## 4. 页面设计

### 4.1 首页 (index.html)

**页面定位**：门户入口、功能导航、用户引导

#### 设计布局

```
┌─────────────────────────────────────────────────────────────────┐
│  顶部导航栏                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Logo  │  实时翻译  视频翻译  词典  关于   [登录/注册]   │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  公告栏（胶囊通知）                                             │
│  🚀 HandTalk AI 2.0 发布：识别准确率提升至 98%！                │
├─────────────────────────────────────────────────────────────────┤
│  Hero 区域                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  🤖                                                      │   │
│  │  让手语翻译像呼吸一样自然                                │   │
│  │                                                         │   │
│  │  [开始翻译] [了解更多]                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  功能卡片网格（响应式）                                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                   │
│  │  📹       │ │  📤       │ │  📖       │                   │
│  │  实时翻译  │ │  上传翻译  │ │  手语词典  │                   │
│  │  0.1s延迟  │ │  批量处理  │ │  500+词汇 │                   │
│  │ [立即开启] │ │ [立即体验] │ │ [开始学习] │                   │
│  └───────────┘ └───────────┘ └───────────┘                   │
├─────────────────────────────────────────────────────────────────┤
│  统计数据                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
│  │ 100k+   │ │ 98%     │ │ 500+    │ │ 1M+     │             │
│  │ 翻译次数  │ │ 准确率   │ │ 词条量   │ │ 用户数   │             │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  底部                                                          │
│  © 2024 HandTalk AI. 让沟通无障碍。                            │
└─────────────────────────────────────────────────────────────────┘
```

#### 功能卡片 HTML 结构

```html
<!-- 功能卡片网格 -->
<div class="row g-4">
    <!-- 实时翻译卡片 -->
    <div class="col-6 col-lg-4" data-aos="fade-up" data-aos-delay="100">
        <div class="card feature-card h-100 border-0 shadow-sm">
            <!-- AI 2.0 徽章 -->
            <div class="feature-badge">
                <span class="badge bg-dark">AI 2.0</span>
            </div>
            
            <div class="card-body p-4 text-center">
                <!-- 图标 -->
                <div class="feature-icon mb-3">
                    <i class="bi bi-camera-video text-primary fs-1"></i>
                </div>
                
                <!-- 标题 -->
                <h5 class="card-title fw-bold mb-2">实时翻译</h5>
                
                <!-- 描述 -->
                <p class="card-text text-muted small mb-3">
                    摄像头实时捕获手语动作，即时转换为文字
                </p>
                
                <!-- 统计数据 -->
                <div class="feature-stats d-flex justify-content-center gap-4 mb-3">
                    <span class="text-muted small">
                        <i class="bi bi-lightning-charge text-warning"></i> 0.1s延迟
                    </span>
                    <span class="text-muted small">
                        <i class="bi bi-check-circle text-success"></i> 99%准度
                    </span>
                </div>
            </div>
            
            <!-- 按钮 -->
            <div class="card-footer bg-white border-0 pb-4">
                <button class="btn btn-dark w-100 rounded-pill py-2" 
                        onclick="location.href='live.html'">
                    立即开启
                </button>
            </div>
        </div>
    </div>
    
    <!-- 视频/图片翻译卡片 -->
    <div class="col-6 col-lg-4" data-aos="fade-up" data-aos-delay="200">
        <div class="card feature-card h-100 border-0 shadow-sm">
            <div class="feature-badge">
                <span class="badge bg-primary-custom">批量处理</span>
            </div>
            
            <div class="card-body p-4 text-center">
                <div class="feature-icon mb-3">
                    <i class="bi bi-cloud-upload text-primary fs-1"></i>
                </div>
                
                <h5 class="card-title fw-bold mb-2">上传翻译</h5>
                <p class="card-text text-muted small mb-3">
                    上传本地视频或图片，支持批量处理
                </p>
                
                <div class="feature-stats d-flex justify-content-center gap-4 mb-3">
                    <span class="text-muted small">
                        <i class="bi bi-film text-info"></i> 视频支持
                    </span>
                    <span class="text-muted small">
                        <i class="bi bi-image text-success"></i> 图片支持
                    </span>
                </div>
            </div>
            
            <div class="card-footer bg-white border-0 pb-4">
                <button class="btn btn-dark w-100 rounded-pill py-2"
                        onclick="location.href='upload.html'">
                    立即体验
                </button>
            </div>
        </div>
    </div>
    
    <!-- 手语词典卡片 -->
    <div class="col-6 col-lg-4" data-aos="fade-up" data-aos-delay="300">
        <div class="card feature-card h-100 border-0 shadow-sm">
            <div class="feature-badge">
                <span class="badge bg-success-custom">500+ 词汇</span>
            </div>
            
            <div class="card-body p-4 text-center">
                <div class="feature-icon mb-3">
                    <i class="bi bi-book text-primary fs-1"></i>
                </div>
                
                <h5 class="card-title fw-bold mb-2">手语词典</h5>
                <p class="card-text text-muted small mb-3">
                    常用手语词汇学习，观看标准示范
                </p>
                
                <div class="feature-stats d-flex justify-content-center gap-4 mb-3">
                    <span class="text-muted small">
                        <i class="bi bi-collection text-warning"></i> 持续更新
                    </span>
                    <span class="text-muted small">
                        <i class="bi bi-play-circle text-info"></i> 视频示范
                    </span>
                </div>
            </div>
            
            <div class="card-footer bg-white border-0 pb-4">
                <button class="btn btn-dark w-100 rounded-pill py-2"
                        onclick="location.href='dictionary.html'">
                    开始学习
                </button>
            </div>
        </div>
    </div>
</div>
```

#### 首页 CSS 样式

```css
/* ==================== 功能卡片样式 ==================== */
.feature-card {
    border-radius: var(--radius-lg);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-hover);
}

.feature-badge {
    position: absolute;
    top: 16px;
    right: 16px;
    z-index: 1;
}

.feature-badge .badge {
    font-weight: 500;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: var(--radius-full);
}

.feature-icon {
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.bg-primary-custom {
    background: var(--color-accent-gradient) !important;
}

.bg-success-custom {
    background: var(--color-success) !important;
}

/* ==================== Hero 区域 ==================== */
.hero-section {
    padding: 80px 0;
    text-align: center;
    background: linear-gradient(180deg, var(--color-gray-50) 0%, #fff 100%);
}

.hero-title {
    font-size: clamp(32px, 5vw, 48px);
    font-weight: 700;
    color: var(--color-primary);
    margin-bottom: 16px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 18px;
    color: var(--color-gray-600);
    margin-bottom: 32px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.hero-buttons .btn {
    padding: 12px 32px;
    font-size: 16px;
    font-weight: 500;
    border-radius: var(--radius-full);
}

/* ==================== 统计数据 ==================== */
.stats-section {
    background: var(--color-primary);
    padding: 60px 0;
}

.stat-item {
    text-align: center;
    color: #fff;
}

.stat-number {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 8px;
}

.stat-label {
    font-size: 14px;
    opacity: 0.8;
}
```

### 4.2 实时翻译页 (live.html)

**页面定位**：核心功能页，摄像头实时捕获与识别

#### 设计布局

```
┌─────────────────────────────────────────────────────────────────┐
│  返回  │  实时手语翻译  │  🔗 分享                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    摄像头预览区域                         │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │                                               │    │   │
│  │  │         [ 实时视频流 + 骨架叠加层  ]            │    │   │
│  │  │                                               │    │   │
│  │  │         🤚 手掌关键点可视化                     │    │   │
│  │  │                                               │    │   │
│  │  │         ⭐ 面部表情关键点可视化                 │    │   │
│  │  │                                               │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  🎯 置信度: ██████████████████░░░░░  85%       │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              悬浮控制栏                                   │   │
│  │  [🔄 镜像] [💡 补光] [🔊 音效] [📸 拍照] [⏸️ 暂停]     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  识别结果（气泡式控制台）                                 │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  你好 (Nǐ hǎo)                                  │    │   │
│  │  │  ─────────────────────────────                  │    │   │
│  │  │  📝 完整句子: 我今天很高兴见到你                │    │   │
│  │  │                                                 │    │   │
│  │  │  [🔊 播放]  [📋 复制]  [📤 分享]                │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 核心代码结构

```html
<!-- 实时翻译页面 -->
<div class="live-page">
    <!-- 顶部导航 -->
    <nav class="live-nav">
        <a href="index.html" class="nav-back">
            <i class="bi bi-arrow-left"></i> 返回
        </a>
        <h5 class="nav-title">实时手语翻译</h5>
        <button class="btn btn-link text-dark" onclick="shareResult()">
            <i class="bi bi-share"></i>
        </button>
    </nav>
    
    <!-- 摄像头容器 -->
    <div class="camera-container">
        <div class="camera-wrapper">
            <!-- 视频流 -->
            <video id="liveVideo" class="live-video" autoplay playsinline></video>
            
            <!-- 骨架叠加 Canvas -->
            <canvas id="skeletonCanvas" class="skeleton-overlay"></canvas>
            
            <!-- 置信度指示器 -->
            <div class="confidence-indicator">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: 85%"></div>
                </div>
                <span class="confidence-text">85%</span>
            </div>
        </div>
    </div>
    
    <!-- 悬浮控制栏 -->
    <div class="control-bar">
        <button class="control-btn" id="mirrorBtn" title="镜像切换">
            <i class="bi bi-arrow-left-right"></i>
        </button>
        <button class="control-btn" id="lightBtn" title="补光">
            <i class="bi bi-lightning-charge"></i>
        </button>
        <button class="control-btn" id="soundBtn" title="音效">
            <i class="bi bi-volume-up"></i>
        </button>
        <button class="control-btn" id="captureBtn" title="拍照">
            <i class="bi bi-camera"></i>
        </button>
        <button class="control-btn active" id="stopBtn" title="停止">
            <i class="bi bi-stop-circle"></i>
        </button>
    </div>
    
    <!-- 识别结果 -->
    <div class="result-panel">
        <div class="result-card">
            <div class="result-header">
                <span class="result-label">识别结果</span>
                <span class="result-time">14:30:25</span>
            </div>
            
            <div class="result-content">
                <h3 class="result-main">你好</h3>
                <p class="result-pinyin">Nǐ hǎo</p>
            </div>
            
            <div class="result-actions">
                <button class="btn btn-outline-dark btn-sm rounded-pill" onclick="playVoice()">
                    <i class="bi bi-volume-up"></i> 播放
                </button>
                <button class="btn btn-outline-dark btn-sm rounded-pill" onclick="copyResult()">
                    <i class="bi bi-clipboard"></i> 复制
                </button>
                <button class="btn btn-outline-dark btn-sm rounded-pill" onclick="shareResult()">
                    <i class="bi bi-share"></i> 分享
                </button>
            </div>
        </div>
    </div>
</div>
```

#### 实时翻译 CSS

```css
/* ==================== 实时翻译页面 ==================== */
.live-page {
    min-height: 100vh;
    background: var(--color-gray-900);
    position: relative;
}

/* 顶部导航 */
.live-nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    z-index: 100;
}

.nav-back {
    color: #fff;
    text-decoration: none;
    font-size: 14px;
}

.nav-title {
    color: #fff;
    font-size: 16px;
    font-weight: 500;
}

/* 摄像头容器 */
.camera-container {
    position: relative;
    width: 100%;
    height: 100vh;
    overflow: hidden;
}

.camera-wrapper {
    position: relative;
    width: 100%;
    height: 100%;
}

.live-video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1);  /* 镜像显示 */
}

.skeleton-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}

/* 置信度指示器 */
.confidence-indicator {
    position: absolute;
    top: 72px;
    left: 16px;
    right: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.confidence-bar {
    flex: 1;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: var(--radius-full);
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-success), var(--color-accent));
    border-radius: var(--radius-full);
    transition: width 0.3s ease;
}

.confidence-text {
    color: #fff;
    font-size: 14px;
    font-weight: 500;
    min-width: 40px;
}

/* 悬浮控制栏 */
.control-bar {
    position: fixed;
    bottom: 100px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 16px;
    padding: 12px 20px;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: var(--radius-full);
    box-shadow: var(--shadow-xl);
    z-index: 100;
}

.control-btn {
    width: 48px;
    height: 48px;
    border: none;
    background: var(--color-gray-100);
    border-radius: 50%;
    font-size: 20px;
    color: var(--color-gray-700);
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}

.control-btn:hover {
    background: var(--color-primary);
    color: #fff;
}

.control-btn.active {
    background: var(--color-primary);
    color: #fff;
}

/* 识别结果面板 */
.result-panel {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 20px;
    background: #fff;
    border-radius: 24px 24px 0 0;
    box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.1);
    z-index: 100;
}

.result-card {
    max-width: 600px;
    margin: 0 auto;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.result-label {
    font-size: 13px;
    color: var(--color-gray-500);
}

.result-time {
    font-size: 13px;
    color: var(--color-gray-400);
}

.result-main {
    font-size: 28px;
    font-weight: 700;
    color: var(--color-primary);
    margin-bottom: 4px;
}

.result-pinyin {
    font-size: 16px;
    color: var(--color-gray-500);
    margin-bottom: 16px;
}

.result-actions {
    display: flex;
    gap: 8px;
}
```

### 4.3 上传翻译页 (upload.html)

**页面定位**：异步翻译，支持视频和图片上传

#### 设计布局

```
┌─────────────────────────────────────────────────────────────────┐
│  返回  │  上传翻译  │  个人中心                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  上传区域                                                  │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  📁                                                │    │   │
│  │  │         拖拽文件到此处，或                        │    │   │
│  │  │         <UButton>点击上传</UButton>               │    │   │
│  │  │                                                 │    │   │
│  │  │  支持 MP4, MOV, PNG, JPG                         │    │   │
│  │  │  单个文件大小限制: 100MB                          │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  上传进度（显示详细进度）                                 │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  ⏳ AI 正在解析第 45/100 帧...                   │    │   │
│  │  │  ████████████████░░░░░░░░░░░░░  45%             │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  翻译结果（对比视图）                                     │   │
│  │  ┌───────────────┬─────────────────────────────────┐    │   │
│  │  │               │                                 │    │   │
│  │  │   原始视频    │    翻译结果                      │    │   │
│  │  │   / 图片      │                                 │    │   │
│  │  │               │   手语识别: 你好                 │    │   │
│  │  │   ▶️ 播放     │   释义: 用于问候                 │    │   │
│  │  │   🔊 音量     │   拼音: Nǐ hǎo                  │    │   │
│  │  │               │                                 │    │   │
│  │  │               │   ─────────────────────         │    │   │
│  │  │               │   📋 一键复制                    │    │   │
│  │  │               │   🔊 语音播放                    │    │   │
│  │  │               │   📤 分享                        │    │   │
│  │  └───────────────┴─────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  历史记录（瀑布流卡片）                                   │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                   │   │
│  │  │ 缩略图  │ │ 缩略图  │ │ 缩略图  │                   │   │
│  │  │ 结果..  │ │ 结果..  │ │ 结果..  │                   │   │
│  │  │ 14:30   │ │ 14:15   │ │ 13:50   │                   │   │
│  │  └─────────┘ └─────────┘ └─────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### 上传区域 HTML

```html
<!-- 上传区域 -->
<div class="upload-section">
    <div class="upload-zone" id="uploadZone">
        <div class="upload-content text-center py-5">
            <i class="bi bi-cloud-upload fs-1 text-muted mb-3"></i>
            <h5 class="fw-bold mb-2">拖拽文件到此处</h5>
            <p class="text-muted small mb-3">或点击下方按钮选择文件</p>
            
            <label class="btn btn-dark rounded-pill px-4">
                选择文件
                <input type="file" class="d-none" id="fileInput" 
                       accept="video/*,image/*" multiple>
            </label>
            
            <p class="text-muted small mt-3 mb-0">
                支持 MP4, MOV, PNG, JPG | 单个文件最大 100MB
            </p>
        </div>
        
        <!-- 进度显示（隐藏） -->
        <div class="upload-progress d-none" id="uploadProgress">
            <div class="text-center mb-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
            
            <h6 class="fw-bold mb-2">AI 正在解析...</h6>
            <p class="text-muted small mb-3" id="progressDetail">
                正在解析第 45/100 帧
            </p>
            
            <div class="progress" style="height: 8px;">
                <div class="progress-bar" id="progressBar" style="width: 45%"></div>
            </div>
        </div>
    </div>
</div>

<!-- 文件列表 -->
<div class="file-list mt-4">
    <h6 class="fw-bold mb-3">已上传文件</h6>
    <div class="row g-3" id="fileList">
        <!-- 由 JS 动态生成 -->
    </div>
</div>
```

#### 对比视图 HTML

```html
<!-- 对比视图 -->
<div class="comparison-section mt-4 d-none" id="comparisonSection">
    <div class="row">
        <!-- 原始文件预览 -->
        <div class="col-md-6 mb-3 mb-md-0">
            <div class="preview-card">
                <div class="preview-header">
                    <span>原始文件</span>
                </div>
                <div class="preview-body">
                    <video id="previewVideo" class="w-100 rounded" controls></video>
                    <img id="previewImage" class="w-100 rounded d-none">
                </div>
            </div>
        </div>
        
        <!-- 翻译结果 -->
        <div class="col-md-6">
            <div class="result-card-custom">
                <div class="result-header-custom">
                    <span>翻译结果</span>
                </div>
                <div class="result-body-custom p-3">
                    <div class="recognition-result mb-3">
                        <label class="text-muted small mb-1">手语识别</label>
                        <h4 class="fw-bold">你好</h4>
                    </div>
                    
                    <div class="recognition-result mb-3">
                        <label class="text-muted small mb-1">拼音</label>
                        <h5 class="text-muted">Nǐ hǎo</h5>
                    </div>
                    
                    <div class="recognition-result mb-3">
                        <label class="text-muted small mb-1">释义</label>
                        <p class="mb-0">用于问候别人，相当于 Hello</p>
                    </div>
                    
                    <hr class="my-3">
                    
                    <div class="d-flex gap-2">
                        <button class="btn btn-outline-dark flex-fill rounded-pill">
                            <i class="bi bi-volume-up"></i> 播放
                        </button>
                        <button class="btn btn-outline-dark flex-fill rounded-pill" onclick="copyText('你好')">
                            <i class="bi bi-clipboard"></i> 复制
                        </button>
                        <button class="btn btn-outline-dark flex-fill rounded-pill">
                            <i class="bi bi-share"></i> 分享
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

### 4.4 手语词典页 (dictionary.html)

**页面定位**：词汇查询与学习

#### 设计布局

```
┌─────────────────────────────────────────────────────────────────┐
│  返回  │  手语词典  │  🔍                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  🔍 搜索栏                                               │   │
│  │  [ 搜索手语词汇...                    ] [ 🎤 语音搜索]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  分类标签                                                │   │
│  │  [全部] [日常问候] [数字] [家庭] [工作] [情绪] [更多 ▾]  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  词汇卡片网格                                             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │  👋     │ │  👋🏻    │ │  👍     │ │  👎     │        │   │
│  │  │  你好   │ │  再见   │ │  好     │ │  不好   │        │   │
│  │  │  nǐ hào│ │  zài jiàn│ │  hǎo    │ │  bù hǎo │        │   │
│  │  │ [播放]  │ │ [播放]  │ │ [播放]  │ │ [播放]  │        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │  ❤️     │ │  💕     │ │  👨     │ │  👩     │        │   │
│  │  │  爱     │ │  喜欢   │ │  爸爸   │ │  妈妈   │        │   │
│  │  │ [播放]  │ │ [播放]  │ │ [播放]  │ │ [播放]  │        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  🏷️ 最近搜索  |  🔥 热门搜索  |  ⭐ 收藏列表             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 词汇卡片 HTML

```html
<!-- 词汇卡片网格 -->
<div class="row g-3" id="wordGrid">
    <!-- 示例卡片 -->
    <div class="col-6 col-md-3">
        <div class="word-card" onclick="showWordDetail('你好')">
            <div class="word-thumbnail">
                <img src="assets/images/placeholders/你好.jpg" alt="你好">
                <button class="play-btn">
                    <i class="bi bi-play-circle-fill"></i>
                </button>
            </div>
            <div class="word-info">
                <h6 class="word-chinese mb-1">你好</h6>
                <p class="word-pinyin text-muted small mb-0">nǐ hǎo</p>
            </div>
        </div>
    </div>
</div>

<!-- 词汇详情弹窗 -->
<div class="modal fade" id="wordDetailModal" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-header border-0">
                <h5 class="modal-title">你好</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <div class="text-center mb-3">
                    <video class="w-100 rounded" controls poster="assets/images/placeholders/你好.jpg">
                        <source src="assets/videos/你好.mp4" type="video/mp4">
                    </video>
                </div>
                
                <div class="word-detail-info">
                    <div class="mb-3">
                        <label class="text-muted small">拼音</label>
                        <p class="fw-bold mb-0">nǐ hǎo</p>
                    </div>
                    
                    <div class="mb-3">
                        <label class="text-muted small">释义</label>
                        <p class="mb-0">用于问候别人，相当于 "Hello"</p>
                    </div>
                    
                    <div class="mb-3">
                        <label class="text-muted small">用法示例</label>
                        <p class="mb-0">"你好，我叫小明。" —— 初次见面时的问候语</p>
                    </div>
                </div>
            </div>
            <div class="modal-footer border-0">
                <button type="button" class="btn btn-outline-dark rounded-pill">
                    <i class="bi bi-heart"></i> 收藏
                </button>
                <button type="button" class="btn btn-dark rounded-pill">
                    <i class="bi bi-share"></i> 分享
                </button>
            </div>
        </div>
    </div>
</div>
```

#### 词汇卡片 CSS

```css
/* ==================== 词汇卡片 ==================== */
.word-card {
    background: #fff;
    border-radius: var(--radius-lg);
    overflow: hidden;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: var(--shadow-sm);
}

.word-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
}

.word-thumbnail {
    position: relative;
    aspect-ratio: 1;
    background: var(--color-gray-100);
    overflow: hidden;
}

.word-thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.play-btn {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 48px;
    height: 48px;
    border: none;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 50%;
    font-size: 24px;
    color: var(--color-primary);
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.3s;
}

.word-card:hover .play-btn {
    opacity: 1;
}

.word-info {
    padding: 12px;
    text-align: center;
}

.word-chinese {
    font-size: 16px;
    font-weight: 600;
    color: var(--color-primary);
}

.word-pinyin {
    font-size: 12px;
}
```

### 4.5 翻译记录页 (orders.html)

**页面定位**：历史记录管理，类似小红书瀑布流

#### 设计布局

```
┌─────────────────────────────────────────────────────────────────┐
│  返回  │  翻译记录  │  🗑️ 批量删除                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  统计概览                                                 │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                   │   │
│  │  │  128    │ │  32     │ │  85%    │                   │   │
│  │  │  总次数  │ │  今日   │ │  平均准度 │                   │   │
│  │  └─────────┘ └─────────┘ └─────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  筛选标签                                                 │   │
│  │  [全部] [实时] [图片] [视频] [今天] [本周]               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  记录列表（瀑布流卡片）                                   │   │
│  │  ┌─────────────┐ ┌─────────────┐                       │   │
│  │  │  📹        │ │  🖼️        │                       │   │
│  │  │ 缩略图      │ │ 缩略图      │                       │   │
│  │  │ ────────   │ │ ────────   │                       │   │
│  │  │ 你好        │ │ 谢谢        │                       │   │
│  │  │ 14:30      │ │ 13:45      │                       │   │
│  │  │ ✅ 96%     │ │ ✅ 92%     │                       │   │
│  │  └─────────────┘ └─────────────┘                       │   │
│  │  ┌─────────────┐ ┌─────────────┐                       │   │
│  │  │  📹        │ │  📹        │                       │   │
│  │  │  ...       │ │  ...       │                       │   │
│  │  └─────────────┘ └─────────────┘                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 记录卡片 HTML

```html
<!-- 记录卡片 -->
<div class="row g-3" id="historyGrid">
    <div class="col-6 col-md-4 col-lg-3">
        <div class="history-card" data-id="1">
            <!-- 选中checkbox -->
            <div class="select-checkbox">
                <input type="checkbox" class="form-check-input">
            </div>
            
            <!-- 缩略图 -->
            <div class="history-thumbnail">
                <img src="assets/images/placeholders/thumb_1.jpg" alt="缩略图">
                <span class="media-type-badge">
                    <i class="bi bi-camera-video"></i>
                </span>
            </div>
            
            <!-- 详情 -->
            <div class="history-content">
                <h6 class="history-result fw-bold">你好</h6>
                <p class="history-time text-muted small">今天 14:30</p>
                
                <div class="history-meta d-flex justify-content-between align-items-center">
                    <span class="confidence-badge">
                        <i class="bi bi-check-circle text-success"></i> 96%
                    </span>
                    <button class="btn btn-sm btn-outline-secondary rounded-pill">
                        查看
                    </button>
                </div>
            </div>
        </div>
    </div>
</div>
```

#### 记录卡片 CSS

```css
/* ==================== 记录卡片 ==================== */
.history-card {
    background: #fff;
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s;
    position: relative;
}

.history-card:hover {
    box-shadow: var(--shadow-md);
}

.select-checkbox {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 10;
    opacity: 0;
    transition: opacity 0.2s;
}

.history-card:hover .select-checkbox,
.select-mode .select-checkbox {
    opacity: 1;
}

.history-thumbnail {
    position: relative;
    aspect-ratio: 16/9;
    background: var(--color-gray-100);
}

.history-thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.media-type-badge {
    position: absolute;
    bottom: 8px;
    right: 8px;
    width: 28px;
    height: 28px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 14px;
}

.history-content {
    padding: 12px;
}

.history-result {
    font-size: 15px;
    color: var(--color-primary);
    margin-bottom: 4px;
}

.history-time {
    font-size: 12px;
}

.confidence-badge {
    font-size: 12px;
    color: var(--color-success);
}
```

### 4.6 管理后台 (admin.html)

**页面定位**：数据分析与系统管理

#### 设计布局

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Logo   │ 仪表盘  用户管理  词条管理  数据分析  设置      │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  欢迎回来，管理员                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  统计卡片                                                 │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ 100,234 │ │  98.5%  │ │  5,678  │ │  1,234  │       │   │
│  │  │ 翻译总量 │ │ 准确率  │ │  用户数  │ │  今日量  │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  实时翻译流量（折线图）                                   │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │                                                 │    │   │
│  │  │         ECharts 图表区域                         │    │   │
│  │  │                                                 │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌───────────────────────────┐ ┌───────────────────────────┐   │
│  │  手语类别分布（饼图）     │ │  识别准确率趋势           │   │
│  │  ┌───────────────────┐   │ │  ┌───────────────────┐    │   │
│  │  │                   │   │ │  │                   │    │   │
│  │  │    ECharts 图表   │   │ │  │    ECharts 图表   │    │   │
│  │  │                   │   │ │  │                   │    │   │
│  │  └───────────────────┘   │ │  └───────────────────┘    │   │
│  └───────────────────────────┘ └───────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  最新翻译记录（表格）                                     │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ 用户  │  类型 │  结果  │ 置信度 │ 时间          │    │   │
│  │  ├───────┼───────┼────────┼────────┼──────────────┤    │   │
│  │  │ 张*明 │ 实时  │ 你好   │ 96%    │ 14:30:25     │    │   │
│  │  │ 李*红 │ 图片  │ 谢谢   │ 92%    │ 14:28:10     │    │   │
│  │  │ ...   │ ...   │ ...    │ ...    │ ...          │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### 管理后台 HTML

```html
<!-- 管理后台 -->
<div class="admin-layout d-flex">
    <!-- 侧边栏 -->
    <nav class="admin-sidebar">
        <div class="sidebar-header">
            <h4 class="text-white mb-0">HandTalk</h4>
            <small class="text-white-50">Admin Panel</small>
        </div>
        
        <ul class="nav flex-column">
            <li class="nav-item">
                <a class="nav-link active" href="#dashboard" data-bs-toggle="tab">
                    <i class="bi bi-grid-1x2 me-2"></i> 仪表盘
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#users" data-bs-toggle="tab">
                    <i class="bi bi-people me-2"></i> 用户管理
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#dictionary" data-bs-toggle="tab">
                    <i class="bi bi-book me-2"></i> 词条管理
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#analytics" data-bs-toggle="tab">
                    <i class="bi bi-bar-chart-line me-2"></i> 数据分析
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#settings" data-bs-toggle="tab">
                    <i class="bi bi-gear me-2"></i> 系统设置
                </a>
            </li>
        </ul>
    </nav>
    
    <!-- 主内容区 -->
    <main class="admin-main flex-grow-1">
        <div class="tab-content">
            <!-- 仪表盘 -->
            <div class="tab-pane fade show active" id="dashboard">
                <!-- 统计卡片 -->
                <div class="row g-4 mb-4">
                    <div class="col-md-6 col-lg-3">
                        <div class="stat-card">
                            <div class="stat-icon bg-primary-light">
                                <i class="bi bi-translate text-primary"></i>
                            </div>
                            <div class="stat-info">
                                <h4 class="stat-number">100,234</h4>
                                <p class="stat-label text-muted mb-0">翻译总量</p>
                            </div>
                        </div>
                    </div>
                    <!-- 更多统计卡片... -->
                </div>
                
                <!-- 图表区域 -->
                <div class="row g-4 mb-4">
                    <div class="col-lg-8">
                        <div class="chart-card">
                            <div class="chart-header">
                                <h5 class="chart-title">实时翻译流量</h5>
                            </div>
                            <div class="chart-body">
                                <div id="trafficChart" style="height: 300px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4">
                        <div class="chart-card">
                            <div class="chart-header">
                                <h5 class="chart-title">手语类别分布</h5>
                            </div>
                            <div class="chart-body">
                                <div id="categoryPie" style="height: 300px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 记录表格 -->
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">最新翻译记录</h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table table-hover mb-0">
                                <thead class="table-light">
                                    <tr>
                                        <th>用户</th>
                                        <th>类型</th>
                                        <th>结果</th>
                                        <th>置信度</th>
                                        <th>时间</th>
                                        <th>操作</th>
                                    </tr>
                                </thead>
                                <tbody id="recentRecords">
                                    <!-- 动态生成 -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>
</div>
```

#### 管理后台 CSS

```css
/* ==================== 管理后台 ==================== */
.admin-layout {
    min-height: 100vh;
}

/* 侧边栏 */
.admin-sidebar {
    width: 240px;
    background: var(--color-primary);
    color: #fff;
    padding: 20px 0;
    position: fixed;
    height: 100vh;
    overflow-y: auto;
}

.sidebar-header {
    padding: 0 20px 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

.admin-sidebar .nav-link {
    color: rgba(255, 255, 255, 0.7);
    padding: 12px 20px;
    transition: all 0.2s;
}

.admin-sidebar .nav-link:hover,
.admin-sidebar .nav-link.active {
    color: #fff;
    background: rgba(255, 255, 255, 0.1);
}

/* 主内容区 */
.admin-main {
    margin-left: 240px;
    padding: 24px;
    background: var(--color-gray-50);
    min-height: 100vh;
}

/* 统计卡片 */
.stat-card {
    background: #fff;
    border-radius: var(--radius-lg);
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: var(--shadow-sm);
}

.stat-icon {
    width: 56px;
    height: 56px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.stat-number {
    font-size: 28px;
    font-weight: 700;
    color: var(--color-primary);
    margin-bottom: 4px;
}

/* 图表卡片 */
.chart-card {
    background: #fff;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.chart-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-border);
}

.chart-title {
    font-size: 16px;
    font-weight: 600;
    margin: 0;
}

.chart-body {
    padding: 20px;
}
```

---

## 5. 组件规范

### 5.1 按钮组件

```html
<!-- Primary 按钮 -->
<button class="btn btn-primary-custom">
    按钮文案
</button>

<!-- Outline 按钮 -->
<button class="btn btn-outline-dark">
    按钮文案
</button>

<!-- 圆形图标按钮 -->
<button class="btn-icon rounded-circle">
    <i class="bi bi-icon-name"></i>
</button>

<!-- 胶囊按钮 -->
<button class="btn btn-dark rounded-pill px-4">
    按钮文案
</button>
```

```css
/* ==================== 按钮样式 ==================== */
.btn-primary-custom {
    background: var(--color-accent-gradient);
    border: none;
    color: #fff;
    font-weight: 500;
    transition: all 0.3s;
}

.btn-primary-custom:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: #fff;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
}

.btn-icon {
    width: 40px;
    height: 40px;
    border: none;
    background: var(--color-gray-100);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-icon:hover {
    background: var(--color-primary);
    color: #fff;
}
```

### 5.2 卡片组件

```html
<div class="card custom-card border-0 shadow-sm">
    <div class="card-body">
        <!-- 内容 -->
    </div>
</div>
```

```css
/* ==================== 卡片样式 ==================== */
.custom-card {
    border-radius: var(--radius-lg);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.custom-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
}
```

### 5.3 输入框组件

```html
<div class="input-group custom-input-group">
    <span class="input-group-text bg-white border-end-0">
        <i class="bi bi-search"></i>
    </span>
    <input type="text" class="form-control border-start-0" 
           placeholder="搜索手语词汇...">
</div>
```

```css
/* ==================== 输入框样式 ==================== */
.custom-input-group .form-control {
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding-left: 0;
}

.custom-input-group .input-group-text {
    border-radius: var(--radius-md) 0 0 var(--radius-md);
    background: #fff;
}

.custom-input-group .form-control:focus {
    box-shadow: none;
    border-color: var(--color-accent);
}

.custom-input-group:focus-within {
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    border-radius: var(--radius-md);
}
```

### 5.4 徽章组件

```html
<span class="badge bg-primary-custom">AI 2.0</span>
<span class="badge bg-success-custom">99% 准度</span>
<span class="badge rounded-pill bg-light text-dark border">批量处理</span>
```

---

## 6. 交互规范

### 6.1 动画效果

```css
/* ==================== 全局动画 ==================== */

/* 淡入动画 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.4s ease-out forwards;
}

/* 悬浮效果 */
.hover-lift {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.hover-lift:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-hover);
}

/* 脉冲动画（用于识别中） */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* 骨架加载动画 */
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.skeleton {
    background: linear-gradient(90deg, 
        var(--color-gray-200) 25%, 
        var(--color-gray-100) 50%, 
        var(--color-gray-200) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}
```

### 6.2 交互反馈

```javascript
// Toast 提示
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} fade-in`;
    toast.innerHTML = `
        <i class="bi bi-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

// 加载状态
function showLoading(message = '加载中...') {
    Swal.fire({
        title: message,
        html: '<div class="spinner-border text-primary" role="status"></div>',
        showConfirmButton: false,
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });
}

function hideLoading() {
    Swal.close();
}

// 确认弹窗
async function showConfirm(title, text) {
    const result = await Swal.fire({
        title: title,
        text: text,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '确认',
        cancelButtonText: '取消',
        confirmButtonColor: '#1A1A1A',
        cancelButtonColor: '#6B7280'
    });
    return result.isConfirmed;
}
```

### 6.3 骨架屏

```html
<!-- 骨架加载占位 -->
<div class="skeleton-card">
    <div class="skeleton skeleton-thumbnail mb-2"></div>
    <div class="skeleton skeleton-text mb-1"></div>
    <div class="skeleton skeleton-text-small"></div>
</div>

<style>
.skeleton-card {
    background: #fff;
    border-radius: var(--radius-lg);
    padding: 12px;
}

.skeleton-thumbnail {
    aspect-ratio: 16/9;
    border-radius: var(--radius-md);
}

.skeleton-text {
    height: 16px;
    border-radius: 4px;
    width: 80%;
}

.skeleton-text-small {
    height: 12px;
    border-radius: 4px;
    width: 50%;
}
</style>
```

---

## 7. API 接口

### 7.1 接口配置

```javascript
// js/config.js
const API_CONFIG = {
    baseURL: 'http://localhost:8000/api/v1',
    timeout: 30000,
    retries: 3
};

// 请求拦截器
const requestInterceptor = (config) => {
    // 添加认证 token
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
};

// 响应拦截器
const responseInterceptor = (response) => {
    return response.data;
};
```

### 7.2 接口定义

| 接口 | 方法 | 路径 | 描述 |
|------|------|------|------|
| 实时识别 | WS | `/ws/recognize` | WebSocket 实时通信 |
| 上传翻译 | POST | `/recognize/upload` | 异步文件翻译 |
| 词典查询 | GET | `/dictionary?q=` | 搜索词汇 |
| 词汇详情 | GET | `/dictionary/{id}` | 获取词汇详情 |
| 翻译历史 | GET | `/user/history` | 获取用户历史记录 |
| 用户注册 | POST | `/auth/register` | 用户注册 |
| 用户登录 | POST | `/auth/login` | 用户登录 |
| 用户信息 | GET | `/user/profile` | 获取用户信息 |

### 7.3 请求示例

```javascript
// 实时识别 WebSocket
function connectRecognitionSocket() {
    const ws = new WebSocket(`${WS_URL}/recognize`);
    
    ws.onopen = () => {
        console.log('WebSocket 连接成功');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateRecognitionResult(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket 错误:', error);
    };
    
    return ws;
}

// 上传文件翻译
async function uploadForTranslation(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading('正在上传并翻译...');
        
        const response = await fetch(`${API_CONFIG.baseURL}/recognize/upload`, {
            method: 'POST',
            body: formData,
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const result = await response.json();
        hideLoading();
        showTranslationResult(result);
        
        return result;
    } catch (error) {
        hideLoading();
        showToast('翻译失败，请重试', 'error');
        throw error;
    }
}
```

---

## 8. 开发规范

### 8.1 代码风格

```javascript
// ==================== JavaScript 编码规范 ====================

// 1. 使用 const 和 let，避免使用 var
const MAX_FILE_SIZE = 100 * 1024 * 1024;
let isRecording = false;

// 2. 箭头函数
const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// 3. 使用模板字符串
const greeting = `你好，${userName}！`;

// 4. 解构赋值
const { userId, userName } = userInfo;
const [first, second, ...rest] = array;

// 5. async/await 替代回调
async function fetchData() {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('获取数据失败:', error);
    }
}
```

### 8.2 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 变量 | camelCase | `isRecording`, `fileList` |
| 常量 | UPPER_SNAKE_CASE | `MAX_FILE_SIZE`, `API_BASE_URL` |
| 函数 | camelCase（动词+名词） | `getUserInfo()`, `formatDate()` |
| CSS 类 | kebab-case | `.feature-card`, `.loading-spinner` |
| ID | camelCase | `#userProfile`, `#fileInput` |

### 8.3 文件命名

```
index.html              # 首页
live.html               # 实时翻译页
upload.html             # 上传翻译页
dictionary.html         # 词典页
orders.html             # 记录页
profile.html            # 个人中心
admin.html              # 管理后台

main.js                 # 主逻辑
camera.js               # 摄像头相关
upload.js               # 上传相关
dashboard.js            # 数据后台
data.js                 # 数据管理

style.css               # 主样式
animations.css          # 动画
```

### 8.4 注释规范

```javascript
/**
 * 功能描述
 * @param {Type} paramName - 参数描述
 * @returns {Type} 返回值描述
 */
function functionName(paramName) {
    // 单行注释
    
    /*
     * 多行注释
     * 第二行
     */
    
    // TODO: 待完成功能
    // FIXME: 需要修复的问题
    // HACK: 临时解决方案
}
```

### 8.5 Git 提交规范

```
feat: 新功能
fix: Bug 修复
docs: 文档更新
style: 代码格式（不影响功能）
refactor: 重构
perf: 性能优化
test: 测试相关
chore: 构建/工具相关

示例：
feat: 添加实时翻译页面骨架
fix: 修复摄像头镜像问题
docs: 更新 README
```

---

## 附录

### A. 常用图标

| 图标 | 类名 | 用途 |
|------|------|------|
| 📹 | `bi-camera-video` | 实时翻译 |
| 📤 | `bi-cloud-upload` | 上传 |
| 📖 | `bi-book` | 词典 |
| 🔊 | `bi-volume-up` | 语音播放 |
| 🔇 | `bi-volume-mute` | 静音 |
| 🔄 | `bi-arrow-left-right` | 镜像 |
| 💡 | `bi-lightning-charge` | 补光 |
| 📸 | `bi-camera` | 拍照 |
| ⏸️ | `bi-pause-circle` | 暂停 |
| ▶️ | `bi-play-circle` | 开始 |
| 📋 | `bi-clipboard` | 复制 |
| 📤 | `bi-share` | 分享 |
| 🏠 | `bi-house` | 首页 |

### B. 浏览器支持

| 浏览器 | 最低版本 | 说明 |
|--------|---------|------|
| Chrome | 90+ | 推荐 |
| Firefox | 88+ | 推荐 |
| Safari | 14+ | 部分功能 |
| Edge | 90+ | 推荐 |
| iOS Safari | 14+ | 部分功能 |
| Android Chrome | 90+ | 部分功能 |

### C. 性能优化

```javascript
// 1. 图片懒加载
<img loading="lazy" src="image.jpg" alt="">

// 2. 视频预加载
<video preload="metadata">
    <source src="video.mp4" type="video/mp4">
</video>

// 3. 事件节流
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// 4. 防抖
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}
```

---

## 变更日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v1.0 | 2026-02-03 | 初始版本，完成基础设计文档 |

---

> 文档维护者：前端开发团队  
> 如有疑问，请联系技术负责人


