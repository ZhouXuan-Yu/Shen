// ==================== 通用类型 ====================

/** API 响应基础结构 */
export interface ApiResponse<T> {
  code: number
  message: string
  data: T
}

/** 分页参数 */
export interface PaginationParams {
  page: number
  limit: number
}

/** 分页响应 */
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  limit: number
  totalPages: number
}

// ==================== 用户相关类型 ====================

/** 用户信息 */
export interface User {
  id: string
  username: string
  email: string
  phone: string
  avatar?: string
  createdAt: string
  updatedAt: string
}

/** 用户注册请求 */
export interface RegisterRequest {
  username: string
  email: string
  phone: string
  password: string
  captcha: string
}

/** 用户登录请求 */
export interface LoginRequest {
  email: string
  password: string
}

/** 登录响应 */
export interface LoginResponse {
  token: string
  user: User
}

// ==================== 识别相关类型 ====================

/** 实时识别结果 */
export interface RecognitionResult {
  text: string
  pinyin: string
  meaning: string
  confidence: number
  timestamp: string
}

/** 骨架关键点 */
export interface SkeletonPoints {
  hand: number[][] // 手部关键点 [21, 3]
  face: number[][] // 面部关键点 [468, 3]
}

/** 识别帧数据 */
export interface RecognitionFrame {
  image: string // base64 编码的图片
  timestamp: number
}

/** 文件上传识别请求 */
export interface UploadRecognitionRequest {
  file: File
  options?: {
    topK?: number
  }
}

/** 文件上传识别响应 */
export interface UploadRecognitionResponse {
  id: string
  results: RecognitionResult[]
  videoDuration?: number
  processedFrames: number
  createdAt: string
}

// ==================== 词典相关类型 ====================

/** 词汇条目 */
export interface Word {
  id: string
  chinese: string
  pinyin: string
  meaning: string
  category: string
  videoUrl?: string
  thumbnailUrl?: string
  example?: string
  gesturePoints?: string[]
  relatedWords?: string[]
  score?: number
  searchType?: string
  createdAt?: string
  updatedAt?: string
}

/** 词汇类别 */
export interface WordCategory {
  id: string
  name: string
  icon: string
  wordCount: number
}

// ==================== 历史记录类型 ====================

/** 翻译历史记录 */
export interface HistoryRecord {
  id: string
  type: 'realtime' | 'upload_image' | 'upload_video' | 'image_sequence'
  /**
   * 缩略图地址
   * - 对于上传图片：使用 base64 data URL，避免 blob URL 失效
   * - 对于视频/实时：可以为空，由前端根据类型展示占位图标
   */
  thumbnail?: string
  result: string
  confidence: number
  duration?: number
  createdAt: string
  /** 是否被用户标记为收藏 */
  favorite?: boolean
}

// ==================== 管理后台类型 ====================

/** 统计数据 */
export interface AdminStats {
  totalTranslations: number
  todayTranslations: number
  totalUsers: number
  accuracy: number
}

/** 用户管理 */
export interface AdminUser extends User {
  status: 'active' | 'banned'
  lastLoginAt: string
}

// ==================== WebSocket 类型 ====================

/** WebSocket 消息 */
export interface WSMessage {
  type: 'frame' | 'result' | 'error' | 'ping'
  data: unknown
  timestamp: number
}

/** 实时识别 WebSocket 消息 */
export interface WSRecognitionMessage extends WSMessage {
  type: 'result'
  data: {
    text: string
    pinyin: string
    meaning: string
    confidence: number
  }
}


