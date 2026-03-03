一、整体方案设计（覆盖你提的 5 个问题并做合理扩展）
登录 / 注册 UI 与配色规范统一（已开始实现）
左侧大背景统一为「蓝绿色光谱」：teal + sky + lime 渐变，去掉所有紫色、深蓝和暗黑风格。
登录/注册页面内所有主标题、说明文字统一用 text-teal-900 / text-slate-600 等柔和深青灰，避免纯黑。
按钮和高亮区采用 from-teal-400 via-sky-400 to-lime-400 渐变；去掉微信 / GitHub 登录组件，仅保留账号密码登录/注册。
后续可按同一配色体系，逐步替换 TheHeader、首页、Profile 等页面中的旧紫色渐变。
注册-登录逻辑修复与本地账号模型梳理（已修复核心问题）
注册时：将 username + email + password 持久化到 localStorage('auth_user')，并引导用户去登录。
登录时：从本地读取 auth_user，对比邮箱和密码，成功后只更新运行时 user + 生成 token，不再覆盖存储的密码，保证退出后仍然可以用同一账号再次登录。
登出时：仅清除 token 和运行时 user，不清理 auth_user，相当于“退出登录但保留账号”。
登录态校验与功能路由保护
新增 auth 中间件：对需要登录的页面（/translate、/video-translate、/recognize、/profile/*、未来的 /profile/favorites）做统一鉴权，未登录时跳转到 /auth/login?redirect=xxx 并提示。
TheHeader 中导航点击逻辑增强：如果未登录，点击这些功能入口会先跳转登录（或弹 Toast “请先登录后使用翻译功能”），避免“未登录也能进入功能页”的错觉。
后续可以在登录成功后自动跳转回 redirect 指定的功能页，提升体验。
个人翻译历史页图片显示问题修复
当前历史记录把 thumbnail 保存为 blob: 预览 URL，在刷新后失效，导致你截图中看到的大白块和损坏图标。
改造策略：
在图片翻译接口响应中，优先使用后端返回的可访问图片地址（如 imageUrl）作为 thumbnail 持久化；
如果没有后端地址，则不存缩略图（历史列表用图标占位），避免无效 blob: 链接；
历史页渲染时增加保护：如果 record.thumbnail 以 blob: 开头且加载失败，则自动降级为图标占位，并给出“缩略图已过期，可重新识别”提示。
“我的收藏”功能重构与前后打通
新增 favorites Store：
结构类似 HistoryRecord，增加 favoriteType: 'image' | 'video'、sourceId（对应历史记录 ID）、thumbnail、text、createdAt 等字段。
支持本地持久化（localStorage('favorites')）、去重、取消收藏。
在 图片翻译页 和 视频翻译页：
在识别结果区域增加“收藏”按钮（图标 + 文案），点击后将当前结果写入 favorites，并关联对应历史记录 ID。
如果已收藏，按钮切换为“已收藏 / 取消收藏”。
新建 /profile/favorites.vue 页面（真正的“我的收藏”页）：
卡片网格展示收藏的翻译（缩略图 + 文本 + 置信度 + 收藏时间）；
支持按类型筛选（全部 / 图片 / 视频），以及清空收藏。
Profile 首页中的“我的收藏”入口改为跳到 /profile/favorites，不再误指向词典页。
在此基础上的拓展与打磨
历史和收藏页：根据真实数据动态计算统计卡片（总次数、今日次数、平均准确率），替换掉现在写死的 128 / 5 / 94%。
统一空状态与错误状态：例如“暂无收藏”“暂无历史记录”“未登录，请先登录查看个人中心”等，并与整体蓝绿系风格匹配。
在登录/注册/收藏等关键操作中补充更友好的 Toast 文案和状态反馈（如“收藏成功，已添加到我的收藏”）。
二、已完成的步骤（第 1、2 步）
登录 / 注册页配色与第三方登录移除
login.vue 左侧背景改为蓝绿色渐变：from-teal-300 via-sky-300 to-lime-300，文案颜色统一为 text-teal-900 / text-slate-600，按钮改为 from-teal-400 via-sky-400 to-lime-400 渐变。
register.vue 左侧同样使用蓝绿色光谱背景和柔和装饰圆；表单标题与 label 全部使用 text-teal-900 / text-slate-700。
登录页和注册页都删除了微信 / GitHub 登录按钮区域，只保留中间的分割线文字（“或者”），未来如果接入第三方可以再挂回。
修复“注册后退出再登录提示未注册”的逻辑漏洞
在 stores/auth.ts 中：
setUser 现在只在有 password 字段时才写入本地缓存，不会在登录后把密码覆盖成 undefined；
login 成功后只设置 token，并用本地缓存的账号信息挂到 user，不再通过 setUser 去改写 auth_user。
这样流程变为：
注册一次 → 本地缓存包含邮箱 + 密码；
登录多次 / 退出再登录 → 始终基于同一份缓存校验，不会出现“注册信息被覆盖，提示未注册”的情况。
三、下一步我将如何分步骤继续实现
接下来我会按下面的顺序逐步实现，过程中每一步都会在保持你当前功能可用的前提下修改：
步骤 3：实现路由鉴权与 Header 导航拦截
新增 middleware/auth.ts，并在需要登录的页面 (translate.vue、video-translate.vue、recognize.vue、pages/profile/*) 加上 definePageMeta({ middleware: 'auth' })。
在 TheHeader.vue 中，对“图片翻译 / 视频翻译 / 实时翻译 / 个人中心 / 翻译历史”等入口增加登录检查，未登录时跳转 /auth/login 并带上 redirect 参数。
步骤 4：修复个人翻译历史中的图片缩略图问题
调整 translate.vue 中写入 recognitionStore.addToHistory 的逻辑，改从接口响应中获取稳定 URL，避免存 blob:。
在 profile/history.vue 中根据 thumbnail 是否可用选择显示图片或图标占位，并给出适当提示。
步骤 5：实现图片 / 视频翻译的“收藏”功能和“我的收藏”页面
新建 stores/favorites.ts，实现添加、删除、查询、持久化。
在 translate.vue、video-translate.vue 识别结果区域增加“收藏 / 取消收藏”按钮，并将数据写入 favorites。
新建 pages/profile/favorites.vue，并把个人中心里的“我的收藏”入口指向这个页面。
步骤 6：数据统计与交互细节拓展
用真实历史 / 收藏数据计算统计卡片，替换写死数值。
补充统一风格的空状态、错误提示和 loading 状态。