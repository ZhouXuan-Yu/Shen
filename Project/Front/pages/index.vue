<template>
  <div class="pt-16 min-h-screen bg-slate-50 relative overflow-hidden">
    <!-- 背景光晕 -->
    <div class="pointer-events-none absolute inset-0 overflow-hidden">
      <div
        class="ambient-blob bg-teal-200/70 top-[-18rem] -left-32 w-[26rem] h-[26rem]"
      ></div>
      <div
        class="ambient-blob bg-cyan-200/80 top-1/3 -right-40 w-[30rem] h-[30rem]"
      ></div>
      <div
        class="ambient-blob bg-lime-200/70 bottom-[-14rem] left-1/4 w-[28rem] h-[28rem]"
      ></div>
    </div>

    <main class="relative z-10">
      <!-- Hero + 体验入口 -->
      <section class="hero-section">
        <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div
            class="grid lg:grid-cols-[1.05fr_minmax(0,0.95fr)] gap-10 lg:gap-14 items-center"
          >
            <!-- 左侧文案 -->
            <div class="space-y-8 fade-in-up">
              <div
                class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-teal-50/80 text-teal-700 text-xs sm:text-sm font-medium border border-teal-100/80 shadow-[0_8px_22px_rgba(15,118,110,0.06)]"
              >
                <span
                  class="inline-flex h-6 w-6 items-center justify-center rounded-full bg-teal-500/10 text-teal-700"
                >
                  <i class="bi bi-stars text-sm"></i>
                </span>
                HandTalk AI · 为听障群体打造的手语翻译助手
              </div>

              <div class="space-y-4">
                <h1
                  class="text-3.5xl sm:text-4xl md:text-5xl lg:text-[3.2rem] font-bold tracking-tight text-slate-900 leading-tight"
                >
                  让
                  <span
                    class="bg-gradient-to-r from-teal-500 via-cyan-500 to-lime-500 bg-clip-text text-transparent"
                  >
                    手语翻译
                  </span>
                  像呼吸一样自然
                </h1>
                <p
                  class="text-base sm:text-lg text-slate-600 max-w-xl leading-relaxed"
                >
                  一款为中国听障人士量身打造的
                  AI 手语翻译工具，支持实时识别、图片序列与视频翻译，
                  用科技让每一次沟通都被温柔接住。
                </p>
              </div>

              <!-- 主要操作入口 -->
              <div class="space-y-4">
                <div
                  class="flex flex-wrap items-center gap-3 sm:gap-4 fade-in-up delay-100"
                >
                  <NuxtLink
                    to="/recognize"
                    class="inline-flex items-center justify-center px-5 sm:px-6 py-3 rounded-full text-sm sm:text-base font-medium text-white bg-gradient-to-r from-teal-500 via-cyan-500 to-lime-500 shadow-[0_12px_30px_rgba(34,197,158,0.35)] hover:shadow-[0_16px_40px_rgba(34,197,158,0.4)] hover:brightness-105 active:scale-[0.98] transition-all"
                  >
                    <i class="bi bi-camera-video mr-2 text-sm sm:text-base"></i>
                                      
                  </NuxtLink>
                  <NuxtLink
                    to="/translate"
                    class="inline-flex items-center justify-center px-4 sm:px-5 py-3 rounded-full text-sm sm:text-base font-medium text-teal-900 bg-white/70 border border-white/80 shadow-[0_10px_28px_rgba(15,23,42,0.06)] hover:bg-white/90 hover:border-teal-200 hover:shadow-[0_16px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl transition-all"
                  >
                    <i class="bi bi-images mr-2 text-sm sm:text-base"></i>
                    图片序列翻译
                  </NuxtLink>
                  <NuxtLink
                    to="/video-translate"
                    class="inline-flex items-center justify-center px-4 sm:px-5 py-3 rounded-full text-sm sm:text-base font-medium text-teal-900/90 hover:text-teal-900 border border-teal-100 bg-teal-50/60 hover:bg-teal-50 shadow-[0_8px_24px_rgba(13,148,136,0.08)] hover:shadow-[0_14px_38px_rgba(13,148,136,0.16)] backdrop-blur-xl transition-all"
                  >
                    <i class="bi bi-film mr-2 text-sm sm:text-base"></i>
                    视频翻译
                  </NuxtLink>
                </div>
                <p class="text-xs sm:text-sm text-slate-500">
                  无需登录，直接体验 · 建议在自然光线环境下拍摄，效果更佳
                </p>
              </div>

              <!-- 小统计 -->
              <div
                class="flex flex-wrap items-center gap-6 text-xs sm:text-sm text-slate-500 fade-in-up delay-150"
              >
                <div class="flex items-center gap-2">
                  <span
                    class="inline-flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-600"
                  >
                    <i class="bi bi-activity"></i>
                  </span>
                  <span> 实时识别 · 毫秒级响应 </span>
                </div>
                <div class="flex items-center gap-2">
                  <span
                    class="inline-flex h-7 w-7 items-center justify-center rounded-full bg-cyan-500/10 text-cyan-600"
                  >
                    <i class="bi bi-shield-check"></i>
                  </span>
                  <span> 本地预览 · 云端计算更安全 </span>
                </div>
              </div>
            </div>

            <!-- 右侧“卡片组合”：上传 / 结果 / 建议（玻璃效果） -->
            <div class="relative fade-in-up delay-75">
              <div
                class="glass-card relative rounded-3xl p-5 sm:p-6 lg:p-7 space-y-5 shadow-[0_18px_60px_rgba(15,23,42,0.12)]"
              >
                <!-- 顶部：模拟上传区 -->
                <div class="space-y-3">
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                      <div
                        class="h-9 w-9 rounded-2xl bg-teal-500/10 flex items-center justify-center text-teal-600"
                      >
                        <i class="bi bi-cloud-arrow-up"></i>
                      </div>
                      <div>
                        <p class="text-xs uppercase tracking-wide text-slate-400">
                          Upload Area
                        </p>
                        <p class="text-sm font-medium text-slate-800">
                          上传手语图片 / 视频
                        </p>
                      </div>
                    </div>
                    <span
                      class="inline-flex items-center gap-1 rounded-full border border-emerald-100 bg-emerald-50/70 px-2.5 py-1 text-[11px] font-medium text-emerald-700"
                    >
                      <span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                      安全加密
                    </span>
                  </div>

                  <div
                    class="group mt-2 flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-teal-200/70 bg-white/60 px-4 py-6 text-center text-xs sm:text-sm text-slate-500 hover:bg-white/80 hover:border-teal-300 hover:shadow-[0_14px_40px_rgba(34,197,158,0.16)] transition-all backdrop-blur-2xl"
                  >
                    <i
                      class="bi bi-upload mb-2 text-lg sm:text-xl text-teal-500/80 group-hover:text-teal-600"
                    ></i>
                    <p class="font-medium text-slate-700 mb-1">
                      拖拽图片 / 视频到这里，或点击选择文件
                    </p>
                    <p class="text-[11px] sm:text-xs text-slate-400">
                      支持 MP4 / JPG / PNG 等常见格式，单次上传不超过 60 秒视频
                    </p>
                  </div>
                </div>

                <!-- 中部：模拟识别结果区 -->
                <div
                  class="rounded-2xl border border-white/80 bg-white/65 px-4 py-3.5 backdrop-blur-2xl shadow-[0_10px_30px_rgba(15,23,42,0.06)]"
                >
                  <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                      <span
                        class="inline-flex h-7 w-7 items-center justify-center rounded-full bg-cyan-500/10 text-cyan-600"
                      >
                        <i class="bi bi-translate"></i>
                      </span>
                      <p class="text-xs font-medium text-slate-500">
                        识别结果（示意）
                      </p>
                    </div>
                    <span class="text-xs text-emerald-600 font-semibold">
                      96% 置信度
                    </span>
                  </div>
                  <p
                    class="text-sm sm:text-base font-medium text-slate-900 leading-relaxed"
                  >
                    你好，我想预约下周三上午的挂号。
                  </p>
                  <p class="mt-1.5 text-[11px] sm:text-xs text-slate-500">
                    拼音：nǐ hǎo，wǒ xiǎng yùyuē xià zhōusān shàngwǔ de guàhào。
                  </p>
                </div>

                <!-- 底部：建议 / 使用提示 -->
                <div
                  class="rounded-2xl border border-white/80 bg-white/55 px-4 py-3.5 backdrop-blur-2xl flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div class="flex items-start gap-2.5">
                    <span
                      class="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-full bg-lime-500/10 text-lime-600"
                    >
                      <i class="bi bi-lightbulb"></i>
                    </span>
                    <div>
                      <p class="text-xs sm:text-sm font-medium text-slate-800">
                        建议：保持单色背景与稳定光线，识别更准确
                      </p>
                      <p class="mt-0.5 text-[11px] text-slate-500">
                        如果是对话场景，可使用
                        <span class="font-medium text-teal-600">实时翻译</span>
                        模式；长视频内容推荐使用
                        <span class="font-medium text-teal-600">视频翻译</span>
                        模式。
                      </p>
                    </div>
                  </div>
                  <NuxtLink
                    to="/about"
                    class="mt-1 inline-flex items-center justify-center rounded-full border border-teal-100 bg-teal-50/60 px-3.5 py-1.5 text-[11px] font-medium text-teal-800 hover:bg-teal-100/80 hover:border-teal-200 hover:text-teal-900 transition-all"
                  >
                    <i class="bi bi-info-circle mr-1.5"></i>
                    了解使用建议
                  </NuxtLink>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 适用场景 -->
      <section class="py-12 sm:py-16 lg:py-18">
        <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="text-center mb-10 fade-in-up">
            <h2
              class="text-2xl sm:text-3xl md:text-4xl font-bold text-slate-900 mb-3"
            >
              在真实场景中，让沟通变得更轻松
            </h2>
            <p class="text-sm sm:text-base text-slate-600 max-w-2xl mx-auto">
              医院问诊、银行办理业务、家庭日常沟通……HandTalk
              希望在你需要的时候，悄悄站在你身后。
            </p>
          </div>

          <div
            class="grid grid-cols-1 md:grid-cols-3 gap-5 sm:gap-6 fade-in-up delay-75"
          >
            <div class="scene-card glass-card p-6 sm:p-7">
              <div
                class="w-12 h-12 rounded-2xl bg-gradient-to-br from-teal-400 to-cyan-400 flex items-center justify-center mb-5 shadow-[0_12px_30px_rgba(34,197,194,0.35)]"
              >
                <i class="bi bi-hospital text-xl text-white"></i>
              </div>
              <h3 class="text-lg font-semibold text-slate-900 mb-2">
                医疗服务场景
              </h3>
              <p class="text-sm text-slate-600 leading-relaxed">
                帮助听障患者在问诊、检查、取药等环节准确表达需求，减轻医患双方的沟通压力。
              </p>
            </div>

            <div class="scene-card glass-card p-6 sm:p-7">
              <div
                class="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-lime-400 flex items-center justify-center mb-5 shadow-[0_12px_30px_rgba(52,211,153,0.35)]"
              >
                <i class="bi bi-bank text-xl text-white"></i>
              </div>
              <h3 class="text-lg font-semibold text-slate-900 mb-2">
                金融与公共服务
              </h3>
              <p class="text-sm text-slate-600 leading-relaxed">
                在银行、政务大厅、车站等窗口场景辅助沟通，提升服务效率，让办理流程更顺畅。
              </p>
            </div>

            <div class="scene-card glass-card p-6 sm:p-7">
              <div
                class="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-400 to-teal-400 flex items-center justify-center mb-5 shadow-[0_12px_30px_rgba(16,185,129,0.35)]"
              >
                <i class="bi bi-house-heart text-xl text-white"></i>
              </div>
              <h3 class="text-lg font-semibold text-slate-900 mb-2">
                家庭与校园沟通
              </h3>
              <p class="text-sm text-slate-600 leading-relaxed">
                轻松覆盖家庭日常、课堂交流等高频沟通场景，让理解与陪伴变得更加自然。
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
// SEO 元信息
useSeoMeta({
  title: '译手 HandTalk AI - 让沟通无障碍',
  description:
    '面向中国市场的 AI 手语翻译应用，用科技传递人文关怀，消除听障人士与健听人之间的沟通障碍。',
  ogTitle: '译手 HandTalk AI - 让沟通无障碍',
  ogDescription: '面向中国市场的 AI 手语翻译应用',
})
</script>

<style scoped>
.hero-section {
  min-height: calc(80vh - 4rem);
  display: flex;
  align-items: center;
}

.glass-card {
  position: relative;
  border-radius: 1.5rem;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(255, 255, 255, 0.85);
  box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
  backdrop-filter: blur(26px);
  -webkit-backdrop-filter: blur(26px);
  transition: background 0.25s ease, border-color 0.25s ease,
    box-shadow 0.25s ease, transform 0.2s ease;
}

.glass-card:hover {
  background: rgba(255, 255, 255, 0.9);
  border-color: rgba(45, 212, 191, 0.6);
  box-shadow: 0 18px 60px rgba(34, 197, 158, 0.25);
  transform: translateY(-2px);
}

.ambient-blob {
  position: relative;
  border-radius: 999px;
  filter: blur(80px);
  opacity: 0.7;
}

.ambient-blob::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: radial-gradient(
    circle at 20% 20%,
    rgba(255, 255, 255, 0.8),
    transparent 60%
  );
  mix-blend-mode: screen;
}

.scene-card {
  position: relative;
  overflow: hidden;
  border-radius: 1.5rem;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  transition: transform 0.2s ease, box-shadow 0.25s ease,
    background 0.25s ease, border-color 0.25s ease;
}

.scene-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(
    circle at 0% 0%,
    rgba(45, 212, 191, 0.1),
    transparent 55%
  );
  opacity: 0;
  transition: opacity 0.25s ease;
  pointer-events: none;
}

.scene-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 18px 55px rgba(15, 23, 42, 0.14);
  background: rgba(255, 255, 255, 0.95);
  border-color: rgba(45, 212, 191, 0.6);
}

.scene-card:hover::before {
  opacity: 1;
}

/* 初次加载浮现动画 */
.fade-in-up {
  opacity: 0;
  transform: translateY(10px);
  animation: fadeInUp 0.7s cubic-bezier(0.19, 1, 0.22, 1) forwards;
}

.fade-in-up.delay-75 {
  animation-delay: 0.15s;
}

.fade-in-up.delay-100 {
  animation-delay: 0.2s;
}

.fade-in-up.delay-150 {
  animation-delay: 0.3s;
}

@keyframes fadeInUp {
  0% {
    opacity: 0;
    transform: translateY(14px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .hero-section {
    min-height: auto;
    padding-top: 1.75rem;
    padding-bottom: 1.75rem;
  }
}
</style>

