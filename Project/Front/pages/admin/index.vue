<template>
  <div class="min-h-screen bg-gray-50 flex">
    <!-- 侧边栏 -->
    <aside class="fixed left-0 top-0 bottom-0 w-60 bg-primary-900 text-white">
      <div class="p-4 border-b border-white/10">
        <div class="flex items-center gap-2">
          <div class="w-10 h-10 rounded-xl bg-accent-gradient flex items-center justify-center">
            <span class="text-white text-lg">🤟</span>
          </div>
          <div>
            <h1 class="font-bold">HandTalk</h1>
            <small class="text-white/50">Admin Panel</small>
          </div>
        </div>
      </div>
      
      <nav class="p-4">
        <ul class="space-y-1">
          <li v-for="item in menuItems" :key="item.path">
            <NuxtLink :to="item.path" 
              class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors"
              :class="$route.path === item.path ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5 hover:text-white'"
            >
              <i :class="item.icon"></i>
              <span>{{ item.label }}</span>
            </NuxtLink>
          </li>
        </ul>
      </nav>
    </aside>

    <!-- 主内容区 -->
    <main class="flex-1 ml-60">
      <!-- 顶部栏 -->
      <header class="bg-white border-b border-gray-200 h-16 flex items-center justify-between px-6">
        <h2 class="font-semibold text-primary-900">{{ currentPageTitle }}</h2>
        <div class="flex items-center gap-4">
          <span class="text-sm text-gray-500">欢迎回来，管理员</span>
          <div class="w-8 h-8 rounded-full bg-accent flex items-center justify-center text-white text-sm">
            A
          </div>
        </div>
      </header>

      <!-- 页面内容 -->
      <div class="p-6">
        <!-- 统计卡片 -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <div v-for="stat in stats" :key="stat.label" class="card p-6">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 mb-1">{{ stat.label }}</p>
                <p class="text-3xl font-bold text-primary-900">{{ stat.value }}</p>
              </div>
              <div class="w-12 h-12 rounded-xl flex items-center justify-center" :class="stat.bgClass">
                <i :class="[stat.icon, stat.iconClass]"></i>
              </div>
            </div>
          </div>
        </div>

        <!-- 图表区域 -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <!-- 流量趋势图 -->
          <div class="card p-6">
            <h3 class="font-semibold text-primary-900 mb-4">实时翻译流量</h3>
            <div id="trafficChart" class="h-72"></div>
          </div>
          <!-- 类别分布图 -->
          <div class="card p-6">
            <h3 class="font-semibold text-primary-900 mb-4">手语类别分布</h3>
            <div id="categoryChart" class="h-72"></div>
          </div>
        </div>

        <!-- 最新记录表格 -->
        <div class="card">
          <div class="card-header bg-white px-6 py-4 border-b border-gray-100">
            <h3 class="font-semibold text-primary-900">最新翻译记录</h3>
          </div>
          <div class="p-0">
            <div class="overflow-x-auto">
              <table class="w-full">
                <thead class="bg-gray-50">
                  <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">用户</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">类型</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">结果</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">置信度</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">时间</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">操作</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-gray-100">
                  <tr v-for="record in recentRecords" :key="record.id" class="hover:bg-gray-50">
                    <td class="px-6 py-4">
                      <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-sm">
                          {{ record.user.charAt(0) }}
                        </div>
                        <span class="text-sm text-primary-900">{{ record.user }}</span>
                      </div>
                    </td>
                    <td class="px-6 py-4">
                      <span class="badge" :class="record.type === '实时' ? 'bg-blue-100 text-blue-700' : 'bg-green-100 text-green-700'">
                        {{ record.type }}
                      </span>
                    </td>
                    <td class="px-6 py-4 text-sm text-primary-900">{{ record.result }}</td>
                    <td class="px-6 py-4">
                      <span class="text-sm" :class="getConfidenceClass(record.confidenceNum)">
                        {{ record.confidence }}
                      </span>
                    </td>
                    <td class="px-6 py-4 text-sm text-gray-500">{{ record.time }}</td>
                    <td class="px-6 py-4">
                      <button class="text-gray-400 hover:text-primary-900"><i class="bi bi-three-dots"></i></button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
interface RecordItem {
  id: number
  user: string
  type: string
  result: string
  confidence: string
  confidenceNum: number
  time: string
}

definePageMeta({ layout: false })

useSeoMeta({ title: '管理后台 - 译手 HandTalk AI' })

const route = useRoute()

const menuItems = [
  { path: '/admin', label: '仪表盘', icon: 'bi bi-grid-1x2' },
  { path: '/admin/users', label: '用户管理', icon: 'bi bi-people' },
  { path: '/admin/dictionary', label: '词条管理', icon: 'bi bi-book' },
  { path: '/admin/analytics', label: '数据分析', icon: 'bi bi-bar-chart-line' },
  { path: '/admin/settings', label: '系统设置', icon: 'bi bi-gear' },
]

const currentPageTitle = computed(() => {
  const item = menuItems.find(m => m.path === route.path)
  return item?.label || '仪表盘'
})

const stats = [
  { label: '翻译总量', value: '100,234', icon: 'bi bi-translate', iconClass: 'text-xl text-blue-600', bgClass: 'bg-blue-100' },
  { label: '准确率', value: '98.5%', icon: 'bi bi-patch-check', iconClass: 'text-xl text-green-600', bgClass: 'bg-green-100' },
  { label: '用户数', value: '5,678', icon: 'bi bi-people', iconClass: 'text-xl text-purple-600', bgClass: 'bg-purple-100' },
  { label: '今日量', value: '1,234', icon: 'bi bi-lightning', iconClass: 'text-xl text-orange-600', bgClass: 'bg-orange-100' },
]

const recentRecords: RecordItem[] = [
  { id: 1, user: '张*明', type: '实时', result: '你好', confidence: '96%', confidenceNum: 96, time: '14:30:25' },
  { id: 2, user: '李*红', type: '图片', result: '谢谢', confidence: '92%', confidenceNum: 92, time: '14:28:10' },
  { id: 3, user: '王*刚', type: '视频', result: '再见', confidence: '89%', confidenceNum: 89, time: '14:25:33' },
  { id: 4, user: '赵*丽', type: '实时', result: '对不起', confidence: '94%', confidenceNum: 94, time: '14:20:18' },
  { id: 5, user: '刘*洋', type: '图片', result: '没关系', confidence: '91%', confidenceNum: 91, time: '14:15:45' },
]

function getConfidenceClass(confidence: number): string {
  return confidence >= 90 ? 'text-success' : 'text-warning'
}

onMounted(async () => {
  const echarts = await import('echarts')
  
  const trafficChartEl = document.getElementById('trafficChart')
  if (trafficChartEl) {
    const trafficChart = echarts.init(trafficChartEl)
    trafficChart.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'] },
      yAxis: { type: 'value' },
      series: [{ data: [820, 932, 901, 934, 1290, 1330, 1320], type: 'line', smooth: true, areaStyle: { opacity: 0.3 }, itemStyle: { color: '#6366F1' } }]
    })
  }

  const categoryChartEl = document.getElementById('categoryChart')
  if (categoryChartEl) {
    const categoryChart = echarts.init(categoryChartEl)
    categoryChart.setOption({
      tooltip: { trigger: 'item' },
      series: [{
        type: 'pie', radius: ['40%', '70%'],
        data: [
          { value: 1048, name: '日常问候' },
          { value: 735, name: '数字' },
          { value: 580, name: '家庭' },
          { value: 484, name: '工作' },
          { value: 300, name: '情绪' }
        ],
        emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
      }]
    })
  }
})
</script>
