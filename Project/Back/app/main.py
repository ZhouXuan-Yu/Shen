"""
FastAPI 主应用模块
手语词典本地向量检索服务（纯内存实现）
"""
import time
import base64
from io import BytesIO
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

app = FastAPI(title="HandTalk AI - Sign Language Dictionary", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 内存数据存储 ====================
# 手语词典数据（模拟数据库）
DICTIONARY_DATA = [
    {"id": "1", "chinese": "你好", "pinyin": "nǐ hǎo", "meaning": "表示问候或打招呼", "category": "greeting", "video_url": "", "thumbnail_url": "", "example": "见面时说"},
    {"id": "2", "chinese": "谢谢", "pinyin": "xiè xiè", "meaning": "表示感激", "category": "greeting", "video_url": "", "thumbnail_url": "", "example": "别人帮助后说"},
    {"id": "3", "chinese": "再见", "pinyin": "zài jiàn", "meaning": "表示告别", "category": "greeting", "video_url": "", "thumbnail_url": "", "example": "分开时说"},
    {"id": "4", "chinese": "对不起", "pinyin": "duì bù qǐ", "meaning": "表示歉意", "category": "politeness", "video_url": "", "thumbnail_url": "", "example": "做错事后说"},
    {"id": "5", "chinese": "没关系", "pinyin": "méi guān xi", "meaning": "表示谅解", "category": "politeness", "video_url": "", "thumbnail_url": "", "example": "别人道歉时说"},
    {"id": "6", "chinese": "请", "pinyin": "qǐng", "meaning": "表示礼貌的请求", "category": "politeness", "video_url": "", "thumbnail_url": "", "example": "请求帮助时说"},
    {"id": "7", "chinese": "妈妈", "pinyin": "mā ma", "meaning": "母亲", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼母亲"},
    {"id": "8", "chinese": "爸爸", "pinyin": "bà ba", "meaning": "父亲", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼父亲"},
    {"id": "9", "chinese": "爷爷", "pinyin": "yé ye", "meaning": "祖父", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼祖父"},
    {"id": "10", "chinese": "奶奶", "pinyin": "nǎi nai", "meaning": "祖母", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼祖母"},
    {"id": "11", "chinese": "哥哥", "pinyin": "gē ge", "meaning": "兄长", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼哥哥"},
    {"id": "12", "chinese": "姐姐", "pinyin": "jiě jie", "meaning": "姐姐", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼姐姐"},
    {"id": "13", "chinese": "弟弟", "pinyin": "dì di", "meaning": "弟弟", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼弟弟"},
    {"id": "14", "chinese": "妹妹", "pinyin": "mèi mei", "meaning": "妹妹", "category": "family", "video_url": "", "thumbnail_url": "", "example": "称呼妹妹"},
    {"id": "15", "chinese": "朋友", "pinyin": "péng yǒu", "meaning": "朋友", "category": "social", "video_url": "", "thumbnail_url": "", "example": "称呼朋友"},
    {"id": "16", "chinese": "老师", "pinyin": "lǎo shī", "meaning": "教师", "category": "social", "video_url": "", "thumbnail_url": "", "example": "称呼老师"},
    {"id": "17", "chinese": "学生", "pinyin": "xué shēng", "meaning": "学生", "category": "social", "video_url": "", "thumbnail_url": "", "example": "学生自称"},
    {"id": "18", "chinese": "医生", "pinyin": "yī shēng", "meaning": "医生", "category": "profession", "video_url": "", "thumbnail_url": "", "example": "称呼医生"},
    {"id": "19", "chinese": "护士", "pinyin": "hù shi", "meaning": "护士", "category": "profession", "video_url": "", "thumbnail_url": "", "example": "称呼护士"},
    {"id": "20", "chinese": "警察", "pinyin": "jǐng chá", "meaning": "警察", "category": "profession", "video_url": "", "thumbnail_url": "", "example": "称呼警察"},
    {"id": "21", "chinese": "吃饭", "pinyin": "chī fàn", "meaning": "进食", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示吃饭"},
    {"id": "22", "chinese": "喝水", "pinyin": "hē shuǐ", "meaning": "喝水", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示喝水"},
    {"id": "23", "chinese": "睡觉", "pinyin": "shuì jiào", "meaning": "睡眠", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示睡觉"},
    {"id": "24", "chinese": "洗澡", "pinyin": "xǐ zǎo", "meaning": "清洁身体", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示洗澡"},
    {"id": "25", "chinese": "穿衣", "pinyin": "chuān yī", "meaning": "穿衣服", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示穿衣"},
    {"id": "26", "chinese": "工作", "pinyin": "gōng zuò", "meaning": "工作", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示工作"},
    {"id": "27", "chinese": "学习", "pinyin": "xué xí", "meaning": "学习", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示学习"},
    {"id": "28", "chinese": "看书", "pinyin": "kàn shū", "meaning": "阅读", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示看书"},
    {"id": "29", "chinese": "写字", "pinyin": "xiě zì", "meaning": "书写", "category": "daily", "video_url": "", "thumbnail_url": "", "example": "表示写字"},
    {"id": "30", "chinese": "打电话", "pinyin": "dǎ diàn huà", "meaning": "通话", "category": "communication", "video_url": "", "thumbnail_url": "", "example": "表示打电话"},
    {"id": "31", "chinese": "听", "pinyin": "tīng", "meaning": "用耳朵接收声音", "category": "senses", "video_url": "", "thumbnail_url": "", "example": "表示听"},
    {"id": "32", "chinese": "说", "pinyin": "shuō", "meaning": "用嘴说话", "category": "senses", "video_url": "", "thumbnail_url": "", "example": "表示说"},
    {"id": "33", "chinese": "看", "pinyin": "kàn", "meaning": "用眼睛观察", "category": "senses", "video_url": "", "thumbnail_url": "", "example": "表示看"},
    {"id": "34", "chinese": "吃", "pinyin": "chī", "meaning": "咀嚼食物", "category": "senses", "video_url": "", "thumbnail_url": "", "example": "表示吃"},
    {"id": "35", "chinese": "闻", "pinyin": "wén", "meaning": "用鼻子嗅", "category": "senses", "video_url": "", "thumbnail_url": "", "example": "表示闻"},
    {"id": "36", "chinese": "高兴", "pinyin": "gāo xìng", "meaning": "愉快的情绪", "category": "emotion", "video_url": "", "thumbnail_url": "", "example": "表示开心"},
    {"id": "37", "chinese": "难过", "pinyin": "nán guò", "meaning": "悲伤的情绪", "category": "emotion", "video_url": "", "thumbnail_url": "", "example": "表示伤心"},
    {"id": "38", "chinese": "生气", "pinyin": "shēng qì", "meaning": "愤怒的情绪", "category": "emotion", "video_url": "", "thumbnail_url": "", "example": "表示生气"},
    {"id": "39", "chinese": "害怕", "pinyin": "hài pà", "meaning": "恐惧的情绪", "category": "emotion", "video_url": "", "thumbnail_url": "", "example": "表示害怕"},
    {"id": "40", "chinese": "喜欢", "pinyin": "xǐ huān", "meaning": "喜爱的感情", "category": "emotion", "video_url": "", "thumbnail_url": "", "example": "表示喜欢"},
    {"id": "41", "chinese": "爱", "pinyin": "ài", "meaning": "爱的感情", "category": "emotion", "video_url": "", "thumbnail_url": "", "example": "表示爱"},
    {"id": "42", "chinese": "一", "pinyin": "yī", "meaning": "数字1", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字1"},
    {"id": "43", "chinese": "二", "pinyin": "èr", "meaning": "数字2", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字2"},
    {"id": "44", "chinese": "三", "pinyin": "sān", "meaning": "数字3", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字3"},
    {"id": "45", "chinese": "四", "pinyin": "sì", "meaning": "数字4", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字4"},
    {"id": "46", "chinese": "五", "pinyin": "wǔ", "meaning": "数字5", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字5"},
    {"id": "47", "chinese": "六", "pinyin": "liù", "meaning": "数字6", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字6"},
    {"id": "48", "chinese": "七", "pinyin": "qī", "meaning": "数字7", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字7"},
    {"id": "49", "chinese": "八", "pinyin": "bā", "meaning": "数字8", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字8"},
    {"id": "50", "chinese": "九", "pinyin": "jiǔ", "meaning": "数字9", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字9"},
    {"id": "51", "chinese": "十", "pinyin": "shí", "meaning": "数字10", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字10"},
    {"id": "52", "chinese": "零", "pinyin": "líng", "meaning": "数字0", "category": "number", "video_url": "", "thumbnail_url": "", "example": "数字0"},
    {"id": "53", "chinese": "大", "pinyin": "dà", "meaning": "体积或面积大", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示大"},
    {"id": "54", "chinese": "小", "pinyin": "xiǎo", "meaning": "体积或面积小", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示小"},
    {"id": "55", "chinese": "多", "pinyin": "duō", "meaning": "数量多", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示多"},
    {"id": "56", "chinese": "少", "pinyin": "shǎo", "meaning": "数量少", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示少"},
    {"id": "57", "chinese": "长", "pinyin": "cháng", "meaning": "长度长", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示长"},
    {"id": "58", "chinese": "短", "pinyin": "duǎn", "meaning": "长度短", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示短"},
    {"id": "59", "chinese": "高", "pinyin": "gāo", "meaning": "高度高", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示高"},
    {"id": "60", "chinese": "矮", "pinyin": "ǎi", "meaning": "高度低", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示矮"},
    {"id": "61", "chinese": "好", "pinyin": "hǎo", "meaning": "质量好", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示好"},
    {"id": "62", "chinese": "坏", "pinyin": "huài", "meaning": "质量差", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示坏"},
    {"id": "63", "chinese": "快", "pinyin": "kuài", "meaning": "速度快", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示快"},
    {"id": "64", "chinese": "慢", "pinyin": "màn", "meaning": "速度慢", "category": "adjective", "video_url": "", "thumbnail_url": "", "example": "表示慢"},
    {"id": "65", "chinese": "来", "pinyin": "lái", "meaning": "来到某处", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示来"},
    {"id": "66", "chinese": "去", "pinyin": "qù", "meaning": "前往某处", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示去"},
    {"id": "67", "chinese": "走", "pinyin": "zǒu", "meaning": "步行", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示走"},
    {"id": "68", "chinese": "跑", "pinyin": "pǎo", "meaning": "快速移动", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示跑"},
    {"id": "69", "chinese": "跳", "pinyin": "tiào", "meaning": "跳跃动作", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示跳"},
    {"id": "70", "chinese": "坐", "pinyin": "zuò", "meaning": "坐下", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示坐"},
    {"id": "71", "chinese": "站", "pinyin": "zhàn", "meaning": "站立", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示站"},
    {"id": "72", "chinese": "躺", "pinyin": "tǎng", "meaning": "躺下", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示躺"},
    {"id": "73", "chinese": "拿", "pinyin": "ná", "meaning": "用手拿取", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示拿"},
    {"id": "74", "chinese": "放", "pinyin": "fàng", "meaning": "放下", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示放"},
    {"id": "75", "chinese": "买", "pinyin": "mǎi", "meaning": "购买", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示买"},
    {"id": "76", "chinese": "卖", "pinyin": "mài", "meaning": "出售", "category": "action", "video_url": "", "thumbnail_url": "", "example": "表示卖"},
    {"id": "77", "chinese": "钱", "pinyin": "qián", "meaning": "货币", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示钱"},
    {"id": "78", "chinese": "书", "pinyin": "shū", "meaning": "书籍", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示书"},
    {"id": "79", "chinese": "笔", "pinyin": "bǐ", "meaning": "笔", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示笔"},
    {"id": "80", "chinese": "桌子", "pinyin": "zhuō zi", "meaning": "家具", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示桌子"},
    {"id": "81", "chinese": "椅子", "pinyin": "yǐ zi", "meaning": "家具", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示椅子"},
    {"id": "82", "chinese": "门", "pinyin": "mén", "meaning": "出入口", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示门"},
    {"id": "83", "chinese": "窗", "pinyin": "chuāng", "meaning": "窗户", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示窗"},
    {"id": "84", "chinese": "房子", "pinyin": "fáng zi", "meaning": "建筑物", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示房子"},
    {"id": "85", "chinese": "车", "pinyin": "chē", "meaning": "交通工具", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示车"},
    {"id": "86", "chinese": "飞机", "pinyin": "fēi jī", "meaning": "航空器", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示飞机"},
    {"id": "87", "chinese": "船", "pinyin": "chuán", "meaning": "水上交通工具", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示船"},
    {"id": "88", "chinese": "电脑", "pinyin": "diàn nǎo", "meaning": "计算机", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示电脑"},
    {"id": "89", "chinese": "手机", "pinyin": "shǒu jī", "meaning": "移动电话", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示手机"},
    {"id": "90", "chinese": "电视", "pinyin": "diàn shì", "meaning": "电视机", "category": "object", "video_url": "", "thumbnail_url": "", "example": "表示电视"},
    {"id": "91", "chinese": "今天", "pinyin": "jīn tiān", "meaning": "今日", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示今天"},
    {"id": "92", "chinese": "明天", "pinyin": "míng tiān", "meaning": "明日", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示明天"},
    {"id": "93", "chinese": "昨天", "pinyin": "zuó tiān", "meaning": "昨日", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示昨天"},
    {"id": "94", "chinese": "时间", "pinyin": "shí jiān", "meaning": "时间概念", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示时间"},
    {"id": "95", "chinese": "年", "pinyin": "nián", "meaning": "年份", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示年"},
    {"id": "96", "chinese": "月", "pinyin": "yuè", "meaning": "月份", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示月"},
    {"id": "97", "chinese": "日", "pinyin": "rì", "meaning": "日期", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示日"},
    {"id": "98", "chinese": "星期", "pinyin": "xīng qī", "meaning": "星期", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示星期"},
    {"id": "99", "chinese": "小时", "pinyin": "xiǎo shí", "meaning": "小时", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示小时"},
    {"id": "100", "chinese": "分钟", "pinyin": "fēn zhōng", "meaning": "分钟", "category": "time", "video_url": "", "thumbnail_url": "", "example": "表示分钟"},
]

# 中文同义词词典
SYNONYM_DICT = {
    "你好": ["您好", "嗨", "喂"],
    "谢谢": ["感谢", "多谢", "感恩"],
    "再见": ["拜拜", "再会", "回头见"],
    "爸爸": ["父亲", "爹", "爸"],
    "妈妈": ["母亲", "妈", "娘"],
    "哥哥": ["兄长", "哥"],
    "姐姐": ["姐"],
    "弟弟": ["弟"],
    "妹妹": ["妹"],
    "朋友": ["好友", "伙伴"],
    "老师": ["教师", "师长"],
    "学生": ["学子"],
    "医生": ["医师", "大夫"],
    "护士": ["护理人员"],
    "警察": ["民警", "公安"],
    "吃饭": ["用餐", "进食"],
    "喝水": ["饮水"],
    "睡觉": ["睡眠", "休息"],
    "工作": ["干活", "上班"],
    "学习": ["读书", "学习"],
    "高兴": ["开心", "快乐", "愉快"],
    "难过": ["伤心", "悲伤", "悲哀"],
    "生气": ["愤怒", "发火"],
    "害怕": ["恐惧", "担心"],
    "喜欢": ["喜爱", "爱"],
    "爱": ["喜欢", "疼爱"],
    "大": ["巨大", "庞大"],
    "小": ["微小", "细小"],
    "多": ["很多", "许多"],
    "少": ["不多", "很少"],
    "来": ["过来"],
    "去": ["过去"],
    "走": ["步行"],
    "跑": ["奔跑"],
    "买": ["购买"],
    "卖": ["出售"],
    "钱": ["货币", "金钱"],
    "书": ["书籍"],
    "车": ["汽车", "车辆"],
    "今天": ["本日", "今日"],
    "明天": ["明日", "次日"],
    "昨天": ["昨日", "前日"],
}

# ==================== 工具函数 ====================
def calculate_similarity(query: str, text: str) -> float:
    """计算查询词与文本的相似度"""
    query = query.lower().strip()
    text = text.lower().strip()
    
    if not query or not text:
        return 0.0
    
    # 精确匹配
    if query in text:
        return 1.0
    
    # 模糊匹配
    query_chars = set(query)
    text_chars = set(text)
    intersection = query_chars & text_chars
    return len(intersection) / len(query_chars) if query_chars else 0.0

def expand_query(query: str) -> List[str]:
    """扩展查询词（包含同义词）"""
    expanded = [query]
    
    # 检查是否有同义词
    for key, synonyms in SYNONYM_DICT.items():
        if query == key or query in synonyms:
            expanded.append(key)
            expanded.extend(synonyms)
            break
    
    return list(set(expanded))

def get_pinyin_score(query: str, pinyin: str) -> float:
    """计算拼音匹配分数"""
    query_no_space = query.replace(" ", "")
    pinyin_no_space = pinyin.replace(" ", "")
    
    if query_no_space == pinyin_no_space:
        return 1.0
    if query_no_space in pinyin_no_space or pinyin_no_space in query_no_space:
        return 0.8
    return 0.0

def calculate_score(query: str, word: dict, search_type: str) -> float:
    """计算综合搜索分数"""
    scores = []
    
    # 中文匹配（权重最高）
    chinese_score = calculate_similarity(query, word.get("chinese", ""))
    if search_type == "keyword":
        scores.append(chinese_score * 4.0)
    elif search_type == "semantic":
        scores.append(chinese_score * 1.5)
    else:
        scores.append(chinese_score * 4.0)
    
    # 拼音匹配
    pinyin_score = get_pinyin_score(query, word.get("pinyin", ""))
    if search_type == "keyword":
        scores.append(pinyin_score * 3.0)
    elif search_type == "semantic":
        scores.append(pinyin_score * 1.0)
    else:
        scores.append(pinyin_score * 3.0)
    
    # 含义匹配
    meaning_score = calculate_similarity(query, word.get("meaning", ""))
    if search_type == "keyword":
        scores.append(meaning_score * 2.0)
    elif search_type == "semantic":
        scores.append(meaning_score * 2.0)
    else:
        scores.append(meaning_score * 1.5)
    
    # 示例匹配（仅语义搜索）
    if search_type in ("semantic", "hybrid"):
        example_score = calculate_similarity(query, word.get("example", ""))
        scores.append(example_score * 1.0)
    
    # 同义词扩展匹配
    expanded_queries = expand_query(query)
    for eq in expanded_queries:
        synonym_score = calculate_similarity(eq, word.get("chinese", ""))
        if search_type == "semantic":
            scores.append(synonym_score * 1.5)
        else:
            scores.append(synonym_score * 1.0)
    
    return max(scores) if scores else 0.0

# ==================== API 路由 ====================
@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "HandTalk AI Sign Language Dictionary API", "version": "1.0.0"}

@app.get("/api/health")
async def health():
    """健康检查"""
    return {"code": 200, "message": "success", "data": {"status": "healthy"}}

@app.get("/api/v1/dictionary/search")
async def search_dictionary(
    query: str = "",
    search_type: str = "hybrid",
    top_k: int = 20,
    category: Optional[str] = None
):
    """
    搜索手语词典
    
    - query: 搜索关键词
    - search_type: 搜索类型 (keyword / semantic / hybrid)
    - top_k: 返回结果数量
    - category: 分类过滤
    """
    start = int(time.time() * 1000)
    
    if not query.strip():
        return {
            "code": 200,
            "message": "success",
            "data": {
                "results": [],
                "total": 0,
                "took_ms": int(time.time() * 1000) - start,
                "search_type": search_type
            }
        }
    
    # 过滤分类
    filtered_words = DICTIONARY_DATA
    if category and category != "all":
        filtered_words = [w for w in filtered_words if w.get("category") == category]
    
    # 计算每个词的相关分数
    scored_words = []
    for word in filtered_words:
        score = calculate_score(query, word, search_type)
        if score > 0:
            scored_words.append((word, score))
    
    # 按分数排序
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    # 取 top_k
    results = []
    for word, score in scored_words[:top_k]:
        result = {
            "id": word["id"],
            "chinese": word["chinese"],
            "pinyin": word["pinyin"],
            "meaning": word["meaning"],
            "category": word["category"],
            "video_url": word.get("video_url"),
            "thumbnail_url": word.get("thumbnail_url"),
            "example": word.get("example"),
            "score": score
        }
        results.append(result)
    
    took = int(time.time() * 1000) - start
    
    return {
        "code": 200,
        "message": "success",
        "data": {
            "results": results,
            "total": len(results),
            "took_ms": took,
            "search_type": search_type
        }
    }

@app.get("/api/v1/dictionary/categories/list")
async def get_categories():
    """获取所有分类"""
    categories = {}
    for word in DICTIONARY_DATA:
        cat = word.get("category", "other")
        if cat not in categories:
            categories[cat] = {"id": cat, "name": cat, "icon": "folder", "wordCount": 0}
        categories[cat]["wordCount"] += 1
    
    return {
        "code": 200,
        "message": "success",
        "data": list(categories.values())
    }

@app.get("/api/v1/dictionary/list")
async def get_words(page: int = 1, limit: int = 100):
    """获取词汇列表"""
    start = (page - 1) * limit
    end = start + limit
    items = DICTIONARY_DATA[start:end]
    
    return {
        "code": 200,
        "message": "success",
        "data": {
            "items": items,
            "total": len(DICTIONARY_DATA),
            "page": page,
            "limit": limit
        }
    }

@app.get("/api/v1/dictionary/{word_id}")
async def get_word(word_id: str):
    """获取指定词汇详情"""
    word = next((w for w in DICTIONARY_DATA if w["id"] == word_id), None)
    if not word:
        return JSONResponse(status_code=404, content={"code": 404, "message": "Word not found", "data": None})
    return {"code": 200, "message": "success", "data": word}

# ==================== 请求模型 ====================

class ImageRecognitionRequest(BaseModel):
    """图片识别请求模型"""
    base64_image: Optional[str] = None


# ==================== 手语识别接口（ResNet迁移学习） ====================

import torch
import numpy as np

# 尝试导入识别模块
try:
    from recognizer import SignLanguageRecognizer, get_recognizer
    RECOGNIZER_AVAILABLE = True
except ImportError:
    RECOGNIZER_AVAILABLE = False
    print("警告: 识别模块不可用，请先训练模型")

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.post("/api/v1/recognize/image")
async def recognize_image(request: ImageRecognitionRequest):
    """
    识别图片中的手语动作

    Args:
        request: 请求体，包含 base64_image

    Returns:
        识别结果
    """
    start = int(time.time() * 1000)

    base64_image = request.base64_image if request else ""

    try:
        # 解析图片
        image = None
        if base64_image:
            # 从 Base64 解析
            if base64_image.startswith("data:image"):
                base64_image = base64_image.split(",")[1]
            image = Image.open(BytesIO(base64.b64decode(base64_image)))
        else:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "message": "请提供 base64_image 参数", "data": None}
            )

        # 转换模式
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # 识别
        if RECOGNIZER_AVAILABLE:
            try:
                recognizer = get_recognizer()
                results = recognizer.recognize(image, top_k=5)

                took = int(time.time() * 1000) - start
                return {
                    "code": 200,
                    "message": "success",
                    "data": {
                        "results": results,
                        "took_ms": took
                    }
                }
            except Exception as e:
                # 模型未训练，返回模拟结果
                return get_mock_recognition_result(start)
        else:
            return get_mock_recognition_result(start)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"识别失败: {str(e)}", "data": None}
        )


@app.post("/api/v1/recognize/video")
async def recognize_video(file: UploadFile = None):
    """
    识别视频中的手语动作
    
    注意: 完整实现需要视频处理流水线
    当前返回模拟结果
    
    Args:
        file: 上传的视频文件
    
    Returns:
        识别结果列表
    """
    start = int(time.time() * 1000)
    
    if not file:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "请上传视频文件", "data": None}
        )
    
    # 读取文件大小
    contents = await file.read()
    file_size = len(contents)
    
    # 模拟视频处理
    # 实际实现应该:
    # 1. 使用 OpenCV 提取视频帧
    # 2. 对每帧进行识别
    # 3. 使用时间序列模型（如 LSTM、Transformer）融合结果
    
    took = int(time.time() * 1000) - start
    
    return {
        "code": 200,
        "message": "success",
        "data": {
            "results": [
                {
                    "text": "模拟识别结果",
                    "confidence": 85.5,
                    "start_time": 0.0,
                    "end_time": 2.0
                }
            ],
            "video_duration": 2.0,
            "processed_frames": 60,
            "took_ms": took
        }
    }


@app.get("/api/v1/recognize/status")
async def get_recognizer_status():
    """获取识别服务状态"""
    status = {
        "available": RECOGNIZER_AVAILABLE,
        "model_loaded": False,
        "device": str(DEVICE) if 'DEVICE' in dir() else "cpu"
    }
    
    if RECOGNIZER_AVAILABLE:
        try:
            recognizer = get_recognizer()
            status["model_loaded"] = recognizer.model is not None
        except:
            pass
    
    return {
        "code": 200,
        "message": "success",
        "data": status
    }


def get_mock_recognition_result(start: int):
    """返回模拟识别结果（用于测试）"""
    import random
    
    mock_results = [
        {"text": "你好", "confidence": 92.5, "class_id": 0},
        {"text": "谢谢", "confidence": 88.3, "class_id": 1},
        {"text": "再见", "confidence": 85.7, "class_id": 2},
        {"text": "妈妈", "confidence": 91.2, "class_id": 3},
        {"text": "爸爸", "confidence": 89.8, "class_id": 4},
    ]
    
    selected = random.sample(mock_results, min(3, len(mock_results)))
    selected.sort(key=lambda x: x['confidence'], reverse=True)
    
    took = int(time.time() * 1000) - start
    
    return {
        "code": 200,
        "message": "success (模拟结果，请训练模型以获得真实识别)",
        "data": {
            "results": selected,
            "took_ms": took,
            "note": "这是模拟结果，请运行 scripts/train_model.py 训练真实模型"
        }
    }


@app.get("/api/v1/recognize/health")
async def recognizer_health():
    """识别服务健康检查"""
    return {
        "code": 200,
        "message": "success",
        "data": {
            "status": "healthy",
            "module_available": RECOGNIZER_AVAILABLE
        }
    }


# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
