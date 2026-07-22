# TikTok “当下热门”商品视频脚本：官方信号源与接入规则

> 研究日期：2026-07-22（Asia/Shanghai）  
> 范围：仅使用 TikTok、TikTok for Business 和 TikTok Shop 官方资料。

## 结论

要增加的不是一份静态“热门脚本库”，而是一个 **live trend grounding** 步骤：每次生成脚本前，先按目标市场、语言、商品类目和时间窗采集当下信号，再把证据抽象成“Hook、叙事结构、演示方式、节奏、CTA 和音乐氛围”。不复制原脚本、原画面、创作者身份或未授权音乐。

“热门”至少要有一类可见证据：

- 搜索/内容需求在增长；
- 同类广告已显示高互动或高转化信号；
- TikTok Shop 已显示品类、关键词或商品需求信号。

只有高累计播放、一个偶然爆款，或个人 FYP 上看到一次，不足以宣称“当下热门”。

## 应该去哪里看

| 优先级 | 官方来源 | 最适合回答的问题 | 建议记录的证据 |
|---|---|---|---|
| 1 | [TikTok One Insights Spotlight / Custom insights](https://ads.tiktok.com/help/article/about-custom-insights-in-tiktok-one) | 目标人群近 7 天在看什么、搜什么，哪些话题正在增长 | 地区、语言、年龄/性别、关键词、关联标签、相关内容 URL、近 7 天观看/帖子数与环比增长、14 天趋势 |
| 2 | [Creative Center – Trends](https://ads.tiktok.com/creative/creativeCenter/trends/hashtag?period=7&region=US) | 目标国家和行业近 7 天的热门标签和相关视频 | 查询时间、region、period、industry、标签排名、帖子/观看量、趋势变化、相关视频、受众/地区分布、相关标签 |
| 3 | [TikTok One – Top Ads Insights](https://ads.tiktok.com/help/article/about-top-ads-insight-in-tiktok-one?lang=en) | 同类产品的什么 Hook、selling point、演示和节奏已显示效果 | 案例 URL/ID、地区、行业、时间窗、selling point、视频播放/CTR/6s 播放率/互动率中实际可见值、Hook、证明动作、CTA |
| 4 | [Creator Search Insights](https://support.tiktok.com/en/using-tiktok/growing-your-audience/creator-search-insights) | 用户正在主动搜索什么，哪些问题有“内容缺口” | 精确查询词、popularity、Recommended Topic/Content gap、相关搜索、相关视频 URL、查询日期、账号市场/语言 |
| 5 | Keyword Insights（[官方功能说明](https://ads.tiktok.com/help/article?aid=14099)；[字段示例](https://ads.tiktok.com/business/creativecenter/tiktok-keyword/top/pc/en)） | 同类广告用什么词表达价值、这些词常出现在口播还是字幕 | 地区、行业、目标、关键短语、Most used as（voice-over/text overlay/ad text）、CTR、相关视频、Popular/Breakout、相关关键词与标签 |
| 6 | [TikTok Shop Product Opportunities（美国 Seller Center）](https://seller-us.tiktok.com/university/essay?knowledge_id=4371484668528427&lang=en) | 当周哪些商品、搜索词、视频标签和子类目有需求 | Featured products、Trending searches、Trending video hashtags、subcategory、供需数据、market trend、Shoppable Video 数据、匹配标准 |
| 7 | [Catalog Insights](https://ads.tiktok.com/help/article/about-catalog-insights?lang=en) | 已连接目录的商家，自己的哪些 SKU/类目正在获得需求和购买意向 | 商品/类目排名、SKU、品牌、价格、库存、匹配商品数、视频、导出 CSV 时间 |
| 8 | [Commercial Music Library](https://ads.tiktok.com/help/article/how-to-use-the-commercial-music-library?lang=en) | 选中的氛围/节奏音乐是否可用于目标地区的商业内容 | 曲目名/ID、目标地区、可用版位、主题、类型、情绪、时长、查询日期、授权状态 |

### 高阶来源（有权限时再用）

- [TikTok Market Scope](https://ads.tiktok.com/help/article/about-tiktok-market-scope?lang=en)：Vertical Insights 可看品类中的热门视频、主题和创作者，Merchandise Insights 可看商品类目需求；仅部分客户可用，需联系 TikTok 销售/客户经理。
- [TikTok One Insights Spotlight 官方介绍](https://ads.tiktok.com/business/en-US/blog/insights-spotlight-trends-tool)：用于实时趋势、搜索行为、受众兴趣和品牌讨论的第一方信号。
- [TikTok Shop Market Insights（美国创作者侧）](https://seller-us.tiktok.com/university/essay?default_language=en&knowledge_id=3431256794089259)：可看 30 天商品/品牌/关键词搜索、品类热门视频和 hashtag 增减；仅 App 内并需电商创作者权限，音乐热度不代表商用授权。

## 最小可执行流程

1. **锁定市场**：必须有目标 country/region、语言、商品类目和发布目标（自然内容、广告或 TikTok Shop）。不应默认用 `All regions` 代替实际市场。
2. **先看近 7 天**：使用 Trends / Insights Spotlight / Creator Search Insights；低量类目才放宽至 30 天，并标记为稳定模式而非即时爆点。
3. **再看效果**：在 Top Ads 中用同地区+同行业+同目标+当前时间窗找可类比广告，查看秒级高峰，不只看总点赞。
4. **补语言信号**：优先用 Keyword Insights；若当前账号/地区不可用，就从 Top Ads 和 Creator Search 的多个案例中手工归纳重复表达，必须标记这是归纳，不是平台排名。
5. **商品内容加一层 Shop 信号**：有 Seller Center 权限时查 Product Opportunities；有已连接目录时查 Catalog Insights。它们回答“用户想买什么”，Top Ads 回答“怎么讲更有效”，两者不能互相替代。
6. **音乐单独过权利门**：“当下流行”不等于“商业可用”。一定按实际投放/发布地区在 CML 复核。
7. **当次生成当次采集**：不把上次的趋势结果默认当成本次的“当下”。脚本交付时附上查询时间和来源 URL。

> 建议的内部门槛（这是工作流规则，不是 TikTok 官方规定）：至少有 **2 类独立信号 + 3 个可访问案例 URL** 才可声称“基于当下热门”。否则只能标记为“平台原生风格/长青叙事”。

## 趋势证据包（建议输出）

```yaml
trend_research:
  checked_at: "ISO-8601 timestamp with timezone"
  target_region: "US / SG / ..."
  language: "en-US / ..."
  product_category: "..."
  timeframe: "last 7 days"
  sources:
    - url: "official TikTok URL"
      source_type: "trends | top_ads | search | shop | cml"
      filters: "region, industry, objective, timeframe"
      observed_signal: "rank/growth/performance actually visible"
  selected_pattern:
    hook_type: "..."
    narrative: "..."
    proof_device: "..."
    pacing: "..."
    cta_style: "..."
  commercial_music:
    status: "verified_in_CML | original_owned | separately_licensed | none"
    region: "..."
  limitations: "login, locale, low sample, unavailable metric, etc."
```

六格故事版中，趋势证据只能影响叙事表达；`identity_master` 、用户产品图和经核验的商品信息仍是人物与产品事实来源。热门视频不是商品功效证据，不得因为趋势而增加未验证功效、价格、折扣或用户反馈。

## 如何“学模式”而不是复制

TikTok 的 [Creative Codes 趋势指南](https://ads.tiktok.com/business/library/Creative_Codes_ENG.pdf) 建议先找趋势、建立品牌/产品相关性，再选择适合的叙事，并把其他视频作为灵感。实际执行时应：

- 只抽取功能模式：Hook 类型、视角、冲突、产品证明动作、镜头节奏、payoff 和 CTA 类型。
- 重写所有口播和屏幕文字，让它们回答该 SKU 的真实问题与真实卖点。
- 至少融合多个案例的抽象特征，不按单条广告逐镜重现。
- 不使用原创作者的脸、声音、标志性口头禅、品牌标识、水印、素材文件或音轨。
- 保留每个参考 URL 和“只借鉴了什么”，方便审核。

TikTok 的 [Intellectual Property Policy](https://www.tiktok.com/legal/page/global/copyright-policy/en?lang=en) 明确区分“思想/事实”与受保护的具体表达，并禁止未经授权使用他人版权内容。Top Ads 应用于研究与抽象，不是素材下载库。

## 音乐红线

TikTok 官方的 [CML 授权说明](https://ads.tiktok.com/help/article/commercial-music-library?lang=en) 要求商业活动（包括自然商业内容、广告和品牌内容）使用 Commercial Music Library；如使用他人原声或其他受许可音乐，需另行确认合法授权。

因此脚本可以学习当前热门音频的 **BPM 区间、情绪、起伏和卡点方式**，但不能默认指定原热门曲。最终曲目必须在目标地区的 CML 中查到可用，或有品牌自有/单独授权证明。

## 2026-07-22 的访问限制与不可假装的状态

- Creative Center 仍是官方免费入口，但登录后才能看到更完整的功能和数据。[Top Ads 官方说明](https://ads.tiktok.com/help/article/how-to-use-the-top-ads-dashboard?lang=en) 明确表示，未登录时只显示 5 条广告；桌面端才有更完整的语言、格式和排序选项。
- 新工作流应优先使用 TikTok One Top Ads Insights。[旧版 Creative Center Top Ads](https://ads.tiktok.com/business/creativecenter/inspiration/topads/pc/en) 可作辅助读取，但已进入迁移，不应当作会持续更新的主实时源。Top Ads 只是经广告主授权展示的高表现创意集合，不是全部 TikTok 广告的完整样本；表现高峰不能被解读为因果证明。
- 新版 Trends 官方页面当前重点展示 hashtag 和 video，creator 标注 `Coming soon`；旧的 music/creator 链接会跳转到新版 TikTok One Creative Suite。因此 Skill 不能写死“一定能获取热门歌曲/创作者榜单”，应先检查当前 UI 是否实际可用。
- Keyword Insights 和 Top Products 的部分旧 Creative Center URL 已跳转至新 Creative Center/TikTok One 首页；功能可见性会受地区、账号、设备与产品迁移影响。如当次无法读取，要明示记录 `unavailable`，不得用旧数据伪装成实时结果。
- Creator Search Insights 需在 TikTok App 内使用；“Searches by followers”过滤器要求账号超过 1,000 粉丝，而且结果会受账号和市场影响。
- Product Opportunities 需 Seller Center 权限，上述官方资料是美国站，不能直接将美国数据外推为新加坡、东南亚或英国趋势。
- Catalog Insights 排名依赖 Catalog 与 pixel/app events 的正确连接；连接不完整时不应当作可靠的商品热度证据。
- CML 的桌面与移动端可用曲目可能不同，并且音乐可用性受地区和版位限制；官方页面一次最多显示 10,000 条 Commercial Sounds。
- Market Scope 只向部分客户开放；没有权限时，不能把它列为必修前置。

## 对新 Skill 能力的建议要求

1. 当用户要求“当下热门/TikTok 爆款”时，必须实时访问官方信号源，不从 Skill 静态记忆中直接生成。
2. 调研前先确定目标市场、语言、类目和发布目标；无地区就不能声称“当下热门”。
3. 默认近 7 天，低量时扩至 30 天并明示降级；交付中保留 checked_at、filters 和官方 URL。
4. 用搜索/趋势信号选题，用 Top Ads 研究结构，用 Shop/Catalog 信号校验商品需求，用 CML 完成音乐授权门。
5. 只迁移抽象模式，必须重写口播、字幕和情节；不得逐镜、逐句或单来源复制。
6. 趋势不得改写 `identity_master`、产品外观和已验证商品事实，也不得生成未验证功效。
7. 无权限、无匹配地区或样本不足时，输出“趋势证据不足”，并降级为平台原生的长青脚本；不能伪造实时热度。
