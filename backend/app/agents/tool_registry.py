from __future__ import annotations

from myagent.backend.app.models import AnalysisPurpose, ToolPlan, ToolSpec


TOOL_SPECS: dict[str, ToolSpec] = {
    "lean_canvas": ToolSpec(
        name="精益画布",
        canonical_name="lean_canvas",
        applicable_purposes=["decision_support"],
        description="梳理问题、用户细分、解决方案、渠道、成本收入和竞争壁垒。",
        output_schema_name="lean_canvas_schema",
        required_evidence_types=["market", "positioning", "pricing"],
        prompt=(
             "完成这几个维度的分析："
            "【0 一句话描述产品】，可使用的方法有：独特卖点 + 品类名称，类比法等；"
            "【1 问题】客户最需要解决的3个问题，可使用的方法有：黄金思维圈，5Why分析法，人性级需求，需求链等；"
            "【2 用户细分】目标用户 & 客户；用户的特征、标签，可使用的方法有：用户洋葱模型，干系人地图，影响地图，用户标签等；"
            "【3 独特卖点】为什么你的产品与众不同，值得购买，可使用的方法有：黄竞争导向，客户导向，战略画布等；"
            "【4 解决方案】产品最重要的3个功能，可使用的方法有：火车模型，需求优先级，KANO 模型，PSPS 模型等；"
            "【5 渠道】如何找到客户，如何推广，可使用的方法有：AARRR等；"
            "【6 关键指标】应该考核哪些东西，可使用的方法有：NPS，数据分析等；"
            "【7 竞争壁垒】无法被对手轻易复制或者买去的竞争优势，可使用的方法有：资金、技术、资源、品牌、网络效应…等；"
            "【8 成本分析】获取客户所需花费、销售花费、网站建设费用、人力资源费用等，可使用的方法有：比特世界，原子世界等；"
            "【9 收入分析】盈利模式、收入、毛利，可使用的方法有：延长价值链等；"
            "证据不足时请明确标记为数据缺口，不要编造。"
        ),
    ),
    "strategy_canvas": ToolSpec(
        name="战略画布",
        canonical_name="strategy_canvas",
        applicable_purposes=["decision_support"],
        description="比较价值曲线和差异化定位，识别目标产品可建立优势的竞争要素。",
        output_schema_name="strategy_canvas_schema",
        required_evidence_types=["positioning", "feature", "pricing"],
        prompt=(
            "基于证据生成战略画布。请识别关键竞争要素，并比较目标产品与竞品在各要素上的高低表现。"
            "在战略画布的横轴上列出产品的主要竞争元素,找出你的产品的主要竞争元素。不同产品的主要竞争元素可能会有很大差别，在实际操作中，可以从以下两个方面去找竞争元素：哪些因素会影响用户选择产品，影响用户体验？·哪些因素会在你的产品与竞品竞争时影响产品成败？"
            "根据竞品的表现，绘制竞品的价值曲线。应用市场调查、用户研究、亲自体验、头脑风暴等方式，在竞品的价值曲线上，对这些基本竞争元素进行加减乘除,举例说明：加：快捷酒店比星级酒店更快捷，特别是退房非常快，减：快捷酒店不像星级酒店那样有豪华的大堂，只有简约的前台。餐饮也不如星级酒店的自助餐，仅提供简单的早餐服务。·乘：快捷酒店创新性地提供了钟点房服务；通过品牌连锁，统一装修风格、统一服务标准。·除：快捷酒店没有游泳池、健身房、瑜伽课，房间也没有双人大浴缸、BOSE音响，也没有帮忙拿行李的门童，但这些都不是必需品，把这些锦上添花的元素去掉后，价格就可以大大降低。"
            "绘制差异化的价值曲线在竞品的价值曲线的基础上做“加减乘除”之后，就可以绘制差异化的价值曲线了,即在横坐标上填写竞争要素，纵坐标上填高，偏高	，偏低或低"
            "每个竞争要素的判断必须带 evidence_ids、confidence、reasoning；"
            "不能从证据推出的维度请标记为 unknown 或 data_gap。"
        ),
    ),
    "swot": ToolSpec(
        name="SWOT分析",
        canonical_name="swot",
        applicable_purposes=["decision_support"],
        description="总结优势、劣势、机会和威胁。",
        output_schema_name="swot_schema",
        required_evidence_types=["positioning", "pricing", "growth", "risk", "market"],
        prompt=(
            "基于证据执行 SWOT 分析。输出 strengths、weaknesses、opportunities、threats 四组。"
            "分析产品相对于竞品的优势（S）、劣势（W）。优势与劣势的参考要素如下所示。优势的参考要素：核心技术，充足的资金，良好的客户认可度，市场份额，生产率，产品/服务质量，管理团队，优秀员工，专利，内部流程，生产成本，研发能力，好的渠道，杰出战略，品牌。劣势的参考要素：核心竞争力，过时的厂房，陈旧的信息系统，缺乏资金，没有相应的专业知识，产品质量，原材料供应紧张，管理水平，低劣品牌，内部权力斗争，高成本结构，低水平的营销队伍，渠道伙伴不靠谱，产品单一。这些参考要素并非都是必须的，要根据竞品分析的目标、产品所在的行业进行选择，最后整理得出的优势与劣势一般每种不超过5个。"
            "分析产品面临的外部机会（O）与威胁（T），它们可能来自外部环境因素的变化，也可能来自竞争对手力量的变化。机会的参考要素：经济增长、消费升级、政策、开拓新市场、相关领域的多元化拓展、转向海外、及时掌握新技术、满足新消费群体、兼并其他公司、有利的政策法规、加入战略联盟、并购机会。威胁的参考要素：不断增加的竞争压力、替代产品出现、市场增长放缓、汇率波动、贸易政策改变、客户削减投资、人力成本持续增加、经济衰退、需求变化、政府管制条例、不断提升的消费者期望、影响环保。这些参考要素并非都是必须的，要根据竞品分析的目标、产品所在的行业进行选择，最后整理得出的机会与威胁一般每种不超过5个。"
            "根据SWOT分析，可以得到很多竞争策略的可选项：扬长（发挥优势）,避短（规避劣势）,趋利（抓住机会）,避害（避开威胁）。还可以把优势、劣势与机会、威胁进行组合，以得出更多的竞争策略,如SO战略就是依靠内部优势去抓住外部机会的战略。WO战略是利用外部机会来克服内部劣势的战略。ST战略就是利用企业的优势去避免或减轻外部威胁的战略。WT战略就是直接减少内部劣势和避免外部威胁的战略。"
            "每条都必须是可追溯结论，带 evidence_ids、confidence、reasoning；"
            "证据不足时降低 confidence 并说明缺口。"
        ),
    ),
    "porter_five_forces": ToolSpec(
        name="波特五力模型",
        canonical_name="porter_five_forces",
        applicable_purposes=["decision_support"],
        description="评估行业吸引力、进入壁垒、替代品、供应商议价和买方议价能力。",
        output_schema_name="porter_five_forces_schema",
        required_evidence_types=["market", "risk", "pricing", "funding"],
        prompt=(
            "基于证据生成波特五力分析：同行业竞争者、潜在进入者的威胁、替代品的威胁、供应商议价能力、买方议价能力。"
            "同行业竞争者:这种竞争力量是企业所面对的最强大的一种力量，评估时要特别关注同行业中现有竞争者的数量和竞争强度"
            "潜在进入者的威胁:如果一个行业的进入门槛很低，或者利润空间很大，必然导致大量的新进入者涌进这个行业。新进入者进入该行业，会抢夺市场份额，与现有企业激烈竞争，使产品价格下跌；另一方面，新进入者要获得资源进行生产，可能会使行业生产成本升高。这两方面都会导致行业的利润水平下降。"
            "替代品的威胁:替代品虽然与你的产品形式不同，但能满足相同的需求，可以相互替代，所以也会影响你的产品的市场份额。替代品的价格如果比较低，它投入市场就会使本行业产品的价格上限只能处在较低的水平，这就限制了本行业的利润。"
            "供应商议价能力:供应商的议价能力的强弱，主要取决于供应商行业的市场现状以及他们所提供商品的重要性。供应商如果提高供应价格，或者降低供应产品或服务的质量，就会使下游行业利润下降。"
            "购买者的议价能力:购买者会要求降低产品价格，或要求高质量的产品和更多的优质服务，这就会使行业的竞争者们相互竞争，导致行业的利润水平下降。"
            "示例，仅供参考：纳米盒App的波特五力模型：同行业竞争者包含家长通、小孩子点读；潜在进入者的威胁包含作业帮；购买者的议价能力包含有很多竞品供选择、转换成本很低；替代品的威胁包含步步高点读机、好未来（线下）；供应商的议价能力包含出版社资源授权。"
            "每个判断都必须带 evidence_ids、confidence、reasoning；"
            "不要用常识替代证据。"
        ),
    ),
    "pest_analysis": ToolSpec(
        name="PEST分析",
        canonical_name="pest_analysis",
        applicable_purposes=["decision_support", "market_warning"],
        description="从政治、经济、社会、技术四个宏观维度识别外部机会与风险。",
        output_schema_name="pest_analysis_schema",
        required_evidence_types=["market", "risk", "growth", "technology"],
        prompt=(
            "基于证据执行 PEST 分析。请从 Political、Economic、Social、Technological 四个维度"
            "政治环境：主要包括政治制度与体制、政策、政府的态度等，也包括政府制定的法律、法规。"
            "经济环境：是影响产业利润的主要因素之一，国家经济环境直接影响消费者的消费能力，从而影响企业的生存环境。经济环境主要包括：GDP及增长率、利率、汇率、通货膨胀率、失业率、消费、投资等。"
            "社会环境：消费者生存在社会环境中，而社会环境的改变直接影响了企业的发展能力。社会环境主要包括人口规模、年龄结构、人口分布、收入、生活方式、种族结构、教育、消费习惯以及社会观念等因素。"
            "技术环境：科技是企业发展的驱动力，也是企业竞争优势所在。技术环境包括技术变革速度、技术发明、专利及保护情况、国家对科技项目的投资等。"
            "举例说明，仅供参考：纳米盒App（面向小学生在线教育产品）的PEST分析表，具体内容如下：政治（Politics）包含政策持续刺激促进教育信息化产业的发展，包括支持三通两平台、微课、创客、STEAM教育；经济（Economy）包含国家与家庭投入拉动需求，资本推动行业竞争，以及消费升级、知识付费的习惯养成；社会（Society）包含移动互联网已经渗透到大众生活的每一个角落，二胎开放，以及“70后”“80后”家庭普遍重视教育；技术（Technology）包含大数据智能分析，语音识别与评测，视频与直播互动，智能设备与终端，以及3D打印、AR、VR等技术出现与完善，不断优化教学。"
            "每个判断必须带 evidence_ids、confidence、reasoning；"
            "无法由证据支撑的宏观推断必须标记为 data_gap，不要凭常识编造。"
        ),
    ),
    "competitor_canvas": ToolSpec(
        name="竞品画布",
        canonical_name="competitor_canvas",
        applicable_purposes=["learning"],
        description="围绕用户痛点、解决方案和体验亮点拆解竞品。",
        output_schema_name="competitor_canvas_schema",
        required_evidence_types=["positioning", "feature", "review"],
        prompt=(
            "基于证据生成竞品画布，聚焦目标用户、用户痛点、解决方案、体验亮点和可借鉴点。"
            "每条结论必须带 evidence_ids、confidence、reasoning；"
            "如果某个竞品证据不足，请单独说明。"
        ),
    ),
    "feature_breakdown": ToolSpec(
        name="功能拆解",
        canonical_name="feature_breakdown",
        applicable_purposes=["learning"],
        description="拆解竞品的一二三级功能、核心路径和版本变化。",
        output_schema_name="feature_tree_schema",
        required_evidence_types=["feature", "review"],
        prompt=(
            "基于证据执行功能拆解。输出每个竞品的一级、二级、三级功能、核心路径和明显体验取舍。"
            "1.按菜单导航拆解:通过竞品主界面的菜单、导航、按钮，可以快速拆解出一级功能、二级功能。"
            "2.按使用流程拆解:通过利用竞品完成一些业务流程，可以发现很多功能点。例如:在使用淘宝购物的整个流程中，可以发现搜索商品、查看商品详情、购物车、提交订单、结算、支付等功能。"
            "3.按交互操作拆解:通过竞品的交互操作方式发现功能，比如双击、长按、拖动、右键等。移动端有更多的传感器与交互方式:滑动(上下左右)、多点触控、音量键、Home键、返回键、耳机孔、重力感应、摄像头、数据线接入、语音输入等。这些传感器与交互方式可能会触发一些功能，这些功能往往不是显而易见的，在功能拆解时要多尝试各种交互方式，避免遗漏功能点。例如，在微信想要发一条纯文本的朋友圈时，需要长按发朋友圈的按钮才会出现这个功能。"
            "4.看产品说明书拆解:竞品的产品说明书、使用手册、版本更新记录往往也会介绍竞品的主要功能。我们要一边拆解竟品功能，一边将获得的信息记录在思维导图或填写在功能分析"
            "举例说明，仅供参考：功能拆解-微信朋友圈，具体内容如下：表头包含竞品名称、一级功能、二级功能、三级功能、探索需求、是否借鉴此功能、总结/备注。竞品名称为微信，一级功能为朋友圈。对应的二级功能、三级功能及备注信息如下：新消息提示；刷新消息，总结/备注为下拉消息列表；发布文字信息，探索需求为寻求认可、获得存在感，满足“骄傲/虚荣”的人性级需求，总结/备注为长按相机图标；发布图片，对应的三级功能有拍照和从手机相册选择，其探索需求同样为寻求认可、获得存在感，满足“骄傲/虚荣”的人性级需求；查看朋友发的图片，对应的三级功能有查看照片、发送给朋友、收藏、保存到手机；查看朋友发的链接，对应的三级功能有查看链接中的内容、分享；评价朋友的信息，对应的三级功能有点赞、评论；删除我的评论；设置朋友圈权限，总结/备注为长按朋友的头像；收藏内容，总结/备注为长按消息；复制内容，总结/备注为长按文字消息；查看好友相册，对应的三级功能有赞封面、查看内容、查看详细资料；查看我的相册，对应的三级功能有消息列表、更换相册封面（总结/备注为点击相册图片）、查看详情。"
            "每个功能或路径判断必须带 evidence_ids、confidence、reasoning；"
            "优先引用官网、版本更新、应用商店和用户评论证据。"
        ),
    ),
    "needs_exploration": ToolSpec(
        name="探索需求",
        canonical_name="needs_exploration",
        applicable_purposes=["learning"],
        description="从评论、社区反馈和差评中挖掘深层需求。",
        output_schema_name="needs_exploration_schema",
        required_evidence_types=["review", "feature"],
        prompt=(
            "基于证据探索用户需求。请从抱怨、差评、帮助文档、功能描述和社区反馈中推断未满足需求。"
            "用户需求有3个层次，依次为：方案级需求→问题级需求→人性级需求"
            "我们拿到用户的需求后，不要直接去满足这一需求，而是要挖掘方案级需求对应的问题级需求，甚至人性级需求。那么，怎么探索更深层次的需求呢？最简单有效的方法，就是多问“为什么”，此处我们推荐“5Why分析法”。就是要对一个问题连续以5个“为什么”来发问，以追究其根本原因"
            "举例说明，仅供参考：问题1：为什么车间的机器停了？答案1：因为机器超载，保险丝烧了。问题2：为什么机器会超载？答案2：因为轴承的润滑不足。问题3：为什么轴承会润滑不足？答案3：因为润滑泵失灵了。问题4：为什么润滑泵会失灵？答案4：因为它的轮轴磨损了。问题5：为什么润滑泵的轮轴会磨损？答案5：因为杂质跑到里面去了。最后总结道：经过连续5次不停地问“为什么”，终于找到问题的真正原因，得出最后的解决方法，即在润滑泵上加装滤网。"
            "在应用5Why分析法时，要注意虽为5个为什么，但使用时不一定是问5次，要找到根本原因为止，有时可能只需要问3次，有时也许需要问6次或更多次。"
            "每条需求必须带 evidence_ids、confidence、reasoning；"
            "区分显性需求、隐性需求和暂无法验证的假设。"
        ),
    ),
    "errac": ToolSpec(
        name="加减乘除(ERRAC)",
        canonical_name="errac",
        applicable_purposes=["learning"],
        description="提出删除、减少、增加、创造的差异化创新动作。",
        output_schema_name="errac_schema",
        required_evidence_types=["feature", "review", "positioning"],
        prompt=(
            "基于证据输出加减乘除(ERRAC)：消除、减少、提升、创造。"
            "在战略画布中，先描绘竞品的价值曲线，再在竞品的价值曲线的基础上做“加减乘除”，这样可以描绘出与竞品完全不同的价值曲线，实现产品的差异化创新。应用战略画布及“加减乘除”绘制差异化的价值曲线的关键步骤如下：1）在战略画布的横轴列出产品的主要竞争元素。2）根据竞品的表现，绘制竞品的价值曲线。3）在竞品的基础上，对这些竞争元素应用“加减乘除”的方法。加：哪些竞争元素的表现可以比竞品好一些？用户对竞品的现状有哪些不满意的地方？针对以上问题，我们可以有目的地进行优化。减：哪些竞争元素的表现可以比竞品差一些？看看竞品是否在功能上过度设计，所提供的超过用户所需的功能徒然增加成本却没有好效果。我们通过弱化这些竞争元素来降低成本。乘：哪些元素是同行中从未有过的，可以创新？也就是要发现并创造新的用户价值，提升产品的竞争力。除：哪些元素是被同行认定为是理所当然的，需要删除？删除为了竞争而攀比的元素，这些元素经常被认为是理所当然的，虽然他们不再具有价值，甚至还减少了产品的价值。4）绘制差异化的价值曲线。"
            "举例说明，仅供参考："
            "每个建议必须能追溯到竞品证据，并带 evidence_ids、confidence、reasoning；"
            "建议要具体、可执行，避免空泛口号。"
        ),
    ),
    "tracking_matrix": ToolSpec(
        name="竞品跟踪矩阵",
        canonical_name="tracking_matrix",
        applicable_purposes=["market_warning"],
        description="按时间线跟踪版本、营销、融资、招聘和技术异动。",
        output_schema_name="tracking_matrix_schema",
        required_evidence_types=["growth", "funding", "hiring", "technology", "risk"],
        prompt=(
            "基于证据生成竞品跟踪矩阵。请按竞品列出近期版本更新、融资、招聘、技术和营销异动。"
            "竞品跟踪矩阵包括几个要素：时间、竞品每个历史版本的版本号、每个版本的变化要点以及外部环境变化"
            "按照时间线，对竞品的各个版本变化情况进行跟踪，记录版本号以及该版本的变化情况，比如：新增了哪些功能、优化了哪些功能、删除（弱化）了哪些功能等。"
            "同时，对外部环境的变化情况也要做好标记（例如，产品相关的政策、经济、技术、行业环境的变化），结合竞品的版本变化情况做进一步的分析解读。"
            "每个异动必须带 evidence_ids、confidence、reasoning；"
            "标出可能构成市场预警的信号。"
        ),
    ),
    "matrix_analysis": ToolSpec(
        name="矩阵分析法",
        canonical_name="matrix_analysis",
        applicable_purposes=["market_warning"],
        description="以二维矩阵比较资源倾斜、风险和增长信号。",
        output_schema_name="matrix_analysis_schema",
        required_evidence_types=["growth", "risk", "funding", "hiring"],
        prompt=(
            "基于证据生成二维矩阵分析。请说明横轴、纵轴、各竞品位置、位置理由和预警含义。"
            "矩阵分析法看起来简单清晰，使用方法也比较简单，具体如下。1）确定两个关键竞争要素，例如，价格与配置。这两个竞争要素应该是用户最关注的，或者是对用户最重要、会影响他们购买决策的产品属性。2）画出二维矩阵，把两个关键竞争要素分别作为横坐标和纵坐标。3）选择几个主要竞品。4）根据竞品在关键竞争要素的表现，把竞品放到矩阵对应的位置。5）在矩阵中思考自己产品的位置。"
            "每个位置判断必须带 evidence_ids、confidence、reasoning；"
            "不要把无证据的趋势写成事实。"
        ),
    ),
    "comparison": ToolSpec(
        name="比较法",
        canonical_name="comparison",
        applicable_purposes=["market_warning"],
        description="对关键指标进行横向硬碰硬比较。",
        output_schema_name="comparison_schema",
        required_evidence_types=["pricing", "feature", "growth", "risk"],
        prompt=(
            "基于证据执行比较法。请围绕关键指标横向比较竞品，指出领先、落后和风险点。"
            "根据比较的形式，比较法可以分为三种：打钩比较法、评分比较法、描述比较法。1.打钩比较法：打钩比较法可以用于产品的功能、配置、特性的对比分析。通过对比产品与竞品的功能，可以全方位地了解竞品的功能分布，为自己产品的功能规划做参考。2.评分比较法：评分比较法可以用于用户体验设计、$APPEALS各要素等方面的横向比较，通过比较可以清晰直观地发现产品与竞品之间的差异，并通过分析得到产品的优势与劣势。3.描述比较法：描述比较法多用于功能细节、界面的比较，可以详细描述各竞品的具体表现、优缺点等。"
            "每项比较必须带 evidence_ids、confidence、reasoning；"
            "缺失数据要明确标注，不要强行比较。"
        ),
    ),
}


ALIASES: dict[str, str] = {
    "精益画布": "lean_canvas",
    "战略画布": "strategy_canvas",
    "SWOT分析": "swot",
    "波特五力模型": "porter_five_forces",
    "PEST分析": "pest_analysis",
    "竞品画布": "competitor_canvas",
    "功能拆解": "feature_breakdown",
    "探索需求": "needs_exploration",
    "加减乘除(ERRAC)": "errac",
    "竞品跟踪矩阵": "tracking_matrix",
    "矩阵分析法": "matrix_analysis",
    "比较法": "comparison",
}


DEFAULT_TOOLS_BY_PURPOSE: dict[AnalysisPurpose, list[str]] = {
    "decision_support": ["lean_canvas", "strategy_canvas", "swot"],
    "learning": ["feature_breakdown", "needs_exploration", "errac"],
    "market_warning": ["tracking_matrix", "matrix_analysis", "comparison"],
}


def normalize_tool_name(name: str) -> str:
    stripped = name.strip()
    if stripped in ALIASES:
        return ALIASES[stripped]

    lower = stripped.lower().replace(" ", "_")
    if lower in ALIASES:
        return ALIASES[lower]
    if lower in TOOL_SPECS:
        return lower

    if "pest" in lower or "宏观环境" in stripped:
        return "pest_analysis"
    if "swot" in lower:
        return "swot"
    if "errac" in lower or "加减乘除" in stripped or "四步动作" in stripped:
        return "errac"
    if "功能" in stripped or "feature" in lower:
        return "feature_breakdown"
    if "需求" in stripped or "needs" in lower:
        return "needs_exploration"
    if "跟踪" in stripped or "追踪" in stripped or "tracking" in lower:
        return "tracking_matrix"
    if "矩阵" in stripped or "matrix" in lower:
        return "matrix_analysis"
    if "比较" in stripped or "comparison" in lower:
        return "comparison"
    if "战略" in stripped or "价值曲线" in stripped or "strategy" in lower:
        return "strategy_canvas"
    if "精益" in stripped or "lean" in lower:
        return "lean_canvas"
    if "五力" in stripped or "porter" in lower:
        return "porter_five_forces"
    if "竞品画布" in stripped or "competitor_canvas" in lower:
        return "competitor_canvas"
    return lower


def get_tool_spec(name: str) -> ToolSpec:
    canonical = normalize_tool_name(name)
    return TOOL_SPECS.get(
        canonical,
        ToolSpec(
            name=name,
            canonical_name=canonical,
            applicable_purposes=[],
            description="通用分析工具。",
            output_schema_name="generic_tool_schema",
            required_evidence_types=["other"],
            prompt=(
                "基于证据输出结构化分析结论。每条结论必须带 evidence_ids、confidence、reasoning；"
                "证据不足时请明确说明数据缺口。"
            ),
        ),
    )


def allowed_tools_for_purpose(purpose: AnalysisPurpose) -> list[ToolSpec]:
    return [TOOL_SPECS[name] for name in DEFAULT_TOOLS_BY_PURPOSE[purpose]]


def tool_specs_for_plan(tool_plan: ToolPlan) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for tool in tool_plan.selected_tools:
        spec = get_tool_spec(tool)
        if spec.canonical_name not in seen:
            seen.add(spec.canonical_name)
            specs.append(spec)
    if not specs:
        specs = allowed_tools_for_purpose(tool_plan.purpose)
    return specs
