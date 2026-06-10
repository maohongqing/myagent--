# MyAgent 竞品分析 Agent

MyAgent 是一个面向竞品分析场景的全栈 Multi-Agent 应用。后端基于 FastAPI 和 LangGraph，前端基于 React 和 Vite，内置多源搜索、网页正文抽取、证据截图、资料卡构建、工具化分析、质量检查和 Markdown 报告生成能力。

项目适合用于：

- 产品立项前的竞品调研
- 已上线产品的市场预警和对手跟踪
- 功能设计阶段的竞品学习借鉴
- 面向目标用户的竞品调研问卷设计与回收结果分析
- 面向产品、运营、战略团队的结构化竞品报告生成

## 目录结构

```text
myagent/
  backend/              FastAPI 后端、LangGraph 工作流、存储、导出、测试
  frontend/             React + Vite 前端
  prompt/               当前主流程使用的 Agent prompt
  search/               多源搜索、网页抽取、截图、OCR、LLM 抽取工具
  docs/                 项目设计文档
  data/                 前端或项目级数据目录
  docker-compose.yml    本地 Postgres 启动配置
  .env.example          环境变量示例
```

说明：`search` 已经移动到 `myagent/search`，代码中通过 `myagent.search` 包导入。命令行推荐使用 `python -m myagent.search.search_cli` 启动。

## 核心功能

### 1. 多阶段竞品分析工作流

后端工作流覆盖从需求理解到最终报告的完整链路：

- 需求解析：识别目标产品、行业、市场、生命周期、分析目的和关键挑战。
- 候选发现：基于搜索 query 和网页 chunk 抽取候选竞品。
- 候选 QA：验证候选是否真实存在、是否相关、是否需要合并或修正类型。
- 市场热度补充：为候选竞品生成补充搜索 query，收集市场热度和增长信号。
- 竞品评分：按热度、增长、相似度和分析目的进行单竞品评分。
- 阵型质检：检查最终竞品组合是否符合分析目标。
- 分析策略规划：生成工具计划和采集计划。
- 资料卡采集：按竞品和行业维度生成 query、抽取字段、补搜缺口、做资料卡 QA。
- 工具执行：按 SWOT、五力模型、精益画布等工具输出结构化 claim。
- 综合分析：汇总资料卡、工具结果和状态树，形成分析结果。
- 最终 QA：检查状态树一致性、证据支撑和建议落地性。
- 报告生成：分章节生成 Markdown 竞品分析报告。

### 2. 搜索与证据能力

`myagent/search` 提供可单独运行的搜索工具，也被后端工作流调用：

- 支持 DuckDuckGo、Tavily、Brave、SearchApi、SerpApi 等搜索源。
- 支持从已知 URL 作为 seed 开始采集。
- 支持 Playwright 打开网页、读取正文、保存截图。
- 支持本地链接深挖和 Tavily Map/Crawl/Extract 深挖。
- 支持按文本 chunk 或 viewport chunk 切分网页。
- 支持 LLM 抽取竞品信息。
- 支持 OCR 和图片理解补充。
- 支持 chunk 级证据截图和 quote 级证据定位。

### 3. 报告与导出

- 竞品画像
- 竞品选择理由
- 工具分析结果
- 关键发现
- 总结建议
- 附录来源索引
- QA 历史和风险提示
- Markdown、JSON、PDF、DOCX 导出

### 4. 问卷设计与用户反馈分析

系统内置问卷辅助能力，用于把公开竞品分析和一手用户反馈结合起来：

- 根据目标产品、竞品集合、行业和分析目标生成调研问卷。
- 自动设计题型，包括单选、多选、排序、量表和开放题。
- 将问题映射到竞品认知、购买决策、满意度、价格敏感度、痛点和改进建议等维度。
- 支持上传问卷回收结果，解析 CSV、JSON 数组或带表头的表格文本。
- 汇总样本规模、关键发现、竞品洞察、用户分群线索、开放题主题和样本局限。
- 可将问卷分析作为附录补充到最终 Markdown 报告中。

相关后端模块：

- `backend/app/agents/survey.py`
- `prompt/survey_agent.py`

### 5. 存储

后端支持：

- SQLite，默认推荐本地开发使用
- JSON 文件存储
- Postgres，可通过 `docker compose` 本地启动

## 环境要求

- Python 3.11+
- Node.js 18+ 和 npm
- 可选：Docker，用于本地 Postgres
- 可选：OpenAI-compatible 模型服务
- 可选：Tavily、Brave、SearchApi、SerpApi API Key
- 可选：Playwright Chromium，用于网页读取和截图
- 可选：Tesseract 或 PaddleOCR，用于 OCR

## 后端启动

建议从 `myagent/backend` 目录启动后端。后端静态目录和默认数据目录依赖当前工作目录，工作目录不正确时可能出现 `Directory 'data' does not exist`。

```powershell
cd F:\study\研究生\字节竞品分析Agent\myagent\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item ..\.env.example .env }
$env:PYTHONPATH=(Resolve-Path ..\..).Path
$env:BACKEND_CORS_ORIGINS="http://localhost:5174,http://127.0.0.1:5174"
python -m uvicorn myagent.backend.app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

如果 `8000` 被占用，可以换成 `8010`：

```powershell
python -m uvicorn myagent.backend.app.main:app --host 127.0.0.1 --port 8010
```

换端口后需要同步调整前端代理或 `VITE_API_BASE`。

## 前端启动

PowerShell 中推荐使用 `npm.cmd`，避免 `npm.ps1` 执行策略问题。

```powershell
cd F:\study\研究生\字节竞品分析Agent\myagent\frontend
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

默认访问：

```text
http://127.0.0.1:5174
```

生产构建：

```powershell
cd F:\study\研究生\字节竞品分析Agent\myagent\frontend
npm.cmd run build
```

## 环境变量

后端会读取 `myagent/backend/.env`。

常用变量：

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | OpenAI-compatible 模型 Key |
| `OPENAI_BASE_URL` | 自定义模型网关地址 |
| `OPENAI_MODEL` | 文本模型名称 |
| `OPENAI_VISION_MODEL` | 图片理解模型名称 |
| `OPENAI_TRUST_ENV` | 是否继承系统代理 |
| `TAVILY_API_KEY` | Tavily Search/Map/Crawl/Extract |
| `BRAVE_SEARCH_API_KEY` | Brave Search |
| `SEARCHAPI_API_KEY` | SearchApi Google/Baidu/DuckDuckGo |
| `SERPAPI_API_KEY` | SerpApi Google/Baidu |
| `DUCKDUCKGO_MAX_RESULTS` | DuckDuckGo 默认结果数 |
| `TAVILY_MAX_RESULTS` | Tavily 默认结果数 |
| `SEARCH_BROWSER_DISABLE_PROXY` | Playwright Chromium 是否禁用系统代理，默认建议 `true` |
| `STORAGE_BACKEND` | `sqlite`、`json`、`postgres` |
| `SQLITE_PATH` | SQLite 文件路径 |
| `JSON_STORAGE_DIR` | JSON 存储目录 |
| `POSTGRES_DSN` | Postgres 连接串 |
| `MAX_REVISION_LOOPS` | QA 最大返工轮次 |
| `TESSERACT_CMD` | Tesseract 可执行文件路径 |

没有配置 `OPENAI_API_KEY` 时，系统仍可走降级工作流，但报告质量会弱于接入真实模型后的结果。

## 搜索工具

`myagent/search` 可作为独立命令行工具使用，也可作为后端采集模块使用。

### 安装搜索依赖

基础搜索和网页读取：

```powershell
cd F:\study\研究生\字节竞品分析Agent
pip install ddgs duckduckgo-search playwright tavily-python openai
python -m playwright install chromium
```

Tesseract OCR，可选：

```powershell
pip install pytesseract Pillow
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

PaddleOCR，可选，中文截图通常更友好但依赖更重：

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddleocr
```

### 快速使用

从仓库根目录运行：

```powershell
cd F:\study\研究生\字节竞品分析Agent
$env:PYTHONPATH=(Resolve-Path .).Path
python -m myagent.search.search_cli --query "飞书 竞品分析"
```

只搜索，不打开结果页，不截图：

```powershell
python -m myagent.search.search_cli --query "飞书 竞品分析" --methods duckduckgo_text --max-results 5 --content none --screenshots none
```

保存结果页截图：

```powershell
python -m myagent.search.search_cli --query "飞书 竞品分析" --methods duckduckgo_text --max-results 3 --screenshots result_pages --full-page
```

开启 LLM 信息抽取和证据截图：

```powershell
python -m myagent.search.search_cli --query "飞书 竞品分析" --methods duckduckgo_text --max-results 3 --extract
```

开启 OCR 和图片理解：

```powershell
python -m myagent.search.search_cli --query "飞书 竞品分析" --methods duckduckgo_text --max-results 3 --extract --ocr --image-understanding
```

使用 PaddleOCR：

```powershell
python -m myagent.search.search_cli --query "飞书 竞品分析" --methods duckduckgo_text --max-results 3 --extract --ocr --ocr-engine paddle
```

从已有 extraction.json 重新定位证据原文并生成高亮截图：

```powershell
python -m myagent.search.search_cli --locate-extraction myagent/search/outputs/<timestamp>/extractions/extraction.json --locate-max-items 20
```

### 深挖模式

`--methods` 控制第一跳搜索源，`--deep-backend` 控制是否继续深挖结果页。

`--deep-backend` 可选值：

| 值 | 含义 | 适用场景 |
|---|---|---|
| `none` | 不深挖 | 成本最低 |
| `local_pagination` | 本地 Playwright 跟随分页链接 | 文章分页、列表翻页 |
| `local_relevant_links` | 本地 Playwright 跟随相关链接 | 从官网首页继续找价格、案例、功能页 |
| `local_all_links` | 本地同时跟随分页和相关链接 | 尽量覆盖本地可见链接 |
| `tavily` | 使用 Tavily Map/Crawl/Extract | 快速扩展站点 URL 和正文 |
| `local_relevant_links_tavily` | 先本地扩展，再 Tavily 深挖 | 覆盖最完整，耗时和 API 成本最高 |

本地深挖示例：

```powershell
python -m myagent.search.search_cli --query "飞书 定价 竞品" --methods duckduckgo_text --deep-backend local_relevant_links --link-scope same_domain --max-linked-pages-per-result 3
```

Tavily 深挖示例：

```powershell
python -m myagent.search.search_cli --query "飞书 竞品分析" --methods duckduckgo_text --deep-backend tavily --content none --screenshots none
```

从已知 URL 开始：

```powershell
python -m myagent.search.search_cli --query "Feishu pricing competitors" --start-urls "https://www.feishu.cn/" --extract
```

### 搜索参数速查

基础搜索：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--query` | 无 | 搜索内容，除 `--locate-extraction` 模式外必填 |
| `--methods` | `duckduckgo_text,duckduckgo_news,tavily` | 搜索源，支持 `duckduckgo_text`、`duckduckgo_news`、`tavily`、`brave_web`、`searchapi_google`、`searchapi_baidu`、`searchapi_duckduckgo`、`serpapi_google`、`serpapi_baidu`、`all` |
| `--max-results` | `5` | 每种搜索方法最大结果数 |
| `--start-urls` | 空 | 逗号分隔 URL，作为 seed 开始 |
| `--output-dir` | `myagent/search/outputs` | 输出根目录 |

正文和截图：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--content` | `result_pages` | 是否打开结果页读取正文，可选 `result_pages`、`none` |
| `--content-max-chars` | `12000` | 每页保存的最大正文字符数 |
| `--screenshots` | `both` | 可选 `search_page`、`result_pages`、`both`、`none` |
| `--full-page` | 关闭 | 保存整页长截图 |

LLM、chunk 和证据：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--extract` | 关闭 | 开启 LLM 竞品信息抽取 |
| `--chunk-mode` | `text` | 可选 `text`、`viewport` |
| `--chunk-size` | `4000` | text 模式 chunk 字符数 |
| `--chunk-overlap` | `400` | text 模式相邻 chunk 重叠字符数 |
| `--evidence-screenshots` | 开启 | 生成 chunkshot 和 quoteshot |
| `--no-evidence-screenshots` | 关闭项 | 禁用证据截图 |
| `--max-evidence-shots` | `20` | 最大证据截图数 |

OCR 和图片理解：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ocr` | 关闭 | 对 chunk 截图做 OCR |
| `--ocr-engine` | `tesseract` | 可选 `tesseract`、`paddle` |
| `--ocr-lang` | `chi_sim+eng` 或 `ch` | OCR 语言 |
| `--image-understanding` | 关闭 | 使用视觉模型总结截图中的非纯文本信息 |

### 搜索输出

每次运行会在 `myagent/search/outputs/<timestamp>/` 下生成：

```text
myagent/search/outputs/<timestamp>/
  results.json
  report.md
  contents/
  screenshots/
  chunk_shots/
  quote_shots/
  extractions/
    chunks.json
    extraction.json
```

主要文件：

- `results.json`：结构化运行结果，包含搜索方法、URL、标题、摘要、正文路径、截图路径、访问状态、错误和 Tavily 深挖统计。
- `report.md`：面向阅读的 Markdown 报告。
- `contents/`：结果页正文、Tavily 正文、chunk 文本和 DOM 文本。
- `screenshots/`：搜索页和结果页截图。
- `chunk_shots/`：chunk 级证据截图。
- `quote_shots/`：事实对应的精确证据截图。
- `extractions/chunks.json`：chunk 元数据、OCR/视觉补充和每个 chunk 的模型抽取结果。
- `extractions/extraction.json`：合并后的摘要、事实、实体、证据和错误列表。

## 后端运行产物

后端默认把数据放在 `myagent/backend/data`：

- `app.db`：SQLite 任务、事件流和报告。
- `logs/app.log`：应用日志。
- `search_runs/<task_id>/...`：搜索 Agent 每次搜索结果、chunk 和搜索报告。
- `task_runs/<task_id>/steps`：每个 workflow 节点的事件快照。
- `task_runs/<task_id>/llm_calls`：每次大模型调用的 request、output 或 error。
- `task_runs/<task_id>/checkpoint.json`：任务断点续跑状态。
- `task_runs/<task_id>/final/report.json` 和 `report.md`：最终报告。

查看某个任务输出文件：

```powershell
curl http://127.0.0.1:8000/api/tasks/<task_id>/outputs
```

## Demo

后端提供真实调用 demo：

```text
myagent/backend/demos/run_agent_demo.py
```

查看帮助：

```powershell
cd F:\study\研究生\字节竞品分析Agent\myagent\backend
$env:PYTHONPATH=(Resolve-Path ..\..).Path
.\.venv\Scripts\python.exe demos\run_agent_demo.py --help
```

运行 DuckDuckGo demo：

```powershell
$env:PYTHONPATH=(Resolve-Path ..\..).Path
.\.venv\Scripts\python.exe demos\run_agent_demo.py duckduckgo --query "飞书 竞品分析"
```

运行完整 workflow demo：

```powershell
$env:PYTHONPATH=(Resolve-Path ..\..).Path
.\.venv\Scripts\python.exe demos\run_agent_demo.py workflow --input demos\sample_task.json --duckduckgo-max-results 1 --max-revision-loops 1
```

这些 demo 会访问真实网络和外部站点，网络、DNS、搜索引擎响应或目标站点反爬都可能导致超时。离线验证请优先运行测试。

## 测试

后端测试：

```powershell
cd F:\study\研究生\字节竞品分析Agent\myagent\backend
$env:PYTHONPATH=(Resolve-Path ..\..).Path
.\.venv\Scripts\python.exe -m pytest -q
```

工作流测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_workflow.py -q
```

搜索工具测试：

```powershell
cd F:\study\研究生\字节竞品分析Agent
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest myagent\search\tests -q
```

## API 概览

主要接口：

- `POST /api/tasks`
- `POST /api/tasks/{task_id}/clarifications`
- `GET /api/tasks`
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/events`
- `GET /api/tasks/{task_id}/report`
- `GET /api/tasks/{task_id}/sources`
- `GET /api/tasks/{task_id}/outputs`
- 问卷设计与问卷分析相关接口，见后端 `main.py` 中的 survey 路由实现
- `GET /api/tasks/{task_id}/export?format=md|json|pdf|docx`
- `GET /health`

## 可选：Postgres

启动本地 Postgres：

```powershell
cd F:\study\研究生\字节竞品分析Agent\myagent
docker compose up -d postgres
```

环境变量：

```env
STORAGE_BACKEND=postgres
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/competitor_agent
```

## 常见问题

### No module named myagent

从仓库根目录或 `myagent/backend` 设置：

```powershell
$env:PYTHONPATH=(Resolve-Path ..\..).Path
```

如果在仓库根目录运行搜索工具：

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
```

### Directory 'data' does not exist

后端启动依赖 `myagent/backend` 作为工作目录。请从 `myagent/backend` 启动 uvicorn。

### PowerShell 禁止 npm

使用 `npm.cmd`：

```powershell
npm.cmd install
npm.cmd run dev
npm.cmd run build
```

### 搜索网页大量 ERR_PROXY_CONNECTION_FAILED

确认 `backend/.env` 中：

```env
SEARCH_BROWSER_DISABLE_PROXY=true
```

然后重启后端。

### 真实搜索或 demo 超时

这通常是外部网络、DNS、搜索引擎响应或目标网站访问限制导致。先用 pytest 验证本地逻辑，再在网络稳定环境中运行真实搜索。
