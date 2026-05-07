# FinAI

A 股每日分析工具的 MVP：每个交易日收盘后产出一份**有数据出处、可点击下钻**的市场日报。详细设计见 [function_desin.md](function_desin.md)。

## 架构

```
Data → Signal → LLM → Report
AkShare    指标/异动      Claude        HTML + JSON
SQLite     板块轮动      结构化输出     ECharts 仪表盘
```

四层职责严格分离：LLM 不做数值计算，所有数字来自信号层；信号层只读不写；数据层只爬取不解读。

## 快速开始

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY

# 3. 用 mock 数据跑一遍（不需要网络/API key）
finai run --source mock --trade-date 2026-05-06
# → reports/daily_2026-05-06.html  (打开看效果)

# 4. 用真实 AkShare 数据
finai run --source akshare

# 5. 启动 Web 服务
finai serve --port 8000
# → http://127.0.0.1:8000/reports/latest

# 6. 启用每日 16:00 调度
finai schedule
```

## CLI 命令

| 命令 | 说明 |
| --- | --- |
| `finai init-db` | 创建 SQLite 表 |
| `finai etl` | 仅抓取并入库，不生成报告 |
| `finai run` | 完整流程：抓取 → 信号 → LLM → 渲染 |
| `finai serve` | 启动 FastAPI（`/healthz`, `/reports`, `/reports/latest`, `/reports/{date}`, `POST /pipeline/run`） |
| `finai schedule` | 后台调度器，工作日 16:00 Asia/Shanghai 自动跑 |

## 模块导航

| 路径 | 职责 |
| --- | --- |
| [finai/data/](finai/data/) | 数据源（akshare / mock）+ 统一 `MarketSnapshot` 契约 |
| [finai/signals/](finai/signals/) | 市场总览、异动检测、板块轮动、历史相似度 |
| [finai/llm/](finai/llm/) | Claude 客户端（带 prompt caching）、归因、叙事生成 |
| [finai/report/](finai/report/) | 报告组装 + Jinja HTML 模板 + ECharts 渲染 |
| [finai/pipeline/](finai/pipeline/) | ETL 编排 + APScheduler 调度 |
| [finai/api/](finai/api/) | FastAPI 服务 |

## 关键设计决策

1. **LLM 强约束**：归因输出走 Pydantic schema（`finai/llm/attribution.py`），任何 LLM 返回的非法字段会抛错；任何提到的新闻必须来自传入的列表。
2. **Prompt caching**：系统提示词被冻结在 [finai/llm/prompts.py](finai/llm/prompts.py)，按 prefix 命中缓存；批量归因时复用同一份 system。
3. **降级路径**：LLM 不可用时（无 API key 或超时）自动退回模板化输出，pipeline 不会中断，报告右上角会显示"部分内容由模板降级生成"。
4. **历史相似度**：基于板块涨跌幅向量的余弦相似度（[finai/signals/similarity.py](finai/signals/similarity.py)），需要至少 60 个交易日的入库历史才能生效。
5. **零配置启动**：默认 SQLite + mock 数据源，`pip install -e .` 之后立刻可跑。

## 测试

```bash
pytest
# 10 passed
```

测试覆盖信号层、LLM 降级路径、端到端管道。LLM 在测试环境强制关闭（`FINAI_LLM_ENABLED=false`），不会真实调用 Claude API。

## 已知限制（v0.1）

- AkShare 接口偶有变动；遇到 column 改名时数据层会日志告警并返回空 frame
- 龙虎榜、连板梯队历史只是 v0 占位（仅当日数据），需要积累几个交易日的历史后才有意义
- 邮件/微信推送、个股深挖页、回测验证未实现（v2 路线图）
- 真正使用 Claude 时建议预算 ~$1/日（启用了 prompt caching）
