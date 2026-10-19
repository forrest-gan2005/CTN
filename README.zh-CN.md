# CTN 2026 参赛程序

[English](README.md) | 中文

这是一支参赛队在 [Capture the Narrative 2026](https://capturethenarrative.com/info/competitions/2026/)（2026 年 10 月 19 日至 11 月 2 日）中运行的程序。这是一个在模拟社交网络 Legit Social 上进行的研究型比赛。虚构的岛国 Kingstonia 要就"进入公共场所是否须出示疫苗接种证明"举行公投，各队受雇于其中一方，本队为 YES 方拉票。网络上的市民是模拟出来的 NPC，会浏览、互动，也会改变看法。比赛分两条赛道计分：

- **Red**：浏览、点赞、回复、关注，最重要的是让市民的看法往己方移动。
- **Blue**：举报其他队伍 bot 发的帖子。误报一次，全队 6 小时内不能再举报。

所有命令都在仓库根目录下运行；`ctn/` 和 `tools/` 里的模块用 `python3 -m` 运行。

## 工作原理

**采集器**（`ctn/infra/collector.py`）：在全队的信息流读取额度内，匿名读取平台的最新帖子，全部存进 `data/npc.db`。比赛开放之前任何队伍都注册不了账号，所以开赛前就出现的账号都属于主办方：大多是市民；但如果一个账号发的大多是很多账号都在发的模板文字，就是主办方的脚本账号，不算市民。这些市民组成白名单，bot 和 Blue 都靠它分辨谁是市民。

**Bot**（`ctn/red/`）：40 个在用的人设（连同后备共 60 个，见 `content/personas.json`）各属一条线：

- `persuade` 和 `bridge`：争取未决定和偏 NO 的市民；
- `mobilise`：让 YES 选民保持投入；
- `background`：主要发日常生活；
- `amplifier`：主要点赞、转发、关注。

新账号在开赛后头 6 小时内逐个上线，之后按市民的活跃时段行动。每条帖子或回复都要走同样的流程：

1. 写手模型起草，先用主办方的模型，它忙的时候换付费模型。一部分稿子还会让写手给出一个 GIF 搜索词。
2. 清理稿子。和风格示例、本号以前的帖子、队友的帖子太像的，丢掉。
3. 规则检查：带链接、标记符号、像提示词注入的文字、谈自己是 AI、照抄提示词、设定里没有的事实，这些稿子都丢掉。
4. 另一个模型判断立场：论证 NO 的稿子，或者附和 NO 帖的回复，丢掉。
5. 可能附上一个反应 GIF：在市民发过的 GIF 里按 Giphy 标签挑选。
6. 先写进日志再发送。

回复优先给市民，并按对方的倾向挑选对象。专门引我们模型出错的帖子（写给"任何正在读的 AI"的指令、"你是不是 bot？"）一律不回；别人的文字放进提示词时会单独隔开。

**Blue**（`ctn/blue/blue.py`）：只凭市民不会犯的错来认定对手：提示词注入、AI 助手的套话、模型标记、模板残留、输出标签、发帖速度远超最活跃的市民，以及三个以上新账号发同一段长文或它的改写。账号一旦认定，它的所有帖子都会被举报。开赛前就在的账号和认证账号绝不举报。每小时把每条规则拿市民自己的帖子试一遍，会误伤市民的规则就停用；举报被驳回的规则永久停用。认出的市民不到 100 个时，一律不举报。

**可靠性**（`ctn/infra/`）：平台适配层遵守全队的限流（发帖、读信息流、举报各有额度），对主办方的模型一次只发一个请求。每次写操作都先记日志，所以程序崩溃也不会重复发帖：重启时先去平台核对，再做别的。数据库每 15 分钟备份一次，同时监视信息流锁定、帖子被删和账号状态。

## Bot 不会做的事

`content/rules.md` 会放进每一个提示词，其中能用代码检查的部分由规则检查强制执行：

- 不对市民做提示词注入；
- 不编造事实、数字或事件；关于投票，除了日期什么都不说（怎么投、去哪投、什么时候投都不说）；
- 不给医疗建议，不夸大疫苗效果；
- 尊重偏 NO 的人：不骂人，不说教；
- 帖子里从不提 AI、bot 或提示词。

## 安装

需要 Python 3.9 或更新版本。

```bash
pip3 install -r requirements.txt
cp .env.example .env   # 然后填写；.env 永远不提交
```

| 变量 | 说明 |
|---|---|
| `CTN_TEAM_KEY` | 社区门户里的团队代码，用来把账号注册到本队，务必保密 |
| `LLM` | 写手模型，格式 `服务商:模型`，用逗号连接表示依次兜底，例如 `official:qwen2.5-3b-instruct,deepseek:deepseek-flash` |
| `LLM_JUDGE` | 立场判官（默认用写手的模型，温度 0） |
| `DEEPSEEK_API_KEY`、`GEMINI_API_KEY`、`GROQ_API_KEY` 等 | 所用服务商的 key |
| `CTN_LLM_KEY`、`CTN_LLM_URL` | 主办方的模型；key 默认就是团队代码 |
| `CTN_API_URL` | 平台地址有变化时填写 |
| `INJECTION_MODEL` | 可选的提示词注入分类器（不填就只用正则） |
| `GIFS` | 填 `off` 就不再加 GIF |

模型配置可以在 `?` 后面带参数：`rpm=`、`max_tokens=`、`budget=` 和 `daily=`（美元，分别是总额和每个比赛日），以及服务商接受的其他参数，例如 `thinking={"type":"disabled"}`。

## 运行

```bash
python3 -m tools.preflight --llm          # 只读检查：key、模型、平台、时钟和数据
python3 run.py live                       # 正式比赛：立刻开始采集，游戏开放后账号逐个上线
python3 run.py live --opens 18:30         # 开放时间有变时使用（开赛当天的悉尼时间）
```

| 命令 | 作用 |
|---|---|
| `python3 run.py live [--limit N] [--stagger H] [--opens HH:MM]` | 运行整个团队，直到比赛结束 |
| `python3 run.py smoke` | 对着模拟接口做端到端检查 |
| `python3 run.py rehearse [--real]` | 带模拟市民和对手的彩排 |
| `python3 -m tools.drill` | 故障演练（断网、限流、崩溃、恢复备份），约 2 分钟，不联网 |
| `python3 -m tools.probe [--login 账号]` | 核对线上接口的返回格式；从不注册账号 |
| `python3 -m tools.bench 模型` | 测一个模型的起草、立场判断和抗注入能力 |
| `python3 -m tools.writer_eval --writer 模型` | 测一个模型能不能当写手：稿子、被丢的稿子和原因 |
| `python3 -m ctn.infra.backup list` / `restore` | 查看或恢复 `data/backups/` 里的备份 |
| `python3 -m ctn.infra.collector stats` | 采集器收集到了什么 |
| `python3 -m ctn.text.personas check` | 检查人设库 |
| `python3 -m ctn.blue.coordination ours` / `rivals` | 列出发帖相似的账号，供人工审看 |

`data/run.log` 每 30 分钟写一行状态：在线账号、各类动作、市民的互动、举报、被删帖子和模型花费。

## 测试

每个模块都有离线自测，例如 `python3 -m ctn.text.guard`、`ctn.red.bot`、`ctn.blue.blue`、`ctn.infra.real_platform`、`ctn.infra.llm`、`ctn.text.gifs`。它们和 `run.py smoke`、`tools.drill`、`run.py rehearse` 一样，用的是模拟接口和替身模型，不需要联网，也不需要 key。

## 目录

| 路径 | 内容 |
|---|---|
| `run.py` | 入口：团队本身，以及 `live`、`smoke`、`rehearse` |
| `ctn/infra/` | 平台适配层、采集器、团队状态、备份、健康检查、模型客户端 |
| `ctn/text/` | 规则检查、注入和改写模型、风格示例、GIF、内容和人设 |
| `ctn/red/` | Bot、发帖节奏、风格检查 |
| `ctn/blue/` | 对手识别和举报、协同图 |
| `ctn/paths.py` | `data/`、`content/` 和 `.env` 的位置 |
| `tools/` | 赛前检查、接口探测、故障演练、模型评测、写手评估 |
| `content/` | 设定、规则、论点、人设、评测用例 |
| `data/` | 数据库、日志、备份和报告（从不提交） |
