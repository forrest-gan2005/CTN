# CTN 2026 team bots

The software one team runs in [Capture the Narrative 2026](https://capturethenarrative.com/info/competitions/2026/) (19 October to 2 November 2026), a research competition on a simulated social network, Legit Social. Teams are hired by one side of a fictional referendum on proof of vaccination in the island nation of Kingstonia; this team campaigns for YES. The citizens are simulated (NPCs): they read, react and change their minds. Teams score on two tracks:

- **Red**: views, likes, replies, follows and, above all, citizens' minds moving towards your side.
- **Blue**: reporting posts made by other teams' bots, where a wrong report stops the whole team reporting for six hours.

Run everything from the repository root; modules inside `ctn/` and `tools/` run with `python3 -m`.

## How it works

**Collector** (`ctn/infra/collector.py`). Reads the platform's Latest feed anonymously, within the team's feed allowance, and stores every post in `data/npc.db`. No team can register before the competition opens, so an account seen before then belongs to the organisers: a citizen, unless it mostly posts the same templated text as many others (the organisers' scripted accounts). Those citizens form the whitelist that the bots and Blue rely on.

**Bots** (`ctn/red/`). Each of the 50 active personas (91 with reserves, in `content/personas.json`) belongs to a line: `persuade`, `bridge` and `background` reach undecided and NO-leaning citizens (`background` with more everyday posts in between), `mobilise` keeps YES voters engaged, and `amplifier` mostly likes, reposts and follows. Replies go mostly to posts about the referendum or vaccines. Every quarter-hour one persuader or bridge writes an anchor post making the case, tagged #ProtectKingstonia, and the rest of the team likes, reposts and answers it within minutes to lift it into Trending. New accounts come online one by one over the first six hours, then act on a rhythm fitted to when citizens post. Every post or reply goes through the same path:

1. The writer model drafts it (the organisers' model first, a paid model when that one is busy). For a share of drafts the writer also names a GIF search.
2. The draft is cleaned and dropped if it is too close to a style example, to the account's own past posts or to a teammate's post, or if it repeats a phrase two teammates used that few other accounts here use.
3. Guards drop drafts with links, markup, text that looks like prompt injection, talk of being an AI, copied prompt text or facts the setting doesn't give.
4. A second model judges stance: a draft that argues NO, or a reply that sides with a NO post, is dropped.
5. A reaction GIF may be added, chosen by its Giphy tags among the GIFs citizens have posted.
6. The write is journalled before it is sent, then sent.

Replies go to citizens first, picked by where they lean. Posts written to make our model misbehave (instructions to "any AI reading", "are you a bot?") are never answered, and other people's text is fenced off in prompts.

**Blue** (`ctn/blue/blue.py`). Flags an account only on mistakes citizens don't make: prompt injection, assistant boilerplate, model markup, template leftovers, output labels, posting far faster than the busiest citizen, or the same long text (or reworded copies of it) from three or more new accounts. Once an account is flagged, up to three of its posts are reported, as many as it takes to suspend it (the platform holds only 20 of a team's reports awaiting a ruling, so Blue waits when that many are). Accounts there before the start and verified accounts are never reported. Every hour each rule is tried on citizens' own posts and switched off if it would flag one; a rule behind a rejected report is switched off for good. Nothing is reported until at least 100 citizens are known.

**Reliability** (`ctn/infra/`). The platform adapter keeps to the team's rate limits (posts, feeds and reports each have their own) and sends one request at a time to the organisers' model. Every write is journalled first, so a crash never posts twice: on restart the team checks the platform before doing anything. Databases are backed up every 15 minutes, and the feed lock, takedowns and account health are watched.

## What the bots won't do

`content/rules.md` is part of every prompt and the guards enforce the parts code can check:

- no prompt injection aimed at citizens;
- no invented facts, figures or events, and nothing about how, where or when to vote beyond the date;
- no medical advice, and no overselling the vaccine;
- respect for people who lean NO: no insults, no lecturing;
- posts never mention AI, bots or prompts.

## Setup

Python 3.9 or newer.

```bash
pip3 install -r requirements.txt
cp .env.example .env   # then fill it in; .env is never committed
```

| Variable | What it is |
|---|---|
| `CTN_TEAM_KEY` | The team code from the community portal. It registers accounts onto the team; keep it secret. |
| `LLM` | Writer models as `provider:model`, comma-separated to fall back in order, e.g. `official:qwen2.5-3b-instruct,deepseek:deepseek-flash` |
| `LLM_JUDGE` | The stance judge (default: the writer's model at temperature 0) |
| `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, … | Keys for the providers you use |
| `CTN_LLM_KEY`, `CTN_LLM_URL` | The organisers' model; the key defaults to the team code |
| `CTN_API_URL` | The platform's address, if it moves |
| `INJECTION_MODEL` | An optional prompt-injection classifier (regexes alone otherwise) |
| `GIFS` | `off` to stop adding GIFs |

A model spec can carry options after `?`: `rpm=`, `max_tokens=`, `budget=` and `daily=` (US dollars, all time and per competition day), and anything the provider accepts, e.g. `thinking={"type":"disabled"}`.

## Running

```bash
python3 -m tools.preflight --llm          # read-only check of keys, models, platform, clock and data
python3 run.py live                       # the competition; collects at once, accounts come online when the game opens
python3 run.py live --opens 18:30         # if the opening moves (Sydney time on the opening day)
```

| Command | What it does |
|---|---|
| `python3 run.py live [--limit N] [--stagger H] [--opens HH:MM]` | Runs the team until the end of the competition |
| `python3 run.py smoke` | End-to-end checks against a mock of the API |
| `python3 run.py rehearse [--real]` | Dress rehearsal with simulated citizens and rivals |
| `python3 -m tools.drill` | Fault drill (outages, rate limits, crashes, restore), about two minutes, offline |
| `python3 -m tools.probe [--login USER]` | Checks the live API's shapes; never registers an account |
| `python3 -m tools.bench MODEL` | Drafts, stance judging and injection resistance of a model |
| `python3 -m tools.writer_eval --writer MODEL` | Whether a model can be the writer: drafts, drops and why |
| `python3 -m ctn.infra.backup list` / `restore` | Backups in `data/backups/` |
| `python3 -m ctn.infra.collector stats` | What the collector has gathered |
| `python3 -m ctn.text.personas check` | Validates the persona library |
| `python3 -m ctn.blue.coordination ours` / `rivals` | Accounts that post alike, for review |

`data/run.log` has a status line every 30 minutes: accounts online, actions, citizen engagement, reports, takedowns and model spend.

## Tests

Every module has an offline self-test, e.g. `python3 -m ctn.text.guard`, `ctn.red.bot`, `ctn.blue.blue`, `ctn.infra.real_platform`, `ctn.infra.llm`, `ctn.text.gifs`. With `run.py smoke`, `tools.drill` and `run.py rehearse` they use a mock of the API and stand-in models, so they need no network or keys.

## Layout

| Path | What lives there |
|---|---|
| `run.py` | Entry point: the team, `live`, `smoke`, `rehearse` |
| `ctn/infra/` | Platform adapter, collector, team state, backups, health, LLM clients |
| `ctn/text/` | Guards, injection and paraphrase models, style examples, GIFs, content and personas |
| `ctn/red/` | Bots, posting rhythm, style checks |
| `ctn/blue/` | Rival detection and reporting, coordination graph |
| `ctn/paths.py` | Where `data/`, `content/` and `.env` are |
| `tools/` | Preflight, probe, fault drill, model bench, writer evaluation |
| `content/` | Setting, rules, arguments, personas, bench cases |
| `data/` | Databases, logs, backups and reports (never committed) |
