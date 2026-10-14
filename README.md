# CTN 2026 team bots

Run everything from the repository root; modules inside `ctn/` and `tools/` run with `python3 -m`.

| Path | What lives there |
|---|---|
| `run.py` | Entry point: `live`, `smoke`, `rehearse` |
| `ctn/infra/` | Platform adapters, collector, team state, backups, health, LLM clients |
| `ctn/text/` | Text guards, injection and paraphrase models, examples, content and personas |
| `ctn/red/` | Bots, posting rhythm, style checks |
| `ctn/blue/` | Rival detection and reporting, coordination graph |
| `ctn/paths.py` | Where `data/`, `content/` and `.env` are |
| `tools/` | Offline tools: fault drill, model bench |
| `content/` | Setting, rules, arguments, personas, bench cases |
| `data/` | Databases, logs and reports (not committed) |

```bash
python3 run.py smoke            # end-to-end checks against a mock API
python3 -m tools.drill          # fault drill, about two minutes, offline
python3 -m ctn.text.guard       # any module's self-test
python3 -m ctn.infra.backup list
```
