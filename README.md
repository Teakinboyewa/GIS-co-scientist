# GIS Co-Scientist

An agent that takes a geospatial research question from data to a drafted manuscript, with checkpoints where
you review and edit its plan. See [the Research mode workflow](docs/RESEARCH_MODE_WORKFLOW.md) for how a run works.

## Run it

Developed with Python 3.13.

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python WebUI/app.py
```

Open http://127.0.0.1:4041/ and add your API key in Settings.

| Variable | Default | Meaning |
|---|---|---|
| `AGM_HOST` | `127.0.0.1` | Address to listen on. `0.0.0.0` lets other computers connect. |
| `AGM_PORT` | `4041` | Port |
| `AGM_DEBUG` | off | `1` turns on Flask's debugger. Never use it on a shared network. |

The agents write and run Python code on your computer. Run it only where you accept that.

## Layout

| Path | What it is |
|---|---|
| `WebUI/` | Flask server and web page |
| `agents/` | The agents |
| `agents/spatial_reasoning/`, `spatial_kb/` | Spatial reasoning method and knowledge base |
| `skills/` | Analysis guides used by the agents |
| `tests/` | Tests (`python -m unittest tests.test_spatial_conditions`) |

The map interpreter needs a `mapper/` folder that is not part of this repository.

## License

[GPL-3.0](LICENSE)
