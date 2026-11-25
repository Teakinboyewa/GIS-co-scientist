# GIS Co-Scientist

<img src="WebUI/icon/GIS-Coscientist_logo.png" alt="GIS Co-Scientist logo" width="110" align="right">

GIS Co-Scientist is a multi-agent framework for autonomous geospatial scientific inquiry. It takes a research question
from data to a drafted manuscript, and pauses so you can review and edit its work along the way.

This repository holds the source code of the web application. To use GIS Co-Scientist without setting up the code:

- **Desktop app (recommended):** [download for Windows or macOS](https://autonomousgis-agsi.github.io/AGSI/desktop-install.html)
- **WebUI:** [www.gis-coscientist.online](https://www.gis-coscientist.online/)

[Project page](https://autonomousgis-agsi.github.io/AGSI/) ·
[User manual](https://autonomousgis-agsi.github.io/AGSI/user_manual) ·
[Paper](https://doi.org/10.1080/10095020.2026.2764024) ·
[Video demos](https://youtube.com/playlist?list=PL6ON3gdLloDE_aLyFjj4X7EYKDDdAfJi2)

## What it does

| Mode | What it does |
|---|---|
| **Data Retrieval** | Finds and downloads geospatial data from a request in plain language |
| **Spatial Analysis** | Plans a geoprocessing workflow, writes the code, and runs it |
| **Research** | Runs the full workflow, from question to drafted manuscript |
| **Chat** | Answers GIS questions |

Two versions of Research mode are included for comparison: **Research (no planning)** and **Single Agent**.

## Run from source

You need Python 3.13 (the version it is developed with) and an [OpenAI API key](https://platform.openai.com/api-keys).

```
git clone https://github.com/Teakinboyewa/GIS-co-scientist.git
cd GIS-co-scientist
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python WebUI/app.py
```

Open http://127.0.0.1:4041/ and add your key in **Settings**. A [GIBD key](https://www.gibd.online) also works, but only
as a backup for trying out the system.

> [!WARNING]
> The agents write Python code and run it on your computer, with your permissions. Run the app only where you accept that.

### Add data sources

The Data Retrieval agent follows *handbooks*, one per data source, and this repository does not include any. To add the
shared ones, get the [data source catalog](https://github.com/Teakinboyewa/GIS-co-scientist-catalog), copy its files
into these folders (create them if needed), and restart the app:

| From the catalog | To |
|---|---|
| `handbooks/*.toml` | `agents/data_agent/DataRetriever_Handbooks/Handbooks/` |
| `keys/*.keys` | `agents/data_agent/DataRetriever_Handbooks/Keys/` |

The key files hold only key names. Enter your own keys in **Settings → Data Source Keys**. You can also add your own
sources in the app, under **Data Sources → My Sources**.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AGM_HOST` | `127.0.0.1` | Address to listen on. `0.0.0.0` lets other computers connect. |
| `AGM_PORT` | `4041` | Port |
| `AGM_DEBUG` | off | `1` turns on Flask's debugger. Never use it on a shared network. |

The model and reasoning effort of each language-model call are set in [`utils/llm_models.toml`](utils/llm_models.toml).
Restart the app after editing it.

## Repository layout

| Path | Contents |
|---|---|
| `WebUI/` | Flask server and web page |
| `agents/` | The agents: question understanding, data, planning and execution, modeling, result analysis, reporting |
| `agents/spatial_reasoning/`, `spatial_kb/` | Spatial reasoning method and its knowledge base |
| `skills/` | Analysis guides the agents use |
| `utils/` | Shared helpers and model settings |
| `docs/` | [How a Research mode run works](docs/RESEARCH_MODE_WORKFLOW.md) |
| `tests/` | Tests |

The map interpreter page needs a `mapper/` folder that is not part of this repository.

## Tests

Run one test module:

```
python -m unittest tests.test_spatial_conditions
```

## Related repositories

| Repository | Contents |
|---|---|
| [GIS-co-scientist-desktop](https://github.com/Teakinboyewa/GIS-co-scientist-desktop) | Desktop app |
| [GIS-co-scientist-catalog](https://github.com/Teakinboyewa/GIS-co-scientist-catalog) | Shared data sources |
| [AGSI](https://github.com/AutonomousGIS-AGSI/AGSI) | Project page, user manual, and case studies |

## Citation

> Akinboyewa, T., & Li, Z. (2026). GIS Co-Scientist: a human-AI collaborative framework for autonomous geospatial
> scientific inquiry. *Geo-spatial Information Science*, 1–22. https://doi.org/10.1080/10095020.2026.2764024

```bibtex
@article{akinboyewa2026giscoscientist,
  title   = {{GIS Co-Scientist}: a human-{AI} collaborative framework for autonomous geospatial scientific inquiry},
  author  = {Akinboyewa, Temitope and Li, Zhenlong},
  journal = {Geo-spatial Information Science},
  year    = {2026},
  pages   = {1--22},
  doi     = {10.1080/10095020.2026.2764024}
}
```

## License

[GPL-3.0](LICENSE)
