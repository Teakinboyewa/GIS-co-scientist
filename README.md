# GIS Co-Scientist

<img src="WebUI/icon/GIS-Coscientist_logo.png" alt="GIS Co-Scientist logo" width="110" align="right">

GIS Co-Scientist is a multi-agent framework for autonomous geospatial scientific inquiry. It takes a geospatial research question from data to a drafted manuscript through a single conversational interface, and pauses along the way so you can review and edit its work.

Specialized agents understand the question, retrieve geospatial data, plan the analysis, write and run the code, interpret the results, and draft the report. For more details on the framework, refer to the published article:

Temitope Akinboyewa and Zhenlong Li. 2026. *"GIS Co-Scientist: a human-AI collaborative framework for autonomous geospatial scientific inquiry."* Geo-spatial Information Science. https://doi.org/10.1080/10095020.2026.2764024

> [!IMPORTANT]
> **Case studies:** The research questions, conversation records, workflows, figures, and manuscripts of the case studies are available on the [project page](https://autonomousgis-agsi.github.io/AGSI/#case-studies).

Project page: https://autonomousgis-agsi.github.io/AGSI/

![GIS Co-Scientist framework: a human in the loop guides a layer of specialized agents, backed by a reproducibility layer](docs/images/framework.jpg)

# Working modes

- **Data Retrieval:** finds and downloads geospatial data from a request in plain language.
- **Spatial Analysis:** plans a geoprocessing workflow, writes the code, and runs it.
- **Research:** runs the full workflow, from research question to drafted manuscript.
- **Chat:** answers GIS questions.

# Getting started

## Desktop app (recommended)

- [Download the installer](https://autonomousgis-agsi.github.io/AGSI/desktop-install.html) for Windows or macOS and follow the steps on the page.
- Open the app and add your OpenAI API key in `Settings`.

## WebUI

- Open [www.gis-coscientist.online](https://www.gis-coscientist.online/) in your browser.
- Add your OpenAI API key in `Settings`.

## API key

Use your own [OpenAI API key](https://platform.openai.com/api-keys). A [GIBD key](https://www.gibd.online) is only a backup if you want to test or try out the system.

# User Manual

The User Manual is available [here](https://autonomousgis-agsi.github.io/AGSI/user_manual).

# Demonstration

### Introduction
<a href="https://www.youtube.com/watch?v=oyf4I1VRlLU" target="_blank">
  <img src="https://img.youtube.com/vi/oyf4I1VRlLU/0.jpg" alt="Video: Introduction to GIS Co-Scientist" width="800">
</a>

### Data Retrieval mode
<a href="https://www.youtube.com/watch?v=TltGDALyw88" target="_blank">
  <img src="https://img.youtube.com/vi/TltGDALyw88/0.jpg" alt="Video: Data Retrieval mode, Example 1" width="800">
</a>

### Spatial Analysis mode
<a href="https://www.youtube.com/watch?v=Ue53TqW_Su0" target="_blank">
  <img src="https://img.youtube.com/vi/Ue53TqW_Su0/0.jpg" alt="Video: Spatial Analysis mode, Example 1" width="800">
</a>

### Research mode
<a href="https://www.youtube.com/watch?v=Ib5p8sdtxGk" target="_blank">
  <img src="https://img.youtube.com/vi/Ib5p8sdtxGk/0.jpg" alt="Video: Research mode, Example 1" width="800">
</a>

Find more examples in the [video playlist](https://youtube.com/playlist?list=PL6ON3gdLloDE_aLyFjj4X7EYKDDdAfJi2) and on the [project page](https://autonomousgis-agsi.github.io/AGSI/#demos).

# Citation

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

# License

[GPL-3.0](LICENSE)
