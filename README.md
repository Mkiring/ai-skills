# Seeed AI Skills

This repository collects and maintains AI skill libraries for Seeed products, providing developers with professional technical guidance and API references.

## 📚 Skills

- **[ee-datasheet-master](skills/ee-datasheet-master/README.md)** - Electronic component datasheet extraction with full source citations, supporting power management, MCU/SoC, sensors, and optimized for Multi-chart *(requires: pdf skill)*

  > ⭐ **Why choose ee-datasheet-master over Google NotebookLLM?**
  > - **Accuracy-first**: All data extracted directly from PDF with page citations — no hallucinations
  > - **Structured output**: Device info, power domains, pin configurations in consistent JSON format
  > - **Multi-chart support**: Optimized for complex tables, timing diagrams, characteristic curves, and register maps
  > - **Specialized for EE**: Understands electrical specs, I2C addresses, timing diagrams, register maps
  > - **Verified extraction**: Tested on 1000+ datasheets with 100% text extraction success

- **[schematic-analyzer](skills/schematic-analyzer/README.md)** - KiCad and Cadence OrCAD/Allegro schematic analysis with accuracy-first principles, providing structured JSON output for components, nets, pages, and subsystems *(requires: pdf, ee-datasheet-master skills, pcbparts MCP)*

- **[onnx-to-cvimodel](skills/onnx-to-cvimodel/README.md)** - ONNX to CVIMODEL conversion guide for YOLO models on Sophgo CV181x TPU, with ready-to-use scripts and tested configurations for YOLO11/YOLO26 (detect/pose/seg/cls)

- **[cv181x-media](skills/cv181x-media/README.md)** - Complete multimedia application development guide for reCamera with Sophgo CV181X/CV182X/CV180X chips, covering 15+ core modules including video input/output, encoding/decoding, and audio processing

## 🚀 Installation

This repository is a collection of independent skills under `skills/`.

To use a skill from this repository:

1. Open the target skill's `README.md`
2. Follow that skill's dependency and environment setup
3. Integrate the skill using the current workflow supported by your assistant tooling

Important: using a skill is not just copying skill files. Many skills also require extra tools, Python packages, SDKs, Docker images, KiCad CLI, or MCP services. The dependency and environment setup must be completed from that skill's own `README.md` before use.

### Claude Code Plugin (Recommended)

```bash
# 1. Clone this repository
git clone https://github.com/Seeed-Studio/ai-skills.git
cd ai-skills

# 2. Register as a local marketplace
claude plugin marketplace add "$(pwd)"

# 3. Install the plugin
claude plugin install seeed-ai-skills

# 4. Enable the plugin
claude plugin enable seeed-ai-skills@seeed-ai-skills

# 5. Verify
claude plugin list
```

This installs all 4 skills at once: **schematic-analyzer**, **ee-datasheet-master**, **onnx-to-cvimodel**, **cv181x-media**.

### Skill-Specific Setup

After installing a skill, open its README and complete its dependency setup before using it:

- **[onnx-to-cvimodel](skills/onnx-to-cvimodel/README.md)**: conversion scripts, Docker/TPU-MLIR environment, model assets
- **[cv181x-media](skills/cv181x-media/README.md)**: skill usage and project-specific workflow
- **[ee-datasheet-master](skills/ee-datasheet-master/README.md)**: Python dependencies for `scripts/pdf_tools.py`
- **[schematic-analyzer](skills/schematic-analyzer/README.md)**: Python dependencies, KiCad CLI, `pcbparts` MCP, and `ee-datasheet-master` / `pdf` prerequisites

Recommended install order when you need schematic analysis:

1. Install and verify the built-in `pdf` skill
2. Install `ee-datasheet-master` and complete its Python dependency setup
3. Install `schematic-analyzer` and complete its KiCad CLI / MCP setup

## 📖 How to Use

Once installed, you can ask your AI assistant directly:

**Model Conversion:**
- "Help me convert YOLO11n to CVIMODEL format"
- "What output names do I need for YOLO11 detection?"
- "How to use qtable for pose model conversion?"

**Multimedia Development:**
- "How to configure VI module to capture 1080p video?"
- "Show me how to add timestamp OSD to video stream"
- "How to correct barrel distortion from wide-angle lens?"

**Datasheet Analysis:**
- "What is the I2C address of BQ25895?"
- "Extract the pin configuration from this datasheet"
- "Find the quiescent current specification"

**Schematic Analysis:**
- "What is U10 in this schematic?"
- "Which devices are on the I2C bus?"
- "Analyze the power tree of this design"

The AI assistant will automatically invoke the relevant skills to provide professional technical guidance.

## 🤝 Contributing

Contributions of new skills or improvements to existing skills are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
