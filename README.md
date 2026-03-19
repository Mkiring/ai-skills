# Seeed AI Skills

This repository collects and maintains AI skill libraries for Seeed products, providing developers with professional technical guidance and API references.

## 📚 Skills

- **[onnx-to-cvimodel](skills/onnx-to-cvimodel/README.md)** - ONNX to CVIMODEL conversion guide for YOLO models on Sophgo CV181x TPU, with ready-to-use scripts and tested configurations for YOLO11/YOLO26 (detect/pose/seg/cls)

- **[cv181x-media](skills/cv181x-media/README.md)** - Complete multimedia application development guide for reCamera with Sophgo CV181X/CV182X/CV180X chips, covering 15+ core modules including video input/output, encoding/decoding, and audio processing

- **[ee-datasheet-master](skills/ee-datasheet-master/README.md)** - Electronic component datasheet extraction with full source citations, supporting power management, MCU/SoC, sensors, and more with English/Chinese optimization *(requires: pdf skill)*

- **[schematic-analyzer](skills/schematic-analyzer/README.md)** - KiCad schematic analysis with accuracy-first principles, providing structured JSON output for components, nets, pages, and subsystems *(requires: pdf, ee-datasheet-master skills, pcbparts MCP)*

## 🚀 Installation

### Claude

```bash
# Install all skills
claude skills install git+https://github.com/Seeed-Studio/ai-skills

# Install individual skill
claude skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/onnx-to-cvimodel
claude skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/cv181x-media
claude skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/ee-datasheet-master
claude skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/schematic-analyzer
```

### Codex

```bash
# Install all skills
codex skills install git+https://github.com/Seeed-Studio/ai-skills

# Install individual skill
codex skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/onnx-to-cvimodel
codex skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/cv181x-media
codex skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/ee-datasheet-master
codex skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/schematic-analyzer
```

### Skill Dependencies

Some skills depend on others. Install in this order if installing individually:

1. **ee-datasheet-master** requires the built-in `pdf` skill
2. **schematic-analyzer** requires:
   - `pdf` and `ee-datasheet-master` skills
   - `pcbparts` MCP server (required for component lookup)

```bash
# Install schematic-analyzer with dependencies
claude skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/ee-datasheet-master
claude skills install git+https://github.com/Seeed-Studio/ai-skills#subdirectory=skills/schematic-analyzer
```

**pcbparts MCP Setup:**

The schematic-analyzer skill requires the pcbparts MCP server for component specifications lookup. Install and configure it before using schematic-analyzer:

```bash
# Install pcbparts MCP
# See: https://pcbparts.dev/mcp
```

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
