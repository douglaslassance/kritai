# Kritai

A Krita docker that sends your canvas through a local [mflux](https://github.com/filipstrand/mflux) image-to-image pipeline and shows the result directly in the panel.

## Requirements

- **Krita 5.x** with Python plugin support enabled
- **uv** — install via Homebrew:
  ```bash
  brew install uv
  ```
- **mflux** — install via uv:
  ```bash
  uv tool install mflux
  ```
  The plugin expects mflux binaries at `~/.local/bin/` (the default uv tool install location).
- **Hugging Face CLI** — required to download gated models (e.g. FLUX.1-Kontext-dev). Install via Homebrew and log in:
  ```bash
  brew install huggingface-cli
  hf auth login
  ```

## Installation

```bash
git clone https://github.com/yourname/kritai.git
cd kritai
bash install.sh
```

On **Windows**, run the terminal as Administrator (or enable Developer Mode) before running `install.sh`, as directory symlinks require elevated privileges.

After running the script:

1. Open Krita
2. Go to **Settings → Configure Krita → Python Plugin Manager**
3. Enable **Kritai**
4. Restart Krita
5. Open the docker via **Settings → Dockers → Kritai**

## Usage

1. Open or paint on a canvas
2. Type a prompt describing what you want
3. Adjust settings (model, steps, guidance, strength, seed) as needed
4. Click **Generate** — the result appears in the docker preview
5. Enable **Auto** to regenerate automatically whenever you modify the canvas (with a short debounce delay)

## Settings

| Setting  | Description                                                                           |
| -------- | ------------------------------------------------------------------------------------- |
| Model    | mflux model name (`dev`, `schnell`, or a HuggingFace path)                            |
| Quantize | Bit-depth quantization (4 is a good default; 0 = none)                                |
| Steps    | Inference steps (more = higher quality, slower)                                       |
| Guidance | How closely to follow the prompt                                                      |
| Strength | How much the init image influences the result (0 = ignore canvas, 1 = full influence) |
| Seed     | Fixed seed for reproducible results; check Random for a new result each time          |
