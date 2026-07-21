# Kritai

A Krita plugin that sends your canvas through a local [mflux](https://github.com/filipstrand/mflux) image-to-image pipeline and displays the result.

This project is similar to the [Interstice](https://www.interstice.cloud/) plugin, but has different goals and is less ambitious in scope. Main focuses are:

- **macOS:** That's what I use and therefore that's what I care about.
- **Local yet fast generation:** Private, and better for the planet.
- **Permissive license:** Models and output can be used in any project.
- **UX**: The plugin should be a joy to use, with a clean and intuitive interface, and minimal setup.

## Requirements

```bash
brew install krita uv hf
uv tool install --upgrade mflux
```

The `mflux` install is optional up front: the first time you run a local action without it, Kritai offers to install it for you (via `uv`, bootstrapping `uv` through Homebrew if needed). The **Mask** tab (background removal) uses [rembg](https://github.com/danielgatis/rembg) and installs it the same way on first use.

> [!tip]
> Some HuggingFace models are gated and require logging in with an access token. Run `hf auth login` before downloading them.

## Installation

```bash
git clone https://github.com/yourname/kritai.git
cd kritai
bash install.sh
```

After running the script:

1. Open Krita
2. Go to **Settings → Configure Krita → Python Plugin Manager**
3. Enable **Kritai**
4. Restart Krita
5. Open the docker via **Settings → Dockers → Kritai**
