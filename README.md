# Kritai

A Krita plugin that sends your canvas through a local [mflux](https://github.com/filipstrand/mflux) image-to-image pipeline and displays the result.

This project is similar to the [Interstice](https://www.interstice.cloud/) plugin, but has different goals and is less ambitious in scope. Main priorities are:

- **Focus on macOS:** That's what I use and therefore that's what I care about.
- **Focus on local yet fast generation:** Easy on the wallet, more private, and better for the planet.
- **Focus on UX**: The plugin should be a joy to use, with a clean and intuitive interface, and minimal setup.

## Requirements

```bash
brew install krita uv hf
uv tool install --upgrade mflux
```

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
