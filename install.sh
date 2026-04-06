#!/usr/bin/env bash
# Kritai install script — creates symlinks so Krita picks up the plugin.
# Run once from the repo root: bash install.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Locate Krita's pykrita directory
case "$(uname -s)" in
  Darwin)
    PYKRITA="$HOME/Library/Application Support/krita/pykrita"
    ;;
  Linux)
    PYKRITA="$HOME/.local/share/krita/pykrita"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    # Windows — use mklink (requires a terminal with admin rights, or Developer Mode)
    APPDATA_KRITA="$APPDATA/krita/pykrita"
    if [ ! -d "$APPDATA_KRITA" ]; then
      echo "Creating pykrita directory: $APPDATA_KRITA"
      mkdir -p "$APPDATA_KRITA"
    fi
    echo "Linking on Windows…"
    cmd //c "mklink \"$APPDATA_KRITA\\kritai.desktop\" \"$REPO_DIR\\kritai.desktop\"" 2>/dev/null || \
      echo "  kritai.desktop already linked or failed (run as admin if needed)"
    cmd //c "mklink /D \"$APPDATA_KRITA\\kritai\" \"$REPO_DIR\\kritai\"" 2>/dev/null || \
      echo "  kritai/ already linked or failed (run as admin if needed)"
    echo "Done. Restart Krita and enable Kritai in the Python Plugin Manager."
    exit 0
    ;;
  *)
    echo "Unsupported OS: $(uname -s)"
    exit 1
    ;;
esac

# macOS / Linux
if [ ! -d "$PYKRITA" ]; then
  echo "Creating pykrita directory: $PYKRITA"
  mkdir -p "$PYKRITA"
fi

ln -sf "$REPO_DIR/kritai.desktop" "$PYKRITA/kritai.desktop"
ln -sf "$REPO_DIR/kritai"         "$PYKRITA/kritai"

echo "Linked into: $PYKRITA"
echo "Done. Restart Krita and enable Kritai in the Python Plugin Manager."
