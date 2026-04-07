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

# Generate Manual.html from README.md (no external dependencies).
python3 - "$REPO_DIR" <<'PYEOF'
import re, sys, html as ht

src = sys.argv[1] + "/README.md"
dst = sys.argv[1] + "/kritai/Manual.html"

with open(src) as f:
    text = f.read()

# Strip GitHub alert syntax like "> [!tip]"
text = re.sub(r">\s*\[!(note|tip|important|warning|caution)\]\s*\n", "", text, flags=re.I)

lines = text.split("\n")
out = []
in_code = False
in_list = False
list_tag = ""

for line in lines:
    # Fenced code blocks
    if line.startswith("```"):
        if in_code:
            out.append("</code></pre>")
            in_code = False
        else:
            out.append("<pre><code>")
            in_code = True
        continue
    if in_code:
        out.append(ht.escape(line))
        continue

    stripped = line.strip()

    # Close open list if line is not a list item or blank
    if in_list and stripped and not re.match(r"^(\d+\.|[-*])\s", stripped):
        out.append(f"</{list_tag}>")
        in_list = False

    # Headings
    m = re.match(r"^(#{1,6})\s+(.*)", line)
    if m:
        n = len(m.group(1))
        out.append(f"<h{n}>{m.group(2)}</h{n}>")
        continue

    # Unordered list
    m = re.match(r"^[-*]\s+(.*)", stripped)
    if m:
        if not in_list or list_tag != "ul":
            if in_list:
                out.append(f"</{list_tag}>")
            out.append("<ul>")
            in_list = True
            list_tag = "ul"
        out.append(f"  <li>{m.group(1)}</li>")
        continue

    # Ordered list
    m = re.match(r"^\d+\.\s+(.*)", stripped)
    if m:
        if not in_list or list_tag != "ol":
            if in_list:
                out.append(f"</{list_tag}>")
            out.append("<ol>")
            in_list = True
            list_tag = "ol"
        out.append(f"  <li>{m.group(1)}</li>")
        continue

    # Blockquote
    if stripped.startswith(">"):
        content = stripped[2:] if stripped.startswith("> ") else stripped[1:]
        out.append(f"<blockquote><p>{content}</p></blockquote>")
        continue

    # Blank line
    if not stripped:
        if in_list:
            pass  # keep list open across blank lines
        out.append("")
        continue

    # Paragraph
    out.append(f"<p>{stripped}</p>")

if in_list:
    out.append(f"</{list_tag}>")

body = "\n".join(out)

# Inline formatting
body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
body = re.sub(r"`(.+?)`", r"<code>\1</code>", body)
body = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', body)

with open(dst, "w") as f:
    f.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            "<title>Kritai Manual</title></head><body>\n")
    f.write(body)
    f.write("\n</body></html>\n")
PYEOF
echo "Generated Manual.html from README.md"

ln -sf "$REPO_DIR/kritai.desktop" "$PYKRITA/kritai.desktop"
ln -sf "$REPO_DIR/kritai"         "$PYKRITA/kritai"
# Clean up stale root-level manual symlink from older installs.
rm -f "$PYKRITA/Manual.html"

echo "Linked into: $PYKRITA"
echo "Done. Restart Krita and enable Kritai in the Python Plugin Manager."
