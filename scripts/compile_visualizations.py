import os
from pathlib import Path


def collect_image_paths(figures_dir: str) -> list[str]:
    allowed_exts = {'.png', '.jpg', '.jpeg', '.svg'}
    paths: list[str] = []
    for root, _, files in os.walk(figures_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in allowed_exts:
                paths.append(os.path.join(root, f))
    return paths


def build_html_gallery(image_paths: list[str], title: str = "Visualizations") -> str:
    items = []
    for p in image_paths:
        rel = p.replace('\\', '/')
        caption = os.path.basename(p)
        items.append(
            f'<div class="card"><img src="../{rel}" alt="{caption}"><div class="cap">{caption}</div></div>'
        )
    grid = "\n".join(items)
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #fafafa; }}
    h1 {{ margin: 0 0 16px; }}
    .subtitle {{ color: #666; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .card img {{ width: 100%; height: auto; display: block; }}
    .cap {{ padding: 8px 12px; font-size: 13px; color: #444; border-top: 1px solid #eee; word-break: break-word; }}
    .empty {{ color: #888; }}
  </style>
  </head>
  <body>
    <h1>{title}</h1>
    <div class=\"subtitle\">Auto-generated gallery of plots from reports/figures</div>
    <div class=\"grid\">
      {grid if grid else '<div class="empty">No images found in reports/figures.</div>'}
    </div>
  </body>
</html>
"""
    return html


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    figures_dir = project_root / 'reports' / 'figures'
    out_dir = project_root / 'reports' / 'visualizations'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'visualizations.html'

    images = collect_image_paths(str(figures_dir))
    html = build_html_gallery(images, title='Breast Cancer Detection - Visualizations')
    out_file.write_text(html, encoding='utf-8')
    print(f'Wrote {out_file} with {len(images)} images.')


if __name__ == '__main__':
    main()


