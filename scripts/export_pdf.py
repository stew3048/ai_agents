"""
將 output/test_overlay/analyze/index.html 匯出為 PDF。

每頁配置：
  第 1 頁：封面（header）
  第 2~6 頁：5 個情境場景，各一頁
  第 7 頁：極端天空情境雷達圖
  第 8 頁：全資料集平均指標 + 結論

輸出：output/test_overlay/analyze/sky_segmentation_report.pdf
"""

import os, sys, textwrap
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
HTML_SRC = _root / "output" / "test_overlay" / "analyze" / "index.html"
PDF_OUT  = _root / "output" / "test_overlay" / "analyze" / "sky_segmentation_report.pdf"

# ─── 讀取原始 HTML，注入 @page 分頁 CSS ──────────────────────────────────────

html_text = HTML_SRC.read_text(encoding='utf-8')

import re

# （verdict 保留原本 <br><br> 間距）

# ── 3. 場景換頁 & 標記最後一個 scenario（日間建築物）
positions = [m.start() for m in re.finditer(r'<section class="scenario', html_text)]
print(f"找到 {len(positions)} 個場景")

# 最後一個 scenario 加上 pre-radar class，讓 CSS 將其 margin-bottom 歸零
last_pos = positions[-1]
html_text = (html_text[:last_pos]
             + html_text[last_pos:].replace('<section class="scenario"',
                                            '<section class="scenario pre-radar"', 1))
# positions 位置不變，繼續插入換頁標記
positions = [m.start() for m in re.finditer(r'<section class="scenario', html_text)]
page_break_div = '<div class="pdf-pagebreak"></div>\n    '
for i in range(len(positions) - 1, -1, -1):
    if i > 0 and i % 2 == 0:
        html_text = html_text[:positions[i]] + page_break_div + html_text[positions[i]:]

# 插入 print 專用 CSS（在 </style> 之前）
SCENARIO_GAP = '52px'   # 同頁兩個 scenario 之間的距離
RADAR_GAP    = '18px'   # 日間建築物 → 雷達圖（與結論同距）
SUMMARY_GAP  = '18px'   # 雷達圖 → 結論

print_css = textwrap.dedent(f"""
  /* ===== PDF（A4 直向）===== */
  @page {{
    size: A4 portrait;
    margin: 8mm 8mm;
  }}
  @media print {{
    body {{ background: white !important; font-size: 12.5px; }}
    main {{ max-width: 100% !important; padding: 0 !important; }}

    /* ── Header：單行極簡，緊貼第一個場景 ── */
    header {{
      display: flex !important;
      align-items: center !important;
      height: 34px !important;
      padding: 0 14px !important;
      margin-bottom: 20px !important;
      page-break-after: avoid !important;
      border-radius: 0 !important;
    }}
    header h1 {{ font-size: 1.3em !important; margin: 0 !important; white-space: nowrap; }}
    header p  {{ display: none !important; }}

    /* ── 換頁點 ── */
    .pdf-pagebreak {{
      page-break-before: always !important;
      height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;
    }}

    /* ── Scenario ── */
    .scenario {{
      page-break-before: avoid !important;
      page-break-inside: avoid !important;
      margin-bottom: {SCENARIO_GAP} !important;
    }}
    /* 日間建築物（P3 最後一個 scenario）margin-bottom 歸零，由 RADAR_GAP 控制 */
    .pre-radar {{
      margin-bottom: 0 !important;
      box-shadow: none !important;
      border: 1px solid #ccc !important;
      border-radius: 6px !important;
      overflow: hidden !important;
    }}
    .scenario h2 {{
      padding: 7px 13px !important;
      font-size: 0.94em !important;
      background: #fafafa !important;
    }}

    /* ── 四宮格 ── */
    .image-grid {{
      display: grid !important;
      grid-template-columns: repeat(2, 1fr) !important;
      gap: 0 !important;
    }}
    .orig-card, .method-card {{
      padding: 7px 8px !important;
      border-right: 1px solid #eee !important;
      border-bottom: 1px solid #eee !important;
    }}
    .img-box {{
      width: 100% !important;
      height: 115px !important;
      object-fit: contain !important;
      background: #f5f5f5 !important;
      border-radius: 4px !important;
      display: block !important;
    }}
    .method-title  {{ font-size: 0.82em !important; margin-bottom: 4px !important; }}
    .metrics-row   {{ font-size: 0.70em !important; gap: 3px !important; margin-top: 4px !important; flex-wrap: wrap !important; display: flex !important; }}
    .metrics-row span {{ padding: 2px 4px !important; }}
    .comment       {{ font-size: 0.72em !important; padding: 3px 5px !important; margin-top: 3px !important; line-height: 1.32 !important; }}

    /* ── 雷達圖：class="summary-section radar-section"
         用雙類選擇器（更高優先度）確保 avoid 蓋過 always ── */
    .summary-section.radar-section {{
      page-break-before: avoid !important;
      break-before: avoid !important;
      page-break-inside: avoid !important;
      box-shadow: none !important;
      border: 1px solid #ccc !important;
      border-radius: 6px !important;
      margin-top: {RADAR_GAP} !important;
      margin-bottom: 0 !important;
    }}
    .summary-section.radar-section h2 {{ padding: 7px 13px !important; font-size: 0.94em !important; }}
    .radar-body {{
      display: flex !important;
      flex-direction: row !important;
      padding: 10px 12px !important;
      gap: 14px !important;
      align-items: flex-start !important;
    }}
    .radar-img-wrap {{ flex: 0 0 240px !important; }}
    .radar-img      {{ width: 100% !important; border-radius: 5px !important; }}
    .radar-text     {{ flex: 1 !important; display: flex !important; flex-direction: column !important; gap: 7px !important; }}
    .radar-item     {{ padding: 6px 8px !important; }}
    .radar-item p   {{ font-size: 0.78em !important; line-height: 1.45 !important; }}
    .radar-method-title {{ font-size: 0.88em !important; margin-bottom: 3px !important; }}

    /* ── 結論：與雷達圖同頁（P3），不換頁 ── */
    .summary-section:not(.radar-section) {{
      page-break-before: avoid !important;
      break-before: avoid !important;
      page-break-inside: avoid !important;
      box-shadow: none !important;
      border: 1px solid #ccc !important;
      border-radius: 6px !important;
      margin-top: {SUMMARY_GAP} !important;
    }}
    .summary-section:not(.radar-section) h2  {{ padding: 6px 13px !important; font-size: 0.90em !important; }}
    .summary-section .note {{
      padding: 4px 13px !important;
      font-size: 0.76em !important;
      line-height: 1.4 !important;
    }}
    table  {{ font-size: 0.76em !important; width: 100% !important; }}
    td, th {{ padding: 4px 7px !important; }}
    .verdict {{
      padding: 6px 13px 8px !important;
      font-size: 0.78em !important;
      line-height: 1.42 !important;
    }}
  }}
""")

html_text = html_text.replace('</style>', print_css + '\n</style>')

# 暫存修改後的 HTML
tmp_html = HTML_SRC.parent / "_print_version.html"
tmp_html.write_text(html_text, encoding='utf-8')
print(f"已建立列印版 HTML：{tmp_html}")

# ─── 用 Playwright 輸出 PDF ────────────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(tmp_html.as_uri(), wait_until='networkidle', timeout=30000)

        # 等圖片載入
        page.wait_for_timeout(2000)

        page.pdf(
            path=str(PDF_OUT),
            format='A4',
            landscape=False,
            print_background=True,
            margin={'top': '8mm', 'bottom': '8mm', 'left': '8mm', 'right': '8mm'},
        )
        browser.close()

    print(f"\nPDF saved: {PDF_OUT}")
    print(f"   Pages: header + 5 scenarios + radar + summary = ~8 pages")

except ImportError:
    print("\n❌ playwright 尚未安裝，請執行：")
    print("   pip install playwright")
    print("   python -m playwright install chromium")
finally:
    # 清理暫存 HTML（保留或刪除皆可）
    # tmp_html.unlink(missing_ok=True)
    pass
