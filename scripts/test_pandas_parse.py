import pandas as pd

fname = "/home/kamau/.gemini/tmp/2b8aa2f42273da2920d6a0846a0beee0179039373f45f52c93793c0248598408/nse_ajax_display_prices_agric.html"
with open(fname) as f:
    html = f.read()

try:
    tables = pd.read_html(html)
    print(f"Found {len(tables)} tables")
    if tables:
        print(tables[0].head())
except Exception as e:
    print(f"Error: {e}")
