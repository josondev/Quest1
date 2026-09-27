import sys
sys.path.insert(0, '/app')
import os
os.chdir('/app')
from src.ingestion import StreamIngestionService
p = StreamIngestionService()
path = p._get_cookies_path()
print('COOKIES_PATH:', path)
if path:
    import pathlib
    f = pathlib.Path(path)
    print('SIZE:', f.stat().st_size)
    lines = f.read_text().splitlines()
    print('LINE_COUNT:', len(lines))
    for l in lines[:5]:
        print('LINE:', l[:80])
else:
    print('NO COOKIES - path is None')
