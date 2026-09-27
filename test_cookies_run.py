import sys
sys.path.insert(0, '/app')
import os
os.chdir('/app')
from src.ingestion import MediaIngestionPipeline
p = MediaIngestionPipeline()
path = p._get_cookies_path()
print('COOKIES_PATH:', path)
if path:
    import pathlib
    f = pathlib.Path(path)
    print('SIZE:', f.stat().st_size)
    print('FIRST_LINE:', f.open().readline().strip())
