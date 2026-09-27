import sys, os
sys.path.insert(0, '/app')
os.chdir('/app')
from src.config import settings
print('conn_str set:', bool(settings.azure_storage_connection_string))
print('blob_name:', settings.yt_cookies_blob_name)
print('container:', settings.azure_blob_container)
from pathlib import Path
from azure.storage.blob import BlobServiceClient
service = BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
client = service.get_container_client(settings.azure_blob_container)
blob = client.get_blob_client(settings.yt_cookies_blob_name)
data = blob.download_blob().readall()
print('cookies downloaded:', len(data), 'bytes')
print('first 100 chars:', data[:100].decode())
