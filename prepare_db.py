import gdown
import zipfile
import os
import shutil

zip_url = "https://drive.google.com/uc?id=1TRCTZ_txfmdzSfEGr_YXS9h4Kx4ZWNEx"
zip_path = "chroma_dataset.zip"
extract_path = "chroma_dataset_temp"
target_dir = "chroma_db"

if os.path.exists(target_dir):
    print("Database already exists.")
    exit()

print("Downloading chroma dataset...")
gdown.download(url=zip_url, output=zip_path, quiet=False)

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
os.remove(zip_path)

contents = os.listdir(extract_path)
if len(contents) == 1:
    shutil.move(os.path.join(extract_path, contents[0]), target_dir)
else:
    shutil.move(extract_path, target_dir)

print("Database is ready.")
