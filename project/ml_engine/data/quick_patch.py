import json
import shutil
from pathlib import Path

db_file = Path('polish_landmarks_db.json')
with open(db_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if 'Olsztynie' in item['name'] or 'Olsztyn' in item['name']:
        item['image_filename'] = 'Zamek_Kapituly_Olsztyn.jpg'
        print(f"Updated {item['name']}")
        break

with open(db_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

ref_dir = Path('reference_images')
ref_dir.mkdir(exist_ok=True)
img_path = ref_dir / 'Zamek_Kapituly_Olsztyn.jpg'
src_img = Path(r'C:\projekty-studia\projekt-zespołowy\project\media\photos\0031e840-9d6a-4e84-bd1a-fd3d68534c47\test.jpg')

if src_img.exists():
    shutil.copyfile(src_img, img_path)
    print('Copied dummy image.')
else:
    print('Source image not found!')
