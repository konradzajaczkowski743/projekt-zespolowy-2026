import os
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
REF_IMAGES_DIR = DATA_DIR / "reference_images"

LANDMARKS = {
    "Zamek Kapituły Warmińskiej w Olsztynie": "Zamek_Kapituły_Warmińskiej_w_Olsztynie",
    "Zamek Królewski na Wawelu": "Zamek_Królewski_na_Wawelu",
    "Zamek Pieskowa Skała": "Zamek_Pieskowa_Skała",
    "Pałac w Wilanowie": "Pałac_w_Wilanowie",
    "Zamek Królewski w Warszawie": "Zamek_Królewski_w_Warszawie",
    "Pałac Kultury i Nauki": "Pałac_Kultury_i_Nauki"
}

def fetch_wiki_images():
    REF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Antigravity/1.0 (test@example.com)"}
    
    for name, title in LANDMARKS.items():
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        lm_dir = REF_IMAGES_DIR / clean_name
        lm_dir.mkdir(exist_ok=True)
        
        print(f"\n[{name}] Searching Wikipedia images...")
        url = f"https://pl.wikipedia.org/w/api.php?action=query&prop=images&titles={urllib.parse.quote(title)}&format=json&imlimit=50"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                pages = data["query"]["pages"]
                for page_id, page_data in pages.items():
                    if "images" in page_data:
                        images = page_data["images"]
                        downloaded = 0
                        for img in images:
                            img_title = img["title"]
                            if not (img_title.lower().endswith(".jpg") or img_title.lower().endswith(".jpeg") or img_title.lower().endswith(".png")):
                                continue
                                
                            img_url_query = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(img_title)}&prop=imageinfo&iiprop=url&format=json"
                            req_img = urllib.request.Request(img_url_query, headers=headers)
                            with urllib.request.urlopen(req_img) as resp_img:
                                data_img = json.loads(resp_img.read())
                                pages_img = data_img["query"]["pages"]
                                for _, p_img in pages_img.items():
                                    if "imageinfo" in p_img:
                                        img_url = p_img["imageinfo"][0]["url"]
                                        filename_encoded = urllib.parse.quote(img_url.split('/')[-1])
                                        thumb_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename_encoded}?width=800"
                                        
                                        image_filename = f"{downloaded:02d}.jpg"
                                        image_path = lm_dir / image_filename
                                        
                                        if image_path.exists():
                                            downloaded += 1
                                            continue
                                            
                                        print(f"  Downloading image {downloaded+1} from Commons...")
                                        try:
                                            req_dl = urllib.request.Request(thumb_url, headers=headers)
                                            with urllib.request.urlopen(req_dl) as resp_dl:
                                                with open(image_path, "wb") as f:
                                                    f.write(resp_dl.read())
                                            downloaded += 1
                                            time.sleep(0.5)
                                        except Exception as e:
                                            print(f"    Failed to download {thumb_url}: {e}")
        except Exception as e:
            print(f"Error {name}: {e}")

if __name__ == "__main__":
    fetch_wiki_images()
