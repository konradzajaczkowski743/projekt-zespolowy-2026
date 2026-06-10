import urllib.request
import urllib.parse
import re
import os
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent
REF_IMAGES_DIR = DATA_DIR / "reference_images"

LANDMARKS = {
    "Zamek Kapituły Warmińskiej w Olsztynie": "Zamek_Kapituły_Warmińskiej_w_Olsztynie",
    "Zamek Królewski na Wawelu": "Zamek_Królewski_na_Wawelu",
    "Zamek Pieskowa Skała": "Zamek_Pieskowa_Skała"
}

def fetch_images():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for query, title in LANDMARKS.items():
        lm_dir = REF_IMAGES_DIR / title
        
        # Wyczyść stary katalog
        if lm_dir.exists():
            for f in lm_dir.glob("*.jpg"):
                os.remove(f)
        lm_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Pobieranie ze strony Wikipedii: {title}...")
        url = f'https://pl.wikipedia.org/wiki/{urllib.parse.quote(title)}'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                html = resp.read().decode('utf-8')
                
                # Złap wszystko co wygląda jak obraz JPG w kodzie HTML Wikipedii
                images = re.findall(r'src=\"(//upload\.wikimedia\.org/wikipedia/commons/thumb/[^\"]+\.jpg/[^\"]+)\"', html, re.IGNORECASE)
                
                # Filtrujemy by wziąć większe wersje, albo chociaż te co są
                unique_images = list(set(images))
                print(f'Znaleziono {len(unique_images)} miniatury dla {title}')
                
                downloaded = 0
                for img in unique_images:
                    if downloaded >= 15:
                        break
                        
                    # Wikipedia miniaturki mają format .../thumb/x/y/nazwa.jpg/120px-nazwa.jpg
                    # Zmiana rozmiaru na 800px:
                    img_url = 'https:' + re.sub(r'/\d+px-', '/800px-', img)
                    
                    try:
                        req2 = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(req2, timeout=5) as r2:
                            content = r2.read()
                            if len(content) > 10000: # Wymagaj min. 10KB
                                with open(lm_dir / f'{downloaded+1:02d}.jpg', 'wb') as f:
                                    f.write(content)
                                downloaded += 1
                    except Exception as e:
                        # Może nie istnieć w rozdzielczości 800px, pobierzmy oryginał
                        try:
                            orig_url = 'https:' + img
                            req3 = urllib.request.Request(orig_url, headers=headers)
                            with urllib.request.urlopen(req3, timeout=5) as r3:
                                content = r3.read()
                                if len(content) > 5000:
                                    with open(lm_dir / f'{downloaded+1:02d}.jpg', 'wb') as f:
                                        f.write(content)
                                    downloaded += 1
                        except:
                            pass
                            
                    time.sleep(0.5)
                print(f"Pomyślnie pobrano {downloaded} zdjęć dla {title}.")
        except Exception as e:
            print(f"Błąd ogólny {title}: {e}")

if __name__ == "__main__":
    fetch_images()
