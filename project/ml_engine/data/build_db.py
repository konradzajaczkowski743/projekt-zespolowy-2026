import os
import json
import urllib.request
import urllib.parse
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent
DB_FILE = DATA_DIR / "polish_landmarks_db.json"
REF_IMAGES_DIR = DATA_DIR / "reference_images"

# -----------------------------------------------------------------------------
# Konfiguracja: Zmień na None, aby pobrać wszystkie (kilkaset/tysiące) zdjęć.
MAX_DOWNLOADS = 15
# -----------------------------------------------------------------------------

def query_wikidata():
    print("Querying Wikidata for Polish landmarks with height and image...")
    # Zapytanie szuka budynków, struktur architektonicznych i pomników w Polsce,
    # które mają przypisaną wysokość (P2048) i zdjęcie (P18).
    query = """
    SELECT ?item ?itemLabel ?image ?height ?typeLabel WHERE {
      { ?item wdt:P31/wdt:P279* wd:Q41176. }
      UNION
      { ?item wdt:P31/wdt:P279* wd:Q811979. }
      UNION
      { ?item wdt:P31/wdt:P279* wd:Q4989906. }
      
      ?item wdt:P17 wd:Q36.
      ?item wdt:P18 ?image.
      ?item p:P2048/psn:P2048/wikibase:quantityAmount ?height.
      ?item wdt:P31 ?type.
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "pl,en". }
    }
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "ProjektZespolowyML/1.0 (test-script@example.com)",
        "Accept": "application/json"
    }
    
    query_params = urllib.parse.urlencode({'query': query})
    full_url = f"{url}?{query_params}"
    
    req = urllib.request.Request(full_url, headers=headers)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                break # Sukces
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"Rate limited (429). Czekam 5 sekund przed kolejną próbą... (Próba {attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                print(f"Błąd HTTP {e.code} podczas pobierania danych SPARQL: {e}")
                return None
        except Exception as e:
            print(f"Błąd podczas pobierania danych SPARQL: {e}")
            return None
    else:
        print("Przekroczono maksymalną liczbę prób połączenia z Wikidata API. Używam wbudowanego zestawu ratunkowego (Fallback).")
        return None
        
    results = data['results']['bindings']
    
    landmarks = []
    seen_names = set()
    
    for row in results:
        name = row['itemLabel']['value']
        if name in seen_names or name.startswith("Q"):
            continue
            
        image_url = row['image']['value']
        height = float(row['height']['value'])
        type_label = row.get('typeLabel', {}).get('value', 'budynek')
        
        lm_type = "budynek"
        lower_type = type_label.lower()
        if "zamek" in lower_type: lm_type = "zamek"
        elif "kościół" in lower_type or "church" in lower_type or "bazylika" in lower_type or "katedra" in lower_type: lm_type = "kościół"
        elif "pomnik" in lower_type or "statue" in lower_type or "monument" in lower_type: lm_type = "pomnik"
        elif "pałac" in lower_type or "palace" in lower_type: lm_type = "pałac"
        elif "wieża" in lower_type or "tower" in lower_type or "latarnia" in lower_type: lm_type = "wieża"
        
        # Tworzenie URL miniaturki (zmniejsza zużycie bandwidth i przyspiesza pobieranie)
        filename_encoded = image_url.split('/')[-1]
        thumb_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename_encoded}?width=800"
        
        landmarks.append({
            "name": name,
            "type": lm_type,
            "height": height,
            "image_url": thumb_url
        })
        seen_names.add(name)
        
    print(f"Found {len(landmarks)} unique landmarks with specified constraints.")
    return landmarks

def fetch_landmarks():
    print("Building local landmark database from Wikidata...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    landmarks = query_wikidata()
    
    if not landmarks:
        print("Trwa używanie zestawu ratunkowego ze względu na niedostępność bazy wiedzy...")
        landmarks = [
            {
                "name": "Zamek Kapituły Warmińskiej w Olsztynie",
                "type": "zamek",
                "height": 35.0,
                "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Olsztyn_Zamek_Kapituly_Warminskiej_1.jpg?width=800"
            },
            {
                "name": "Pomnik Chrystusa Króla w Świebodzinie",
                "type": "pomnik",
                "height": 36.0,
                "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Christ_the_King_Statue_1.JPG?width=800"
            },
            {
                "name": "Zamek Królewski na Wawelu",
                "type": "zamek",
                "height": 50.0,
                "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Wawel_Katedra_i_Zamek_z_lotu_ptaka.jpg?width=800"
            }
        ]
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }

    db_entries = []
    downloaded = 0
    
    for lm in landmarks:
        if MAX_DOWNLOADS is not None and downloaded >= MAX_DOWNLOADS:
            print(f"Reached MAX_DOWNLOADS limit ({MAX_DOWNLOADS}). Stopping.")
            break
            
        clean_name = "".join(c for c in lm["name"] if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        if not clean_name:
            continue
            
        image_filename = f"{clean_name}.jpg"
        image_path = REF_IMAGES_DIR / image_filename
        
        if not image_path.exists():
            print(f"Downloading: {lm['name']} (H: {lm['height']}m)...")
            try:
                req = urllib.request.Request(lm["image_url"], headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    content = response.read()
                    
                if len(content) > 5000:
                    with open(image_path, 'wb') as f:
                        f.write(content)
                    downloaded += 1
                    time.sleep(0.5)
                else:
                    print(f"  -> Failed: file too small.")
                    continue
            except Exception as e:
                print(f"  -> Error downloading: {e}")
                continue
        else:
            print(f"Exists: {lm['name']} (H: {lm['height']}m)")
            
        db_entries.append({
            "name": lm["name"],
            "type": lm["type"],
            "height": lm["height"],
            "image_filename": image_filename
        })
        
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db_entries, f, ensure_ascii=False, indent=2)
        
    success_count = sum(1 for e in db_entries if e["image_filename"])
    print(f"Database built successfully. Processed {len(db_entries)} landmarks ({success_count} images).")

if __name__ == "__main__":
    fetch_landmarks()
