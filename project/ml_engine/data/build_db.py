import json
import urllib.request
import urllib.parse
import time
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent
DB_FILE = DATA_DIR / "polish_landmarks_db.json"
REF_IMAGES_DIR = DATA_DIR / "reference_images"

# LIMIT POBIERANIA DLA POC (Na serwerze można ustawić na None)
DOWNLOAD_LIMIT = 50

def fetch_landmarks():
    print("Fetching from Wikidata...")
    url = "https://query.wikidata.org/sparql"
    
    query = """
    SELECT ?item ?itemLabel ?typeLabel ?height ?image WHERE {
      VALUES ?class { wd:Q23413 wd:Q16560 wd:Q1311063 wd:Q39715 wd:Q4989906 }
      ?item wdt:P31/wdt:P279* ?class .
      ?item wdt:P17 wd:Q36 .
      ?item wdt:P18 ?image .
      OPTIONAL { ?item wdt:P2048 ?height . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "pl". }
    }
    """
    
    headers = {
        "User-Agent": "ProjektZespolowy-ML/1.0 (test@example.com)",
        "Accept": "application/json"
    }
    
    try:
        req = urllib.request.Request(f"{url}?query={urllib.parse.quote(query)}&format=json", headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data from Wikidata: {e}")
        print("Using fallback mock data due to Wikidata API outage...")
        # Fallback mock data matching the Wikidata SPARQL JSON format
        data = {
            "results": {
                "bindings": [
                    {
                        "itemLabel": {"value": "Zamek Kapituły Warmińskiej w Olsztynie"},
                        "typeLabel": {"value": "zamek"},
                        "height": {"value": "35"},
                        "image": {"value": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Olsztyn_Zamek_Kapituly_Warminskiej_1.jpg/800px-Olsztyn_Zamek_Kapituly_Warminskiej_1.jpg"}
                    },
                    {
                        "itemLabel": {"value": "Zamek Królewski na Wawelu"},
                        "typeLabel": {"value": "zamek"},
                        "height": {"value": "50"},
                        "image": {"value": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Wawel_Katedra_i_Zamek_z_lotu_ptaka.jpg/800px-Wawel_Katedra_i_Zamek_z_lotu_ptaka.jpg"}
                    },
                    {
                        "itemLabel": {"value": "Zamek w Malborku"},
                        "typeLabel": {"value": "zamek"},
                        "height": {"value": "50"},
                        "image": {"value": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Malbork_Castle_-_Poland_-_panoramio.jpg/800px-Malbork_Castle_-_Poland_-_panoramio.jpg"}
                    }
                ]
            }
        }
        
    landmarks = []
    seen = set()
    downloaded_count = 0
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(data['results']['bindings'])} results. Processing up to {DOWNLOAD_LIMIT}...")
    
    for bind in data["results"]["bindings"]:
        if DOWNLOAD_LIMIT is not None and downloaded_count >= DOWNLOAD_LIMIT:
            break
            
        name = bind.get("itemLabel", {}).get("value", "")
        if not name or name.startswith("Q") or name in seen:
            continue
            
        image_url = bind.get("image", {}).get("value", "")
        if not image_url:
            continue
            
        seen.add(name)
        
        type_str = bind.get("typeLabel", {}).get("value", "").lower()
        height_str = bind.get("height", {}).get("value")
        
        height = None
        if height_str:
            try:
                height = float(height_str)
            except:
                pass
                
        if not height:
            if "zamek" in type_str or "zamek" in name.lower():
                height = 30.0
            elif "pałac" in type_str or "pałac" in name.lower():
                height = 20.0
            elif "wieża widokowa" in type_str or "wieża widokowa" in name.lower():
                height = 25.0
            elif "wieża" in type_str or "wieża" in name.lower():
                height = 35.0
            elif "latarnia" in type_str or "latarnia" in name.lower():
                height = 30.0
            elif "pomnik" in type_str or "pomnik" in name.lower():
                height = 10.0
            else:
                height = 15.0
                
        # Zapisywanie zdjęcia
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        image_filename = f"{clean_name}.jpg"
        image_path = REF_IMAGES_DIR / image_filename
        
        if not image_path.exists():
            try:
                print(f"Downloading image for: {name}")
                req_img = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
                with urllib.request.urlopen(req_img, timeout=10) as response, open(image_path, 'wb') as out_file:
                    out_file.write(response.read())
                time.sleep(2.0) # Dłuższy czas oczekiwania, żeby uniknąć 429
            except Exception as e:
                print(f"Failed to download image for {name}: {e}. Using local fallback image.")
                import shutil
                try:
                    fallback_path = Path(r"C:\projekty-studia\projekt-zespołowy\project\media\photos\0031e840-9d6a-4e84-bd1a-fd3d68534c47\test.jpg")
                    if fallback_path.exists():
                        shutil.copyfile(fallback_path, image_path)
                    else:
                        continue
                except:
                    continue
                
        downloaded_count += 1
        
        landmarks.append({
            "name": name,
            "type": type_str,
            "height": height,
            "image_filename": image_filename
        })
        
    print(f"Processed {len(landmarks)} landmarks with images.")
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(landmarks, f, ensure_ascii=False, indent=2)
    print(f"Saved database to {DB_FILE}")

if __name__ == "__main__":
    fetch_landmarks()
