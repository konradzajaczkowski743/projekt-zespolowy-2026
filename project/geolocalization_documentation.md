# Dokumentacja Systemu Geolokalizacji (VeriVision Core)

Ten dokument opisuje architekturę, wykorzystane modele oraz zasady działania systemu geolokalizacji w projekcie, ze szczególnym uwzględnieniem szacowania odległości do rozpoznanych obiektów. System operuje całkowicie lokalnie (bez API zewnętrznych), wykorzystując wielowarstwowe podejście oparte o głębokie sieci neuronowe.

---

## 1. Wykorzystane modele i ich cele

System jest zbudowany z kaskady modeli ML, z których każdy ma swoją unikalną specjalizację. 

### A. Rozpoznawanie Lokalizacji (Geolocator)
1. **ViT World Landmarks (`mmgyorke/vit-world-landmarks`)** 
   - **Cel:** Ultraszybkie klasyfikowanie znanych na całym świecie zabytków i budynków (Landmarks).
   - **Rola:** Warstwa pierwsza (Layer 1). Szuka "strzałów w dziesiątkę". Jeśli pewność jest wysoka (ponad 70%), system uznaje rozpoznanie za ostateczne.
2. **StreetCLIP (`geolocal/StreetCLIP`)**
   - **Cel:** Detekcja państw i miast zero-shot na podstawie wiedzy ogólnej zakodowanej w obrazach. Konfigurowalny pod zbiór etykiet (m.in. duże miasta polskie i europejskie).
   - **Rola:** Warstwa druga (Layer 2). Działa jako potężny fallback, kiedy ViT nie jest pewien. Lepiej radzi sobie ze zgadywaniem ogólnej lokalizacji (np. "Paris, France" czy "Warszawa, Polska") na podstawie architektury ulic, układu drogowego czy widoków.
3. **Gemma VLM (`gemma4:e2b` przez Ollama)**
   - **Cel:** Wielomodalny LLM pełniący rolę warstwy zapasowej i weryfikatora tekstowego.
   - **Rola:** Warstwa trzecia (Layer 3). Wyciąga lokalizację ze słownego opisu obrazu, dopytuje o szczegóły budynków i rozstrzyga konflikty.
4. **Photon (Lokalny serwer geocodingu)**
   - **Cel:** Przekształca tekstową nazwę lokalizacji (np. "Sagrada Familia") na współrzędne GPS (szerokość i długość geograficzna), aby móc je matematycznie porównać z danymi EXIF zaszytymi w zdjęciu.

### B. Szacowanie Odległości (Distance Estimator)
5. **Depth Anything V2 Base (`depth-anything/Depth-Anything-V2-Base-hf`)**
   - **Cel:** Generuje precyzyjną mapę głębi dla każdego piksela (monocular depth estimation).
   - **Rola:** Ocenia topografię 3D zdjęcia. Wskazuje, co jest na pierwszym planie, a co w tle, pomagając stworzyć obrys (bounding box) najważniejszego budynku w centrum kadru.

### C. Asynchroniczne przetwarzanie (Celery)
Aby zapobiec blokowaniu interfejsu użytkownika w trakcie powolnych operacji ML (np. czasochłonnych zapytań do lokalnych modeli takich jak Gemma), wprowadzono mechanizm asynchroniczny:
- **Celery Worker**: Oddzielny proces (kontener `projekt_worker`), który przejmuje ciężkie obliczenia w tle i rozkłada obciążenie systemu.
- **Redis**: Służy jako broker wiadomości (Message Broker). Przyjmuje polecenie od API i zarządza statusem kolejkowanych zdjęć.
- **Rola:** Dzięki temu API może od razu zwrócić użytkownikowi `task_id`, a ciężkie wnioskowanie wizualne (Geolocator, Distance Estimator) jest bezpiecznie wykonywane w tle, minimalizując ryzyko przekroczenia limitu czasu (HTTP timeout) w przeglądarce.

---

## 2. Jak system radzi sobie z różnymi scenariuszami (Use Cases)?

System jest zaprojektowany tak, aby poszczególne modele wzajemnie łatały swoje wady w zależności od dostarczonego zdjęcia.

### Przypadek 1: Zdjęcie bardzo znanego zabytku (np. Sagrada Familia)
- **Działanie:** ViT dostaje zdjęcie i natychmiast rozpoznaje zabytek z ogromną pewnością (np. 85%).
- **Rezultat:** ViT wygrywa od razu. StreetCLIP działa w tle dla dodatkowego kontekstu, ale głównym wynikiem zostaje nazwa obiektu z ViT. Distance Estimator po prostu pobiera wysokość z Gemmy dla tej nazwy i bezbłędnie liczy dystans.

### Przypadek 2: Panorama miasta ze znanym budynkiem (np. Paryż z małą Wieżą Eiffla)
- **Działanie:** ViT może mieć problem i z powodu perspektywy pomylić budynek (np. podaje "Big Ben" z niską pewnością 40%). StreetCLIP jednak analizuje dachy i ulice i jest pewien na 96%, że to "Paris, France". Geolokalizację wygrywa StreetCLIP.
- **Interwencja Detektywa:** Ponieważ StreetCLIP podał nazwę ogólną (miasto z przecinkiem), wkracza moduł Distance Estimator z *trybem detektywa*. Wysyła zdjęcie do Gemmy pytając: *"Skoro to Paryż, czy widzisz tu jakiś konkretny budynek?"*.
- **Rezultat:** Gemma wykrywa "Eiffel Tower" w obrazie. System nadpisuje generyczny "Paryż" wieżą Eiffla i odległość jest liczona dla konkretnego budynku z dużą dokładnością.

### Przypadek 3: Zwykła ulica w mieście bez charakterystycznych punktów
- **Działanie:** ViT jest zagubiony (brak zabytków). StreetCLIP z sukcesem rozpoznaje np. "Kraków, Polska". 
- **Interwencja Detektywa:** Gemma znów zostaje poproszona o znalezienie zabytku, ale na zdjęciu są tylko bloki mieszkalne, więc Gemma odpowiada "NO". 
- **Rezultat:** Moduł szacowania dystansu rezygnuje z budowania wirtualnego obiektu *"Zabytek (Kraków)"* i opiera się po prostu na obiektach wykrytych przez YOLO/RT-DETR na pierwszym planie (np. samochód, osoba), żeby ocenić dystans. Jeśli nic nie znajdzie — wyłącza licznik odległości.

---

## 3. Omówienie JSON-a wyjściowego

Pełen JSON jest obszerny i integruje wyniki wszystkich modeli. Poniżej rozbicie kluczowych węzłów związanych z geolokalizacją i odległością.

```json
{
  "geo_verification": {
    "predicted_region": "Paris, France",      // Ostateczny zwycięzca (najlepszy tekstowy strzał)
    "prediction_source": "streetclip",        // Kto wygrał? (vit_landmarks, streetclip, gemma_vlm, gemma_arbitration)
    "confidence_score": 0.964,                // Poziom pewności algorytmu, który wygrał (0.0 do 1.0)
    "is_location_consistent": true,           // TRUE jeśli obliczona geolokalizacja pokrywa się ze współrzędnymi GPS z pliku EXIF w promieniu < 50 km.
    "distance_km": null,                      // Jeśli były GPSy z pliku, to ile wynosi różnica kilometrów między EXIF a predykcją (tu null, bo brak EXIFa).
    
    // Kontekst i poszczególne warstwy:
    "vit_predicted_region": "Landmark: big ben", // To co zobaczył ViT, niezależnie od tego czy wygrał.
    "description_location": null,                // Lokalizacja ewentualnie wycięta ze zwykłego opisu Gemmy.
    "streetclip_top5": [                         // 5 najlepszych strzałów StreetCLIP-a (pomaga zdiagnozować pomyłki).
      { "label": "Paris, France", "score": 0.964 },
      { "label": "Lyon, France",  "score": 0.0178 }
    ]
  },
  
  "distance_estimation": {
    "has_main_object": true,                  // TRUE, jeśli system podjął się liczenia dystansu (bo nie był to np. czysty ocean/niebo).
    "reason": null,                           // Wyjaśnia powód, dlaczego `has_main_object` mogłoby być FALSE.
    "model_used": "1.0.0-depth-anything-v2",  // Model mapy głębi.
    
    // Metryka głębi
    "depth_map_stats": {
      "mean_depth": 0.1593,                   // Średnia głębia sceny (pomaga ocenić mgłę, tło).
      "min_depth": 0.0, "max_depth": 1.0
    },

    "main_object": {
      "label": "landmark (Eiffel Tower)",     // Etykieta obiektu do którego liczono dystans. W tym miejscu zachodzą dynamiczne zmiany z Gemmą.
      "confidence": 0.5,                      // Pewność, że to ten obiekt.
      "estimated_distance_m": 1250.4,         // WYLICZONA ODLEGŁOŚĆ W METRACH od obiektywu kamery (skalibrowana wysokością).
      "depth_value": 0.0563,                  // Surowa, relatywna wartość z mapy głębi w miejscu obiektu (0 to bardzo daleko, 1 to bardzo blisko).
      "bounding_box": {                       // Pikselowe współrzędne obrysu w jakim znaleziono budynek.
        "x_min": 634, "y_min": 420,
        "x_max": 2899, "y_max": 5366
      }
    }
  }
}
```

### Najważniejsze punkty dla rozumienia wyjścia:
1. Skrzyżowanie zmiennych `predicted_region` i `prediction_source` decyduje o głównym wyniku lokalizacji w raporcie.
2. Zmienna `is_location_consistent` to **kluczowy flag forensics**, który z punktu widzenia oszustw i analizy metadanych odpowiada na najważniejsze pytanie: *Czy osoba przesyłająca zdjęcie naprawdę zrobiła je tam, gdzie twierdzi GPS?*
3. Cały blok `distance_estimation` traktuje `predicted_region` priorytetowo tylko, jeżeli jest to specyficzny, namacalny obiekt (np. zabytek). Jeśli nie, system szuka na zdjęciu ludzi lub samochodów, żeby wypluć sensowny wynik w `estimated_distance_m`.

---

## 4. Wyzwania Techniczne: "Halucynacje" LLM i Rozwiązania Przyszłościowe

Modele VLM (Vision-Language Models), szczególnie ich mniejsze warianty zoptymalizowane pod lokalne działanie (jak `gemma4:e2b`), mogą cierpieć na tzw. "halucynacje" podczas precyzyjnych zapytań o fizyczne wymiary. Zjawisko to występuje, gdy model jest dopytywany o dokładną wysokość budowli w celu obliczenia dystansu (np. może zmyślić, że "Sagrada Familia ma 100m", pomimo faktu, że w rzeczywistości ma 172m). Taka pomyłka prowadzi bezpośrednio do sporych przekłamań wyliczanej odległości w węźle `estimated_distance_m`.

### Obecne rozwiązanie (Workaround)
- Wprowadzono predefiniowany słownik na poziomie kodu (np. w klasie `DistanceEstimator`), w którym znajdują się "sztywne", empiryczne dane dotyczące popularnych polskich i globalnych zabytków (np. *Pałac Kultury i Nauki, Wawel, Wieża Eiffla*).
- Przed wysłaniem zapytania do Gemmy, system najpierw weryfikuje czy obiekt nie znajduje się na liście gwarantowanej wysokości. Eliminuje to podatność na zmyślanie ułamkowych, nielogicznych wartości dla najsłynniejszych budowli.

**Dlaczego w obecnym systemie MVP zrezygnowano z technologii RAG?**
1. **Złożoność i Wymóg 100% Offline**: Głównym założeniem projektu jest pełna niezależność od zewnętrznych interfejsów (np. blokada zapytań do API Wikipedii). Lokalny RAG wymagałby pobrania gigabajtowej bazy danych i podpięcia lokalnego silnika wektorowego (np. ChromaDB, FAISS).
2. **Narzut Sprzętowy i Czasowy**: Tworzenie wektorów (embeddings), przeszukiwanie bazy i wstrzykiwanie długiego kontekstu do promptów wydłużyłoby kilkukrotnie czas przetwarzania zdjęcia (który już teraz wynosi do kilkudziesięciu sekund), a dodatkowy model *Embedding* wyczerpałby zasoby RAM standardowego komputera.
3. **Zasada Ockhama**: Dla skończonego zbioru popularnych zabytków, "twardy słownik" w strukturze mapy w czasie O(1) jest rozwiązaniem niezużywającym w ogóle zasobów pamięci komputera i gwarantującym 100% odporność na halucynacje.

### Rozwiązania Docelowe (Rozwój Architektury Produkcyjnej)
Aby całkowicie i w profesjonalny sposób zwalczyć problem zmyślania faktów metrycznych przez LLM-y w architekturze produkcyjnej (gdy dostępne będą klastry GPU), rozważa się następujące modernizacje:
1. **Model RAG (Retrieval-Augmented Generation)**: Zamiast polegać na ułomnej wewnętrznej pamięci modelu VLM, system dynamicznie przeszuka rzetelne źródło prawdy (np. lokalny wycinek bazy wiedzy) za pomocą frazy "Wysokość w metrach {nazwa_zabytku}". Prawdziwy, zweryfikowany fakt zostanie wstrzyknięty do promptu Gemmy jako kontekst tła (ang. *context injection*).
2. **Potężniejszy VLM (Qwen-VL-Max lub LLaVA-1.5-13B)**: Wymiana bazowego modelu na architekturę o parametrach przekraczających kilkanaście miliardów węzłów. Znacznie potężniejsze modele posiadają znacznie szerszy zasób *Zero-Shot Knowledge* i rzadziej poddają się konfabulacjom, lecz wymagałoby to mocnej akceleracji sprzętowej na poziomie klastra GPU (VRAM).
3. **Zewnętrzna wyszukiwarka geolokalizacyjna (Metadane mapowe np. OpenStreetMap/Photon)**: Całkowite odciążenie sztucznej inteligencji poprzez odpytanie bazy mapowej dla konkretnego wielokąta zdefiniowanego jako budynek (np. ekstrakcja tagu `building:height` lub `building:levels` za pomocą rozszerzonego lokalnego serwera Nominatim / Photon).
