# Status Projektu

Obecnie projekt to aplikacja w Django. Wstępna konfiguracja została ukończona. Wdrożono wstępnie bazę danych PostgreSQL oraz podstawowe endpointy API.

## Struktura Katalogów i Plików

### Gdzie co dodawać?

* Modele ML: Dodawać w `ml_engine/`.
* Zdjęcia od użytkowników: Zapisywane są automatycznie w `media/photos/`.
* Nowe endpointy API: Tworzyć w `api/` (odpowiednio w `views.py`, `serializers.py` i `urls.py`).
* Nowe modele ORM: Definiować w `api/models.py`.

### Drzewo projektu

* `.env` - Ukryty plik ze zmiennymi środowiskowymi (hasła do DB, klucze, itp.). Git go ignoruje! Tutaj konfigurujesz dostęp do PostgreSQL.
* `api/` - Aplikacja odpowiedzialna za interfejs REST. Przetwarza żądania, zapisuje do bazy i przekazuje zadania do `ml_engine`, znajduje się tu:
  - `models.py` - modele bazy danych dla api.
  - `serializers.py` - dostosowywanie, "tłumaczenie" danych prodókowanych przez ml na JSON.
  - `urls.py` - definiowanie URL pod którym "wysyła się" zdjęcia do analizy.
  - `views.py` - odbiera zdjęcia, "zleca" ich analize i oddaje wynik.
* `config/` - Katalog konfiguracyjny.
* `media/photos/` - Miejsce docelowe dla przesyłanych zdjęć.
* `ml_engine/` - Aplikacja trzymająca logikę eksperymentalnych usług ML.
  Poszczególne pilki w folderze są tylko mockami faktycnzej logiki i usług które zaimplementujemy, kolejno:
  - `alpr.py` (automatic licence plate recognition) - rozpoznawanie tablic rejestracyjnych.
  - `forensics.py` - analiza metadanych.
  - `geolocator.py` - rozpoznawanie geolokalizacji.
  - `object_det.py` - detekcja i rozpoznawanie obiektów.
  - `pipeline.py` - plik który realizuje po kolei usługi (przeprowadza pełną analizę) na jednym zdjęciu (albo na zbiorze zdjęć ale po kolei nie na wszytkich na raz).
* `manage.py` - Główne narzędzie wiersza poleceń Django.
* `requirements.txt` - Plik z zależnościami bibliotek.

