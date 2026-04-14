##  Uruchomienie

## cuda toolkit

sprawdź czy masz kompatybiną [kartę tutaj](https://developer.nvidia.com/cuda/gpus)

jest to potrzebne przed odpaleniem docker compose(szybkość pracy modelu)
sterownik karty ma być 531+ sprawdzisz tą komendą

```bash
  nvidia-smi 
```
## o ile masz docker desktop i windows 11
trzeba wejść w cmd i wpisać:
```bash 
wsl -d ubuntu #nie musi być ubuntu ale no

```
następnie wklejasz to jak masz to ubuntu/debian
to jest twój **1** krok 
 
```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

```
po instalce 
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```
i zrestartuj dockera. szybki test czy działa
```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```
powinno wypisać to sterowniki karty twojej karty jak wszystko
jest git to możesz usunąć ten image/kontener

### 1. Uruchom kontenery
jeśli zrobiłeś\aś wcześniejsze kroki została jedna rzecz 
w docker-compose.yaml w gemma jest zakomentowana zmienna 
usuń aby użyć gpu
```bash
    # gpus: all
```
```bash
docker-compose up --build 
```

### 2. Uruchomienie w tle (detached mode)

```bash
docker-compose up -d
```

Aby wyświetlić logi:
```bash
docker-compose logs -f web #nie musi być web to jest przykład
```

## Gemma 4 e2b
odpalenie trochę może trwać więc wrazie czego zmień zmienne czasowe timeoutu 
testowanie jest proste jak pałki lodowe. Wklej do cmd po odpaleniu kontenera  poczekaj chwilę i gem się przywita
```bash
  docker exec -it projekt_gemma ollama run gemma4:e2b "Hello"
```
##  Testowanie API

### API Root - lista dostępnych endpointów

```bash
curl http://localhost:8000/api/v1/
```

Odpowiedź:
```json
{
  "message": "Projekt API",
  "version": "1.0",
  "endpoints": {
    "analysis": "/api/v1/analysis/"
  }
}
```


##  Zatrzymanie kontenerów

```bash
docker-compose down
```

Aby usunąć również dane bazy danych:
```bash
docker-compose down -v
```

##  Przydatne komendy

| Komenda | Opis |
|---------|------|
| `docker-compose ps` | Lista uruchomionych kontenerów |
| `docker-compose logs web` | Pokaż logi aplikacji Django |
| `docker-compose logs db` | Pokaż logi bazy danych |
| `docker-compose exec web bash` | Wejdź w bash wewnątrz kontenera web |
| `docker-compose exec db psql -U postgres` | Połącz się z bazą danych PostgreSQL |
| `docker-compose restart web` | Zrestartuj serwis web |

##  Migracje Django

Krótka instrukcja: jak tworzyć i stosować migracje lokalnie i w Dockerze oraz jak bezpiecznie dodać nienullowalne pola relacyjne.

- **W kontenerze (zalecane)**: uruchamiaj polecenia w środowisku aplikacji, żeby uniknąć problemów z nazwą hosta `db`.

```bash
docker-compose exec web python manage.py makemigrations
docker-compose exec web python manage.py makemigrations api    # tylko dla konkretnej aplikacji
docker-compose exec web python manage.py migrate --noinput
docker-compose exec web python manage.py showmigrations
docker-compose exec web python manage.py sqlmigrate api 0002   # podejrzyj SQL dla migracji
```

- **Lokalnie (host)**: jeśli chcesz robić migracje poza kontenerem, ustaw poprawnie zmienne środowiskowe (DB_HOST=127.0.0.1 lub inny adres) i uruchom z katalogu `project`.

```powershell
#$env:DB_HOST = '127.0.0.1'   # PowerShell
cd project
python manage.py makemigrations
python manage.py migrate
python manage.py showmigrations
```

- **Dodawanie nienullowalnego pola (bez interaktywnego promptu)** — bezpieczna, 2-etapowa strategia:

  1. Tymczasowo dopuść wartości NULL w modelu (np. `request = models.OneToOneField(..., null=True, blank=True, ...)`).
  2. Uruchom `makemigrations` i `migrate` — pole zostanie dodane jako nullable.
  3. Wykonaj backfill danych (migracja danych z RunPython, skrypt `manage.py shell` lub dedykowany management command) żeby wypełnić nowe pole dla istniejących wierszy.
  4. Usuń `null=True, blank=True` z modelu (przywróć nienullowalność) i ponownie `makemigrations` oraz `migrate` — Django nie poprosi już o domyślną wartość.

  Uwaga: jeśli w trakcie `makemigrations` zobaczysz prompt "It is impossible to add a non-nullable field... Please select an option:", wybierz **opcja 2 (Quit)** i zastosuj powyższą strategię. Opcja 1 (one-off default) przypisze jedną wartość do wszystkich istniejących wierszy i zwykle nie jest odpowiednia dla relacji FK/OneToOne.

- **Backfill — szybki przykład (manage.py shell)**:

```python
from api.models import AnalysisResult, UploadedImage
for ar in AnalysisResult.objects.all():
    if getattr(ar, 'image_id', None):
        ui = UploadedImage.objects.filter(pk=ar.image_id).first()
        if ui and ui.request_id:
            ar.request_id = ui.request_id
            ar.save(update_fields=['request'])
```

- **Diagnostyka i przydatne polecenia**:

```bash
docker-compose exec web python manage.py showmigrations
docker-compose exec web python manage.py migrate --plan   # pokaż plan migracji
docker-compose exec db psql -U postgres -d postgres -c "select app, name, applied from django_migrations order by applied desc limit 20;"
```

- **Problemy z hostem `db`**: jeżeli uruchamiasz `manage.py` lokalnie i widzisz błąd "could not translate host name 'db' to address", uruchom polecenia wewnątrz kontenera (`docker-compose exec web ...`) lub ustaw `DB_HOST=127.0.0.1` i upewnij się, że kontener Postgres jest dostępny na porcie 5432.

- **Commituj migracje**: zawsze dodawaj pliki migracji do repozytorium (np. `api/migrations/0002_...`) i otwórz PR — inni członkowie zespołu powinni uruchomić `migrate` po zmergowaniu.

##  Dostęp do bazy danych

### Z hosta (localhost):
```bash
psql -h localhost -U postgres -d postgres
```

### Z kontenera:
```bash
docker-compose exec db psql -U postgres
\dt
```
tabele zainteresowań na razie pierwsze 3 tabele z góry

Hasło: `postgres` (z `.env`)

##  Serwisy

- **web**: Django aplikacja na `http://localhost:8000`
- **db**: PostgreSQL na `localhost:5432`

##  Konfiguracja

Zmienne środowiskowe znajdują się w pliku `.env`:
potem zmienimy ale narazie jest takie coś
```env
DEBUG=True
SECRET_KEY=sekretny_klucz_6_7
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
MODEL_NAME=gemma4:e2b
EXIFTOOL_API=http://exiftool:8001
GEMMA_API=http://gemma:11434
```

Zmień wartości w `.env` i zrestartuj kontenery:
```bash
docker-compose down
docker-compose up -d
```

##  Troubleshooting

### gem coś szpońci 
no to jest problem ale spokojenie wystarczy zrestartować docker desktop
i usunąć kontenery/images (narazie nie ma tam żadnych danych wię możemy tak zrobić) 


### Port 8000 już zajęty
```bash
docker-compose down
# Lub zmień port w docker-compose.yaml: "9000:8000" (zewnętrzny:wewnętrzny)
```

### Port 5432 już zajęty
```bash
# Zmień port w docker-compose.yaml: "5433:5432"
```

### Baza danych nie odpowiada
```bash
docker-compose logs db
docker-compose restart db
```

### Wyczyść wszystko
```bash
docker-compose down -v
docker-compose up --build
```
