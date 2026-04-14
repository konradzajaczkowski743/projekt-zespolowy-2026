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
