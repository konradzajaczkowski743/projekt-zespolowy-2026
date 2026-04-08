##  Uruchomienie

### 1. Uruchom kontenery
```bash
docker-compose up --build 
```

### 2. Uruchomienie w tle (detached mode)

```bash
docker-compose up -d
```

Aby wyświetlić logi:
```bash
docker-compose logs -f web
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
```

Hasło: `postgres` (z `.env`)

## 📦 Serwisy

- **web**: Django aplikacja na `http://localhost:8000`
- **db**: PostgreSQL na `localhost:5432`

##  Konfiguracja

Zmienne środowiskowe znajdują się w pliku `.env`:

```env
DEBUG=True
SECRET_KEY=sekretny_klucz_6_7
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
```

Zmień wartości w `.env` i zrestartuj kontenery:
```bash
docker-compose down
docker-compose up -d
```

##  Troubleshooting

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
