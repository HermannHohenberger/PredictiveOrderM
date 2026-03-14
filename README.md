# PredictiveOrder

Streamlit-Prototyp zur Analyse und Prognose von Auftragsmengen auf Kalenderwochen-Basis.

## Voraussetzungen

- Python 3.10+ (lokal)
- Docker + Docker Compose (Server)

## Lokal starten

1. In den Projektordner wechseln:

```bash
cd /Users/Hohenberger/Documents/CLI-Projekte/PredictiveOrderM
```

2. Virtuelle Umgebung erstellen (empfohlen):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

4. Streamlit-App starten:

```bash
streamlit run app.py
```

## Deployment auf Hostinger (Docker + Nginx Proxy)

### 1) Projekt auf den Server bringen

Per `git clone` oder Upload (SFTP/rsync) in ein Zielverzeichnis, z. B.:

```bash
mkdir -p /opt/predictiveorder
cd /opt/predictiveorder
```

### 2) Environment-Datei anlegen

```bash
cp .env.example .env
```

Optional `APP_PORT` in `.env` anpassen (Standard `8501`).

### 3) Container bauen und starten

```bash
docker compose up -d --build
```

### 4) Funktion intern prüfen

```bash
docker compose ps
curl http://127.0.0.1:8501
```

### 5) Subdomain einrichten

Bei deinem DNS-Anbieter (oder Hostinger DNS) für z. B. `predictive.deinedomain.tld`:

- `A`-Record auf die Server-IP setzen.

### 6) Nginx Proxy (z. B. Nginx Proxy Manager)

Proxy Host anlegen:

- **Domain Names**: `predictive.deinedomain.tld`
- **Forward Hostname/IP**: Server-IP (oder Docker-Service, je nach Setup)
- **Forward Port**: `8501`
- **Websockets Support**: aktivieren
- **SSL**: Let's Encrypt aktivieren, Force SSL einschalten

### 7) Update-Workflow

Nach Änderungen:

```bash
cd /opt/predictiveorder
docker compose up -d --build
```

## Hinweise

- In `.dockerignore` ist `assets` ausgeschlossen; entferne den Eintrag, falls du Bilder im Container brauchst.
- Beispiel-Daten sollten nicht ungewollt öffentlich bereitgestellt werden.
