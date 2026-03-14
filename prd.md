Product Requirements Document (PRD): Project "PredictiveOrder"
1. Vision & Zielsetzung

Problem: Kunden bestellen kurzfristig ("on-demand"), was zu Lastspitzen und Leerläufen in der Produktion führt. Kunden behaupten, ihr Bedarf sei unvorhersehbar.
Hypothese: Die historischen Daten zeigen eine starke Korrelation mit Kalenderwochen (Saisonalität). Das Kaufverhalten ist vorhersehbarer als von Kunden kommuniziert.
Ziel: Eine Web-Anwendung, die basierend auf historischen Daten und externen Faktoren eine Bedarfsprognose (Rolling Forecast) für die nächsten 4–12 Wochen erstellt, um die Produktionsplanung zu optimieren.
2. Zielgruppe

    Produktionsplaner: Benötigen aggregierte Zahlen für Kapazitätsplanung.

    Einkauf: Benötigt Vorhersagen für Rohmaterialbestellungen.

    Geschäftsführung: Benötigt Dashboards zur Auslastungserwartung.

3. Datenstrategie
3.1 Interne Daten (Bestand)

    Auftragshistorie: Zeitstempel (Bestelldatum/Wunschlieferdatum), Menge, Artikelnummer, Artikelgruppe, Kunden-ID.

    Stammdaten: Kundenstandorte, Produktkategorien.

3.2 Externe Datenquellen (Integration)

Um die Genauigkeit zu erhöhen, soll die App folgende Daten einbeziehen:

    Kalender-Modul: Feiertage (bundesweit/regional), Schulferien, Brückentage (Python-Library: holidays).

    Arbeitstage: Netto-Produktionstage pro Woche (berechnet).

    Wirtschaftsindikatoren (Optional via API): z.B. IFO-Index oder branchenspezifische Indizes (via Quandl oder Yahoo Finance).

    Wetterdaten (Optional): Nur falls Korrelation zu Produkten besteht (via OpenWeatherMap API).

4. Funktionale Anforderungen (Features)
Phase 1: Daten-Engine & Analyse

    Daten-Upload: CSV/Excel-Upload von historischen Aufträgen.

    EDA-Modul (Exploratory Data Analysis): Automatische Visualisierung von Saisonalität pro KW über die Jahre hinweg (Heatmaps).

    Anomalie-Erkennung: Identifikation von Einmaleffekten (z.B. Großprojekt eines Kunden), die die Prognose nicht verzerren sollen.

Phase 2: Forecasting Modell

    Baseline: "Last Year Same Week" Modell.

    Advanced: Integration von Facebook Prophet oder XGBoost, um Trends und Saisonalitäten (KW) automatisch zu lernen.

    Features: Das Modell muss "Feiertage" als Regressoren berücksichtigen.

Phase 3: UI & Output

    Forecast-Dashboard: Darstellung der Vorhersage vs. Vorjahr.

    Konfidenzintervalle: Anzeige von "Best Case / Worst Case" Szenarien für die Produktion.

    Export: CSV-Export der prognostizierten Mengen pro Artikelgruppe/KW für das ERP-System.

5. Technischer Stack (Empfehlung für Cursor)

    Frontend/Backend: Streamlit (perfekt für Data Apps und schnelles Prototyping in Python).

    Datenverarbeitung: Pandas, NumPy.

    Forecasting: prophet (robust für Business-Daten) oder skforecast.

    Visualisierung: Plotly (interaktive Charts).

6. Implementierungs-Roadmap für Cursor (Anweisungen)

Du kannst die folgenden Blöcke nacheinander in Cursor eingeben, um die App zu bauen:
Schritt 1: Grundstruktur & Data Ingestion

    "Erstelle eine Streamlit-App, die eine CSV-Datei mit Auftragsdaten entgegennimmt (Trennzeichen ';'). Die Datei enthält die Spalten: 'ARTIKEL', 'AUFTRAG', 'POSITION', 'KUNDE', 'WUNSCHLIEFERTERMIN', 'LIEFERTERMIN', 'WUNSCHMENGE'. Nutze zunächst 'WUNSCHLIEFERTERMIN' als Datumsfeld und 'WUNSCHMENGE' als Zielgröße. Erstelle eine Funktion, die das Datum in ISO-Kalenderwochen (KW) und Jahre umwandelt und die Gesamtmenge pro KW aggregiert."

Schritt 2: Visualisierung der Saisonalität

    "Erstelle eine Heatmap-Visualisierung (mit Plotly), die die Jahre auf der Y-Achse und die Kalenderwochen (1-52) auf der X-Achse zeigt. Die Farbe repräsentiert die aggregierte WUNSCHMENGE. Füge zunächst Filter für 'KUNDE' und 'ARTIKEL' hinzu. Falls später eine Artikelgruppierung verfügbar ist, ergänze optional einen Filter für 'Artikelgruppe'."

Schritt 3: Integration von Feiertagen

    "Nutze die Library python-holidays, um für einen gegebenen Zeitraum in Deutschland alle Feiertage zu markieren. Erstelle eine Feature-Tabelle, die für jede KW die Anzahl der Arbeitstage (Mo-Fr minus Feiertage) berechnet und je Woche mindestens die Felder 'jahr', 'kw', 'arbeitstage', 'anzahl_feiertage' enthält."

Schritt 4: Das Prognose-Modell

    "Implementiere ein Forecasting-Modell mit prophet. Trainiere es auf den wöchentlich aggregierten Daten (Zielgröße: WUNSCHMENGE). Berücksichtige die berechneten Feiertage und Arbeitstage als zusätzliche Regressoren. Erstelle eine Vorhersage für die nächsten 12 Wochen inklusive Konfidenzintervallen (yhat_lower/yhat_upper) und gib die Prognose als Tabelle plus Plot aus."

Schritt 5: Vergleich & Validierung

    "Erstelle ein Backtesting für einen definierten historischen Zeitraum (z.B. letzte 12 verfügbare Wochen) und visualisiere Ist-Werte vs. Modellvorhersage im selben Chart. Ergänze mindestens MAE und MAPE als Gütekennzahlen, um die Prognosegüte nachvollziehbar zu demonstrieren."

7. Besondere Berücksichtigung: "Der unplanbare Kunde"

Um dein Argument gegenüber der Geschäftsführung zu stärken, sollte die App ein "Planbarkeits-Ranking" enthalten:

    Berechne für jeden Kunden den Coefficient of Variation (CV = Standardabweichung / Mittelwert) auf Basis der wöchentlichen Mengen und erstelle daraus ein Ranking.

    Zeige eine interpretierbare Aussage, z.B.: "Kunde X behauptet, er sei unplanbar, aber seine Abweichung vom KW-Schnitt beträgt nur 12% (CV=0,12)." -> Das ist ein starkes Argument für eine Vorproduktion.
