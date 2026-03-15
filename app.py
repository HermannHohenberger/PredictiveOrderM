from typing import Tuple

import holidays
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


REQUIRED_COLUMNS = {
    "ARTIKEL",
    "AUFTRAG",
    "POSITION",
    "KUNDE",
    "WUNSCHLIEFERTERMIN",
    "LIEFERTERMIN",
    "WUNSCHMENGE",
}


def read_orders(uploaded_file, file_name: str | None = None) -> pd.DataFrame:
    detected_name = file_name
    if detected_name is None and hasattr(uploaded_file, "name"):
        detected_name = str(uploaded_file.name)
    file_name_lower = (detected_name or "").lower()

    if file_name_lower.endswith(".csv"):
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    if file_name_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Bitte eine CSV- oder Excel-Datei hochladen.")


def validate_columns(df: pd.DataFrame) -> Tuple[bool, set]:
    normalized_columns = {c.strip().upper() for c in df.columns}
    missing = REQUIRED_COLUMNS - normalized_columns
    return len(missing) == 0, missing


def prepare_orders_data(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df.columns = [c.strip().upper() for c in working_df.columns]
    working_df["ARTIKEL"] = working_df["ARTIKEL"].astype(str).str.strip()
    working_df["ARTIKELGRUPPE"] = working_df["ARTIKEL"].str[:10]

    working_df["WUNSCHLIEFERTERMIN"] = pd.to_datetime(
        working_df["WUNSCHLIEFERTERMIN"], errors="coerce", infer_datetime_format=True
    )
    working_df["WUNSCHMENGE"] = pd.to_numeric(
        working_df["WUNSCHMENGE"], errors="coerce"
    )

    working_df = working_df.dropna(subset=["WUNSCHLIEFERTERMIN", "WUNSCHMENGE"])

    iso = working_df["WUNSCHLIEFERTERMIN"].dt.isocalendar()
    working_df["JAHR"] = iso.year.astype(int)
    working_df["KW"] = iso.week.astype(int)

    return working_df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["JAHR", "KW"], as_index=False)["WUNSCHMENGE"]
        .sum()
        .rename(columns={"WUNSCHMENGE": "GESAMTMENGE"})
        .sort_values(["JAHR", "KW"])
    )

    return weekly


def build_weekly_calendar_features(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    all_days = pd.date_range(start=start_date.normalize(), end=end_date.normalize(), freq="D")
    years = sorted(set(all_days.year.tolist()))
    de_holidays = holidays.Germany(years=years)

    calendar_df = pd.DataFrame({"datum": all_days})
    calendar_df["is_weekday"] = calendar_df["datum"].dt.weekday < 5
    calendar_df["is_holiday"] = calendar_df["datum"].dt.date.map(lambda d: d in de_holidays)
    calendar_df["is_workday"] = calendar_df["is_weekday"] & ~calendar_df["is_holiday"]

    iso = calendar_df["datum"].dt.isocalendar()
    calendar_df["jahr"] = iso.year.astype(int)
    calendar_df["kw"] = iso.week.astype(int)

    weekly_features = (
        calendar_df.groupby(["jahr", "kw"], as_index=False)
        .agg(
            arbeitstage=("is_workday", "sum"),
            anzahl_feiertage=("is_holiday", "sum"),
        )
        .sort_values(["jahr", "kw"])
    )

    weekly_features["arbeitstage"] = weekly_features["arbeitstage"].astype(int)
    weekly_features["anzahl_feiertage"] = weekly_features["anzahl_feiertage"].astype(int)
    return weekly_features


def iso_week_start_date(year: int, week: int) -> pd.Timestamp:
    return pd.Timestamp.fromisocalendar(int(year), int(week), 1)


def build_prophet_training_data(
    weekly_df: pd.DataFrame, calendar_features_df: pd.DataFrame
) -> pd.DataFrame:
    model_df = weekly_df.copy()
    model_df["ds"] = model_df.apply(
        lambda row: iso_week_start_date(row["JAHR"], row["KW"]), axis=1
    )
    model_df = model_df.rename(columns={"GESAMTMENGE": "y"})

    features_df = calendar_features_df.rename(columns={"jahr": "JAHR", "kw": "KW"})
    model_df = model_df.merge(
        features_df[["JAHR", "KW", "arbeitstage", "anzahl_feiertage"]],
        on=["JAHR", "KW"],
        how="left",
    )
    model_df = model_df.dropna(subset=["ds", "y", "arbeitstage", "anzahl_feiertage"])
    model_df = model_df.sort_values("ds")
    return model_df


def build_future_regressors(
    last_ds: pd.Timestamp, periods: int
) -> pd.DataFrame:
    future_ds = pd.date_range(start=last_ds + pd.Timedelta(weeks=1), periods=periods, freq="W-MON")
    future_df = pd.DataFrame({"ds": future_ds})

    min_future = future_df["ds"].min()
    max_future = future_df["ds"].max() + pd.Timedelta(days=6)
    future_features = build_weekly_calendar_features(min_future, max_future)
    future_features["ds"] = future_features.apply(
        lambda row: iso_week_start_date(row["jahr"], row["kw"]), axis=1
    )
    future_df = future_df.merge(
        future_features[["ds", "arbeitstage", "anzahl_feiertage"]], on="ds", how="left"
    )
    future_df["arbeitstage"] = future_df["arbeitstage"].fillna(5)
    future_df["anzahl_feiertage"] = future_df["anzahl_feiertage"].fillna(0)
    return future_df


def compute_error_metrics(actual: pd.Series, forecast: pd.Series) -> tuple[float, float]:
    abs_error = (actual - forecast).abs()
    mae = float(abs_error.mean())

    non_zero = actual != 0
    if non_zero.any():
        mape = float((abs_error[non_zero] / actual[non_zero].abs()).mean() * 100)
    else:
        mape = float("nan")
    return mae, mape


def compute_customer_plannability(prepared_df: pd.DataFrame) -> pd.DataFrame:
    customer_weekly = (
        prepared_df.groupby(["KUNDE", "JAHR", "KW"], as_index=False)["WUNSCHMENGE"]
        .sum()
        .rename(columns={"WUNSCHMENGE": "menge_kw"})
    )

    ranking = (
        customer_weekly.groupby("KUNDE", as_index=False)
        .agg(
            wochen_mit_daten=("menge_kw", "count"),
            mittelwert_kw=("menge_kw", "mean"),
            std_kw=("menge_kw", "std"),
        )
        .fillna({"std_kw": 0.0})
    )

    ranking = ranking[ranking["mittelwert_kw"] > 0].copy()
    ranking["cv"] = ranking["std_kw"] / ranking["mittelwert_kw"]
    ranking["abweichung_prozent"] = ranking["cv"] * 100
    ranking = ranking.sort_values(["cv", "wochen_mit_daten"], ascending=[True, False])
    return ranking


def detect_outliers_zscore(
    series: pd.Series, window: int, threshold: float
) -> pd.DataFrame:
    df = pd.DataFrame({"y": series}).copy()
    min_periods = max(3, min(window, len(df)))

    # Shift(1): Statistik aus Vergangenheit, damit die aktuelle Woche neutral bewertet wird.
    df["roll_mean"] = (
        df["y"].shift(1).rolling(window=window, min_periods=min_periods).mean()
    )
    df["roll_std"] = (
        df["y"].shift(1).rolling(window=window, min_periods=min_periods).std()
    )

    fallback_std = float(df["y"].std(ddof=1)) if len(df) > 1 else 0.0
    if pd.isna(fallback_std) or fallback_std == 0:
        fallback_std = 1.0

    df["roll_std"] = df["roll_std"].replace(0, pd.NA).fillna(fallback_std)
    df["roll_mean"] = df["roll_mean"].fillna(df["y"].expanding().mean().shift(1))
    df["roll_mean"] = df["roll_mean"].fillna(df["y"].mean())

    df["z_score"] = (df["y"] - df["roll_mean"]) / df["roll_std"]
    df["is_outlier"] = df["z_score"].abs() > threshold
    df["is_outlier"] = df["is_outlier"].fillna(False)
    df["upper_band"] = df["roll_mean"] + threshold * df["roll_std"]
    df["lower_band"] = df["roll_mean"] - threshold * df["roll_std"]
    return df


def build_autocorr_table(series: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for lag in range(1, max_lag + 1):
        rows.append({"lag": lag, "autokorrelation": series.autocorr(lag=lag)})
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="PredictiveOrder - Data Ingestion", layout="wide")
    st.title("PredictiveOrder: Schritt 1 - Data Ingestion")
    st.caption(
        "CSV/Excel hochladen, Daten validieren und die Gesamtmenge pro ISO-Kalenderwoche aggregieren."
    )

    uploaded_file = st.file_uploader(
        "Auftragsdatei hochladen", type=["csv", "xlsx", "xls"]
    )

    source_df = None
    source_label = None

    if uploaded_file is not None:
        source_df = read_orders(uploaded_file)
        source_label = uploaded_file.name

    if source_df is None:
        st.info("Bitte eine Datei hochladen.")
        return

    st.subheader(f"Geladene Daten: {source_label}")
    st.dataframe(source_df.head(20), use_container_width=True)

    is_valid, missing = validate_columns(source_df)
    if not is_valid:
        st.error(f"Fehlende Spalten: {', '.join(sorted(missing))}")
        st.stop()

    prepared_df = prepare_orders_data(source_df)

    st.subheader("Saisonalität (Heatmap)")
    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

    kunden = sorted(prepared_df["KUNDE"].dropna().astype(str).unique().tolist())
    artikel = sorted(prepared_df["ARTIKEL"].dropna().astype(str).unique().tolist())
    artikelgruppen = sorted(
        prepared_df["ARTIKELGRUPPE"].dropna().astype(str).unique().tolist()
    )

    selected_kunden = filter_col_1.multiselect(
        "Filter KUNDE", options=kunden, default=[]
    )
    selected_artikel = filter_col_2.multiselect(
        "Filter ARTIKEL", options=artikel, default=[]
    )
    selected_artikelgruppen = filter_col_3.multiselect(
        "Filter ARTIKELGRUPPE (erste 10 Stellen)",
        options=artikelgruppen,
        default=[],
    )

    filtered_df = prepared_df.copy()
    if selected_kunden:
        filtered_df = filtered_df[
            filtered_df["KUNDE"].astype(str).isin(selected_kunden)
        ]
    if selected_artikel:
        filtered_df = filtered_df[
            filtered_df["ARTIKEL"].astype(str).isin(selected_artikel)
        ]
    if selected_artikelgruppen:
        filtered_df = filtered_df[
            filtered_df["ARTIKELGRUPPE"].astype(str).isin(selected_artikelgruppen)
        ]

    if filtered_df.empty:
        st.warning("Keine Daten für die aktuelle Filterauswahl.")
        return

    weekly_df = aggregate_weekly(filtered_df)

    st.success("Spaltenprüfung erfolgreich. Alle Auswertungen nutzen die aktuelle Filterauswahl.")
    st.subheader("Aggregierte Mengen pro Jahr/KW")
    st.dataframe(weekly_df, use_container_width=True)

    heatmap_df = (
        filtered_df.groupby(["JAHR", "KW"], as_index=False)["WUNSCHMENGE"]
        .sum()
        .rename(columns={"WUNSCHMENGE": "GESAMTMENGE"})
    )
    pivot = (
        heatmap_df.pivot(index="JAHR", columns="KW", values="GESAMTMENGE")
        .fillna(0)
        .sort_index()
    )
    pivot = pivot.reindex(columns=range(1, 53), fill_value=0)

    fig = px.imshow(
        pivot,
        labels={"x": "Kalenderwoche (KW)", "y": "Jahr", "color": "Menge"},
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig.update_xaxes(tickmode="linear")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Aggregation als CSV herunterladen",
        data=weekly_df.to_csv(index=False).encode("utf-8"),
        file_name="weekly_aggregation.csv",
        mime="text/csv",
    )

    st.subheader("Kalender-Features je Kalenderwoche")
    min_date = filtered_df["WUNSCHLIEFERTERMIN"].min()
    max_date = filtered_df["WUNSCHLIEFERTERMIN"].max()
    calendar_features_df = build_weekly_calendar_features(min_date, max_date)

    st.caption(
        f"Berechnet für den Zeitraum {min_date.date()} bis {max_date.date()} "
        "(Deutschland, Mo-Fr minus Feiertage)."
    )
    st.dataframe(calendar_features_df, use_container_width=True)
    st.download_button(
        "Kalender-Features als CSV herunterladen",
        data=calendar_features_df.to_csv(index=False).encode("utf-8"),
        file_name="calendar_features_kw.csv",
        mime="text/csv",
    )

    st.subheader("Prognose (Schritt 4: Prophet)")
    horizon_weeks = st.slider(
        "Prognosehorizont (Wochen)", min_value=4, max_value=24, value=12
    )

    if Prophet is None:
        st.warning(
            "Prophet ist nicht installiert. Bitte `pip install -r requirements.txt` ausführen "
            "und die App neu starten."
        )
        return

    model_input_df = build_prophet_training_data(weekly_df, calendar_features_df)
    if len(model_input_df) < 10:
        st.warning("Zu wenige Wochen für ein stabiles Prophet-Modell (mindestens 10 empfohlen).")
        return

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )
    model.add_regressor("arbeitstage")
    model.add_regressor("anzahl_feiertage")
    model.fit(model_input_df[["ds", "y", "arbeitstage", "anzahl_feiertage"]])

    future_df = build_future_regressors(model_input_df["ds"].max(), horizon_weeks)
    forecast_df = model.predict(future_df[["ds", "arbeitstage", "anzahl_feiertage"]])

    forecast_output = forecast_df[
        ["ds", "yhat", "yhat_lower", "yhat_upper", "arbeitstage", "anzahl_feiertage"]
    ].copy()
    forecast_output["jahr"] = forecast_output["ds"].dt.isocalendar().year.astype(int)
    forecast_output["kw"] = forecast_output["ds"].dt.isocalendar().week.astype(int)
    st.dataframe(
        forecast_output[
            ["jahr", "kw", "ds", "yhat", "yhat_lower", "yhat_upper", "arbeitstage", "anzahl_feiertage"]
        ],
        use_container_width=True,
    )

    chart_df_hist = model_input_df[["ds", "y"]].copy()
    chart_df_hist["jahr"] = chart_df_hist["ds"].dt.isocalendar().year.astype(int)
    chart_df_hist["kw"] = chart_df_hist["ds"].dt.isocalendar().week.astype(int)
    chart_df_fc = forecast_output[
        ["ds", "yhat", "yhat_lower", "yhat_upper", "jahr", "kw"]
    ].copy()
    fig_forecast = go.Figure()
    fig_forecast.add_trace(
        go.Scatter(
            x=chart_df_hist["ds"],
            y=chart_df_hist["y"],
            mode="lines",
            name="Ist-Werte",
            customdata=chart_df_hist[["jahr", "kw"]].to_numpy(),
            hovertemplate="Datum: %{x|%d.%m.%Y}<br>KW: %{customdata[1]} (%{customdata[0]})<br>Menge: %{y:,.0f}<extra></extra>",
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=chart_df_fc["ds"],
            y=chart_df_fc["yhat"],
            mode="lines",
            name="Prognose (yhat)",
            customdata=chart_df_fc[["jahr", "kw"]].to_numpy(),
            hovertemplate="Datum: %{x|%d.%m.%Y}<br>KW: %{customdata[1]} (%{customdata[0]})<br>Prognose: %{y:,.0f}<extra></extra>",
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=chart_df_fc["ds"],
            y=chart_df_fc["yhat_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=chart_df_fc["ds"],
            y=chart_df_fc["yhat_lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            name="Konfidenzintervall",
            hoverinfo="skip",
        )
    )
    fig_forecast.update_layout(
        xaxis_title="Datum (Wochenstart)",
        yaxis_title="WUNSCHMENGE",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.download_button(
        "Prognose als CSV herunterladen",
        data=forecast_output.to_csv(index=False).encode("utf-8"),
        file_name="forecast_12_weeks.csv",
        mime="text/csv",
    )

    st.subheader("Rückblickender Modelltest (Schritt 5)")
    max_backtest_weeks = min(24, len(model_input_df) - 8)
    if max_backtest_weeks < 4:
        st.info("Zu wenige historische Wochen für den Modelltest verfügbar.")
        return

    backtest_weeks = st.slider(
        "Testzeitraum (letzte Wochen)",
        min_value=4,
        max_value=max_backtest_weeks,
        value=min(12, max_backtest_weeks),
    )

    train_df = model_input_df.iloc[:-backtest_weeks].copy()
    test_df = model_input_df.iloc[-backtest_weeks:].copy()

    backtest_model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )
    backtest_model.add_regressor("arbeitstage")
    backtest_model.add_regressor("anzahl_feiertage")
    backtest_model.fit(train_df[["ds", "y", "arbeitstage", "anzahl_feiertage"]])

    test_pred = backtest_model.predict(
        test_df[["ds", "arbeitstage", "anzahl_feiertage"]]
    )
    backtest_result = test_df[["ds", "y"]].merge(
        test_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
    )
    backtest_result["jahr"] = backtest_result["ds"].dt.isocalendar().year.astype(int)
    backtest_result["kw"] = backtest_result["ds"].dt.isocalendar().week.astype(int)
    backtest_result = backtest_result.rename(columns={"y": "ist"})

    mae, mape = compute_error_metrics(backtest_result["ist"], backtest_result["yhat"])
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("MAE", f"{mae:,.2f}")
    if pd.isna(mape):
        metric_col_2.metric("MAPE", "n/a (Ist-Werte = 0)")
    else:
        metric_col_2.metric("MAPE", f"{mape:.2f}%")

    fig_backtest = go.Figure()
    fig_backtest.add_trace(
        go.Scatter(
            x=backtest_result["ds"],
            y=backtest_result["ist"],
            mode="lines+markers",
            name="Ist-Werte",
            customdata=backtest_result[["jahr", "kw"]].to_numpy(),
            hovertemplate="Datum: %{x|%d.%m.%Y}<br>KW: %{customdata[1]} (%{customdata[0]})<br>Ist: %{y:,.0f}<extra></extra>",
        )
    )
    fig_backtest.add_trace(
        go.Scatter(
            x=backtest_result["ds"],
            y=backtest_result["yhat"],
            mode="lines+markers",
            name="Vorhersage",
            customdata=backtest_result[["jahr", "kw"]].to_numpy(),
            hovertemplate="Datum: %{x|%d.%m.%Y}<br>KW: %{customdata[1]} (%{customdata[0]})<br>Vorhersage: %{y:,.0f}<extra></extra>",
        )
    )
    fig_backtest.add_trace(
        go.Scatter(
            x=backtest_result["ds"],
            y=backtest_result["yhat_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_backtest.add_trace(
        go.Scatter(
            x=backtest_result["ds"],
            y=backtest_result["yhat_lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            name="Konfidenzintervall",
            hoverinfo="skip",
        )
    )
    fig_backtest.update_layout(
        xaxis_title="Datum (Wochenstart)",
        yaxis_title="WUNSCHMENGE",
    )
    st.plotly_chart(fig_backtest, use_container_width=True)
    st.dataframe(
        backtest_result[
            ["jahr", "kw", "ds", "ist", "yhat", "yhat_lower", "yhat_upper"]
        ],
        use_container_width=True,
    )

    st.subheader("Planbarkeits-Ranking pro Kunde")
    plannability_df = compute_customer_plannability(filtered_df)
    if plannability_df.empty:
        st.info("Keine ausreichenden Kundendaten für CV-Ranking vorhanden.")
        return

    ranking_view = plannability_df.copy()
    ranking_view["cv"] = ranking_view["cv"].round(4)
    ranking_view["abweichung_prozent"] = ranking_view["abweichung_prozent"].round(2)
    ranking_view["mittelwert_kw"] = ranking_view["mittelwert_kw"].round(2)
    ranking_view["std_kw"] = ranking_view["std_kw"].round(2)
    st.dataframe(
        ranking_view[
            [
                "KUNDE",
                "wochen_mit_daten",
                "mittelwert_kw",
                "std_kw",
                "cv",
                "abweichung_prozent",
            ]
        ],
        use_container_width=True,
    )

    customer_options = ranking_view["KUNDE"].astype(str).tolist()
    selected_customer = st.selectbox(
        "Kunde für Detailaussage", options=customer_options, index=0
    )

    selected_row = ranking_view[ranking_view["KUNDE"].astype(str) == selected_customer].iloc[0]
    st.success(
        f'Kunde {selected_customer} hat eine Planbarkeits-Abweichung '
        f'gegenüber dem KW-Schnitt von {selected_row["abweichung_prozent"]:.2f}%.'
    )

    customer_ts = (
        filtered_df[filtered_df["KUNDE"].astype(str) == selected_customer]
        .groupby(["JAHR", "KW"], as_index=False)["WUNSCHMENGE"]
        .sum()
        .sort_values(["JAHR", "KW"])
    )
    customer_ts["ds"] = customer_ts.apply(
        lambda row: iso_week_start_date(row["JAHR"], row["KW"]), axis=1
    )
    customer_ts["kw_schnitt"] = customer_ts["WUNSCHMENGE"].mean()

    fig_customer = go.Figure()
    fig_customer.add_trace(
        go.Scatter(
            x=customer_ts["ds"],
            y=customer_ts["WUNSCHMENGE"],
            mode="lines+markers",
            name="Wochenmenge",
        )
    )
    fig_customer.add_trace(
        go.Scatter(
            x=customer_ts["ds"],
            y=customer_ts["kw_schnitt"],
            mode="lines",
            name="KW-Schnitt",
        )
    )
    fig_customer.update_layout(
        xaxis_title="Datum (Wochenstart)",
        yaxis_title="WUNSCHMENGE",
    )
    st.plotly_chart(fig_customer, use_container_width=True)

    st.subheader("Muster-Check (Zeitreihe)")
    pattern_scope = st.radio(
        "Analysebasis",
        options=["Gesamtreihe", "Ausgewählter Kunde"],
        horizontal=True,
    )

    if pattern_scope == "Gesamtreihe":
        pattern_df = weekly_df.copy().sort_values(["JAHR", "KW"])
        pattern_df["ds"] = pattern_df.apply(
            lambda row: iso_week_start_date(row["JAHR"], row["KW"]), axis=1
        )
        pattern_df["y"] = pattern_df["GESAMTMENGE"]
        scope_label = "Gesamt"
    else:
        pattern_df = customer_ts.copy().sort_values(["JAHR", "KW"])
        pattern_df["y"] = pattern_df["WUNSCHMENGE"]
        scope_label = f"Kunde {selected_customer}"

    if len(pattern_df) < 8:
        st.info("Zu wenig Historie für Muster-Check (mindestens 8 Wochen).")
        return

    col_lag, col_window, col_threshold = st.columns(3)
    max_lag = col_lag.slider(
        "Maximale Verzögerung (Lag)",
        min_value=4,
        max_value=24,
        value=12,
        help="Größte Anzahl an Wochenverschiebungen, für die die Autokorrelation berechnet wird.",
    )
    roll_window = col_window.slider(
        "Rollierendes Fenster",
        min_value=3,
        max_value=16,
        value=6,
        help="Anzahl der Wochen, über die gleitender Mittelwert und Streuung berechnet werden.",
    )
    z_threshold = col_threshold.slider(
        "Ausreißer-Schwelle |Z|",
        min_value=1.5,
        max_value=4.0,
        value=2.5,
        step=0.1,
        help="Grenzwert für den Z-Score: Werte oberhalb dieser Schwelle werden als Ausreißer markiert.",
    )

    series = pattern_df["y"].astype(float).reset_index(drop=True)
    max_lag = min(max_lag, len(series) - 1)
    autocorr_df = build_autocorr_table(series, max_lag=max_lag)
    if not autocorr_df.empty:
        strongest_row = autocorr_df.iloc[autocorr_df["autokorrelation"].abs().idxmax()]
        st.caption(
            f"Stärkste Verzögerung für {scope_label}: "
            f"{int(strongest_row['lag'])} Wochen (Autokorrelation={strongest_row['autokorrelation']:.3f})"
        )

    fig_acf = px.bar(
        autocorr_df,
        x="lag",
        y="autokorrelation",
        title=f"Autokorrelation nach Verzögerung ({scope_label})",
    )
    st.plotly_chart(fig_acf, use_container_width=True)

    outlier_df = detect_outliers_zscore(series, window=roll_window, threshold=z_threshold)
    plot_df = pattern_df.copy().reset_index(drop=True)
    plot_df["roll_mean"] = outlier_df["roll_mean"]
    plot_df["roll_std"] = outlier_df["roll_std"]
    plot_df["is_outlier"] = outlier_df["is_outlier"]
    plot_df["upper_band"] = outlier_df["upper_band"]
    plot_df["lower_band"] = outlier_df["lower_band"]

    fig_pattern = go.Figure()
    fig_pattern.add_trace(
        go.Scatter(
            x=plot_df["ds"],
            y=plot_df["y"],
            mode="lines+markers",
            name="Menge",
        )
    )
    fig_pattern.add_trace(
        go.Scatter(
            x=plot_df["ds"],
            y=plot_df["roll_mean"],
            mode="lines",
            name=f"Rollierender Mittelwert ({roll_window})",
        )
    )
    fig_pattern.add_trace(
        go.Scatter(
            x=plot_df["ds"],
            y=plot_df["upper_band"],
            mode="lines",
            line={"width": 1, "dash": "dot"},
            name=f"Obere Schwelle (|Z|={z_threshold})",
        )
    )
    fig_pattern.add_trace(
        go.Scatter(
            x=plot_df["ds"],
            y=plot_df["lower_band"],
            mode="lines",
            line={"width": 1, "dash": "dot"},
            name=f"Untere Schwelle (|Z|={z_threshold})",
        )
    )
    outliers_plot = plot_df[plot_df["is_outlier"]]
    if not outliers_plot.empty:
        fig_pattern.add_trace(
            go.Scatter(
                x=outliers_plot["ds"],
                y=outliers_plot["y"],
                mode="markers",
                marker={"size": 10, "symbol": "x"},
                name="Ausreißer",
            )
        )

    fig_pattern.update_layout(
        xaxis_title="Datum (Wochenstart)",
        yaxis_title="Menge",
    )
    st.plotly_chart(fig_pattern, use_container_width=True)
    st.caption(f"Erkannte Ausreißer: {int(plot_df['is_outlier'].sum())}")


if __name__ == "__main__":
    main()
