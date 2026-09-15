import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline
from pathlib import Path
from datetime import date
import uuid

# Einstellungen

st.set_page_config(
page_title="Fundkiste",
page_icon="🔎",
layout="centered"
)

APP_NAME = "Fundkiste"
SCHOOL_NAME = "KATHARINEUM ZU LÜBECK"
SCHOOL_YEAR = "seit 1531"

DATA_FOLDER = Path("data")
UPLOAD_FOLDER = DATA_FOLDER / "bilder"
DATA_FILE = DATA_FOLDER / "fundstuecke.csv"

DATA_FOLDER.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

KATEGORIEN = [
"Kleidung",
"Schlüssel",
"Trinkflasche",
"Tasche",
"Handy",
"Schmuck",
"Schulmaterial",
"Sonstiges"
]

# Hugging-Face-KI

@st.cache_resource
def lade_ki():
return pipeline(
"zero-shot-image-classification",
model="openai/clip-vit-base-patch32"
)

def erkenne_kategorie(bild):
ki = lade_ki()

```
kategorien_ki = [
    "clothing",
    "keys",
    "water bottle",
    "bag",
    "mobile phone",
    "jewelry",
    "school supplies",
    "other object"
]

ergebnisse = ki(
    bild,
    candidate_labels=kategorien_ki
)

uebersetzung = {
    "clothing": "Kleidung",
    "keys": "Schlüssel",
    "water bottle": "Trinkflasche",
    "bag": "Tasche",
    "mobile phone": "Handy",
    "jewelry": "Schmuck",
    "school supplies": "Schulmaterial",
    "other object": "Sonstiges"
}

bestes_ergebnis = ergebnisse[0]

kategorie = uebersetzung.get(
    bestes_ergebnis["label"],
    "Sonstiges"
)

sicherheit = bestes_ergebnis["score"] * 100

return kategorie, sicherheit
```

# Daten

def lade_fundstuecke():
if not DATA_FILE.exists():
return pd.DataFrame(
columns=[
"id",
"gegenstand",
"kategorie",
"farbe",
"fundort",
"datum",
"beschreibung",
"bild",
"erledigt"
]
)

```
try:
    return pd.read_csv(DATA_FILE)
except Exception:
    return pd.DataFrame(
        columns=[
            "id",
            "gegenstand",
            "kategorie",
            "farbe",
            "fundort",
            "datum",
            "beschreibung",
            "bild",
            "erledigt"
        ]
    )
```

def speichere_fundstuecke(df):
df.to_csv(
DATA_FILE,
index=False,
encoding="utf-8"
)

# Navigation

if "seite" not in st.session_state:
st.session_state.seite = "start"

if "ki_kategorie" not in st.session_state:
st.session_state.ki_kategorie = "Sonstiges"

# Design

st.markdown(
""" <style>
.stApp {
background-color: #f5f0df;
}

```
h1, h2, h3, p, label {
    color: #111111;
}

.titel {
    text-align: center;
    font-size: 58px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 5px;
}

.schule {
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 2px;
}

.jahr {
    text-align: center;
    font-size: 16px;
    margin-bottom: 35px;
}

.erklaerung {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

div.stButton > button {
    width: 100%;
    min-height: 65px;
    border-radius: 12px;
    border: 2px solid #111111;
    background-color: #d7e85b;
    color: #111111;
    font-size: 22px;
    font-weight: 600;
}

.fundstueck {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #cccccc;
    margin-bottom: 20px;
}
</style>
""",
unsafe_allow_html=True
```

)

# Startseite

if st.session_state.seite == "start":

```
st.markdown(
    '<div class="titel">Fundkiste</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="schule">{SCHOOL_NAME}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="jahr">{SCHOOL_YEAR}</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="erklaerung">'
    'Verlorene Sachen suchen oder gefundene Sachen eintragen.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔎 Suchen", key="start_suchen"):
        st.session_state.seite = "suchen"
        st.rerun()

with col2:
    if st.button("➕ Eingeben", key="start_eingeben"):
        st.session_state.seite = "eingeben"
        st.rerun()
```

# Suche

elif st.session_state.seite == "suchen":

```
if st.button("← Zurück"):
    st.session_state.seite = "start"
    st.rerun()

st.title("🔎 Fundstücke suchen")

st.write(
    "Suche nach einem verlorenen Gegenstand."
)

df = lade_fundstuecke()

if not df.empty and "erledigt" in df.columns:
    df["erledigt"] = df["erledigt"].astype(str)
    df = df[
        df["erledigt"].str.lower() != "true"
    ]

col1, col2 = st.columns(2)

with col1:
    suche_gegenstand = st.text_input(
        "Gegenstand",
        placeholder="z. B. Rucksack"
    )

    suche_farbe = st.text_input(
        "Farbe",
        placeholder="z. B. schwarz"
    )

with col2:
    suche_kategorie = st.selectbox(
        "Kategorie",
        ["Alle"] + KATEGORIEN
    )

    suche_fundort = st.text_input(
        "Fundort",
        placeholder="z. B. Sporthalle"
    )

if st.button("🔎 Suchen", type="primary"):

    if df.empty:
        st.info(
            "Es wurden noch keine Fundstücke eingetragen."
        )

    else:
        ergebnis = df.copy()

        if suche_gegenstand:
            ergebnis = ergebnis[
                ergebnis["gegenstand"]
                .fillna("")
                .str.contains(
                    suche_gegenstand,
                    case=False,
                    na=False
                )
            ]

        if suche_kategorie != "Alle":
            ergebnis = ergebnis[
                ergebnis["kategorie"]
                .fillna("")
                .str.lower()
                == suche_kategorie.lower()
            ]

        if suche_farbe:
            ergebnis = ergebnis[
                ergebnis["farbe"]
                .fillna("")
                .str.contains(
                    suche_farbe,
                    case=False,
                    na=False
                )
            ]

        if suche_fundort:
            ergebnis = ergebnis[
                ergebnis["fundort"]
                .fillna("")
                .str.contains(
                    suche_fundort,
                    case=False,
                    na=False
                )
            ]

        if ergebnis.empty:
            st.warning(
                "Leider wurde kein passendes Fundstück gefunden."
            )

        else:
            st.success(
                f"{len(ergebnis)} Fundstück(e) gefunden."
            )

            for _, fundstueck in ergebnis.iterrows():

                st.markdown(
                    '<div class="fundstueck">',
                    unsafe_allow_html=True
                )

                bild_pfad = fundstueck.get(
                    "bild",
                    ""
                )

                if (
                    pd.notna(bild_pfad)
                    and str(bild_pfad).strip()
                    and Path(str(bild_pfad)).exists()
                ):
                    st.image(
                        str(bild_pfad),
                        use_container_width=True
                    )

                st.subheader(
                    str(
                        fundstueck.get(
                            "gegenstand",
                            "Unbekannter Gegenstand"
                        )
                    )
                )

                st.write(
                    f"**Kategorie:** "
                    f"{fundstueck.get('kategorie', '-')}"
                )

                st.write(
                    f"**Farbe:** "
                    f"{fundstueck.get('farbe', '-')}"
                )

                st.write(
                    f"**Fundort:** "
                    f"{fundstueck.get('fundort', '-')}"
                )

                st.write(
                    f"**Funddatum:** "
                    f"{fundstueck.get('datum', '-')}"
                )

                beschreibung = fundstueck.get(
                    "beschreibung",
                    ""
                )

                if (
                    pd.notna(beschreibung)
                    and str(beschreibung).strip()
                ):
                    st.write(
                        f"**Beschreibung:** "
                        f"{beschreibung}"
                    )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                if st.button(
                    "✓ Gegenstand abgeholt",
                    key=f"erledigt_{fundstueck['id']}"
                ):

                    df_alle = lade_fundstuecke()

                    df_alle.loc[
                        df_alle["id"].astype(str)
                        == str(fundstueck["id"]),
                        "erledigt"
                    ] = True

                    speichere_fundstuecke(df_alle)

                    st.success(
                        "Das Fundstück wurde als abgeholt markiert."
                    )

                    st.rerun()
```

# Fundstück eingeben

elif st.session_state.seite == "eingeben":

```
if st.button("← Zurück"):
    st.session_state.seite = "start"
    st.rerun()

st.title("➕ Fundstück eintragen")

st.write(
    "Du hast etwas gefunden? "
    "Trage es hier ein, damit andere danach suchen können."
)

bild = st.file_uploader(
    "📷 Foto des Fundstücks",
    type=["jpg", "jpeg", "png"]
)

if bild is not None:

    image = Image.open(bild)

    st.image(
        image,
        caption="Hochgeladenes Foto",
        use_container_width=True
    )

    if st.button("🤖 Bild mit KI erkennen"):

        with st.spinner(
            "Die KI schaut sich das Bild an ..."
        ):

            try:
                kategorie, sicherheit = erkenne_kategorie(
                    image
                )

                st.session_state.ki_kategorie = kategorie

                st.success(
                    f"Die KI schlägt **{kategorie}** vor."
                )

                st.write(
                    f"Übereinstimmung: "
                    f"**{sicherheit:.1f} %**"
                )

            except Exception as fehler:

                st.error(
                    "Die Bilderkennung konnte nicht "
                    "ausgeführt werden."
                )

                st.write(
                    f"Technischer Fehler: {fehler}"
                )

st.divider()

gegenstand = st.text_input(
    "Gegenstand *",
    placeholder="z. B. schwarzer Rucksack"
)

vorgeschlagene_kategorie = (
    st.session_state.ki_kategorie
)

if vorgeschlagene_kategorie not in KATEGORIEN:
    vorgeschlagene_kategorie = "Sonstiges"

kategorie = st.selectbox(
    "Kategorie *",
    KATEGORIEN,
    index=KATEGORIEN.index(
        vorgeschlagene_kategorie
    )
)

farbe = st.text_input(
    "Farbe",
    placeholder="z. B. schwarz"
)

fundort = st.text_input(
    "Fundort *",
    placeholder="z. B. Sporthalle"
)

datum = st.date_input(
    "Funddatum",
    value=date.today()
)

beschreibung = st.text_area(
    "Beschreibung",
    placeholder=(
        "Weitere Informationen zum Fundstück."
    )
)

if st.button(
    "💾 Fundstück speichern",
    type="primary"
):

    if not gegenstand.strip():
        st.error(
            "Bitte gib einen Gegenstand ein."
        )

    elif not fundort.strip():
        st.error(
            "Bitte gib den Fundort ein."
        )

    else:

        df = lade_fundstuecke()

        fundstueck_id = str(
            uuid.uuid4()
        )

        bild_pfad = ""

        if bild is not None:

            dateiname = (
                fundstueck_id
                + Path(bild.name).suffix.lower()
            )

            bild_pfad = str(
                UPLOAD_FOLDER / dateiname
            )

            with open(
                bild_pfad,
                "wb"
            ) as datei:
                datei.write(
                    bild.getbuffer()
                )

        neuer_eintrag = pd.DataFrame(
            [
                {
                    "id": fundstueck_id,
                    "gegenstand": gegenstand.strip(),
                    "kategorie": kategorie,
                    "farbe": farbe.strip(),
                    "fundort": fundort.strip(),
                    "datum": str(datum),
                    "beschreibung": beschreibung.strip(),
                    "bild": bild_pfad,
                    "erledigt": False
                }
            ]
        )

        df = pd.concat(
            [df, neuer_eintrag],
            ignore_index=True
        )

        speichere_fundstuecke(df)

        st.session_state.ki_kategorie = "Sonstiges"

        st.success(
            "✅ Das Fundstück wurde gespeichert!"
        )

        st.info(
            "Andere können es jetzt über die Suche finden."
        )

        if st.button("🔎 Zur Suche"):
            st.session_state.seite = "suchen"
            st.rerun()
```
