import streamlit as st
from PIL import Image
from transformers import pipeline


st.set_page_config(
    page_title="Fundkiste",
    page_icon="🔎",
    layout="centered"
)


# Hugging-Face-KI nur einmal laden
@st.cache_resource
def lade_ki():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


# Startseite
st.title("Fundkiste")
st.write("KATHARINEUM ZU LÜBECK")
st.write("seit 1531")

st.divider()

st.header("🔎 Gefundenen Gegenstand erkennen")

st.write(
    "Lade ein Foto des gefundenen Gegenstands hoch. "
    "Die KI versucht anschließend, den Gegenstand zu erkennen."
)

bild = st.file_uploader(
    "Foto auswählen",
    type=["jpg", "jpeg", "png"]
)

if bild is not None:

    image = Image.open(bild)

    st.image(
        image,
        caption="Hochgeladenes Bild",
        use_container_width=True
    )

    if st.button("Gegenstand erkennen", type="primary"):

        with st.spinner("Die KI schaut sich das Bild an ..."):

            try:
                ki = lade_ki()

                kategorien = [
                    "clothing",
                    "keys",
                    "water bottle",
                    "bag",
                    "mobile phone",
                    "jewelry",
                    "school supplies",
                    "other object"
                ]

                ergebnis = ki(
                    image,
                    candidate_labels=kategorien
                )

                bestes_ergebnis = ergebnis[0]

                uebersetzung = {
                    "clothing": "Kleidung",
                    "keys": "Schlüssel",
                    "water bottle": "Trinkflasche",
                    "bag": "Tasche",
                    "mobile phone": "Handy",
                    "jewelry": "Schmuck",
                    "school supplies": "Schulmaterial",
                    "other object": "Sonstiger Gegenstand"
                }

                kategorie = uebersetzung.get(
                    bestes_ergebnis["label"],
                    "Sonstiger Gegenstand"
                )

                sicherheit = bestes_ergebnis["score"] * 100

                st.success(
                    f"Erkannte Kategorie: **{kategorie}**"
                )

                st.write(
                    f"Übereinstimmung: **{sicherheit:.1f} %**"
                )

                st.info(
                    "Die Erkennung ist nur eine Einschätzung. "
                    "Bitte kontrolliere die Kategorie vor dem Speichern."
                )

            except Exception as fehler:

                st.error(
                    "Die Bilderkennung konnte gerade nicht ausgeführt werden."
                )

                st.write(f"Fehler: {fehler}")
