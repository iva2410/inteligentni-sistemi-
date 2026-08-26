# FakeAd — detekcija lažnih oglasa za posao

Sistem koji na osnovu teksta oglasa za posao procenjuje verovatnoću da je oglas prevara i objašnjava
korisniku **zbog čega** je označen kao sumnjiv.

Projekat iz predmeta *Inteligentni sistemi*.

---

## Problem

Oglasnici za posao su čest kanal za prevare — lažni oglasi služe za krađu ličnih podataka, naplatu
„kotizacije” za obuku ili regrutovanje za pranje novca. Prijave se obično obrađuju ručno, sa zakašnjenjem.
Cilj rada je automatska detekcija takvih oglasa iz samog teksta.

## Podaci

**EMSCAD** (*Employment Scam Aegean Dataset*) — 17.880 realnih oglasa za posao, od kojih je 866 (**4,84%**)
označeno kao lažno.

Skup je izrazito nebalansiran, što je uticalo na sve ključne odluke u radu: stratifikovana podela podataka,
težine klasa u funkciji gubitka i izbor PR-AUC umesto tačnosti kao glavne metrike.

> `DataSet.csv` **nije** u repozitorijumu (veličina + uslovi korišćenja). Preuzima se sa zvaničnog izvora
> EMSCAD dataset-a i ručno učitava u prvoj ćeliji notebook-a.

## Arhitektura

Tri sloja, sa jasno podeljenim odgovornostima:

| Sloj | Model | Uloga |
|---|---|---|
| 1. Baseline | TF-IDF + logistička regresija | referentna tačka, potpuno interpretabilan (koeficijenti se čitaju direktno) |
| 2. Klasifikator | DistilBERT (fine-tuning) | razumevanje konteksta, kvantitativna ocena rizika |
| 3. Objašnjenje | LLM (Groq, `gpt-oss-20b`) | prevodi predikciju u konkretne „crvene signale” na srpskom |

Podela između sloja 2 i 3 je namerna: **odluku donosi klasifikator** treniran na 14.304 realna primera, a LLM
samo obrazlaže — jer je sklon halucinacijama i ne sme mu se poveriti konačna procena.

## Rezultati

Test skup: 3.576 oglasa (20%), identičan za oba modela (isti `random_state`, ista stratifikacija).
Metrike se odnose na klasu **„lažno”**:

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| TF-IDF + logistička regresija | 0,766 | 0,908 | 0,831 | 0,921 |
| **DistilBERT** | **0,924** | 0,844 | **0,882** | **0,934** |

DistilBERT smanjuje broj lažnih uzbuna sa 48 na 12, uz nešto niži odziv (157 → 146 otkrivenih lažnih
oglasa). PR-AUC je blago u njegovu korist (0,921 → 0,934) — **složeniji model nije apsolutno bolji, već
pomera kompromis** između preciznosti i odziva.

PR krive, matrice konfuzije i raspodela dužine teksta po klasi nalaze se u notebook-u (sekcije 2.1 i 4.5).

## Struktura repozitorijuma

```
fakeAD/
├── mvp.ipynb                     # ceo tok rada: EDA → baseline → DistilBERT → LLM → izvoz modela
└── inteligentni sistemi/
    ├── app.py                    # Streamlit aplikacija
    ├── requirements.txt
    ├── fakead_model/             # (nije u git-u) izvezen DistilBERT
    └── baseline.joblib           # (nije u git-u) izvezen TF-IDF + logistička regresija
```

## Pokretanje

### 1. Treniranje (Google Colab)

Notebook je pisan za Colab sa **GPU runtime-om** (`Runtime → Change runtime type → T4 GPU`);
fine-tuning na CPU-u je nepraktično spor.

1. Otvori `mvp.ipynb` u Colab-u.
2. Pokreni ćelije redom — prva traži da učitaš `DataSet.csv`.
3. Za LLM sloj (sekcija 5) treba besplatan API ključ sa [console.groq.com/keys](https://console.groq.com/keys).
4. Sekcije 6.1–6.3 izvoze modele: preuzmi `fakead_model.zip` i `baseline.joblib`.

### 2. Aplikacija (lokalno)

Raspakuj preuzete modele u folder `inteligentni sistemi/` tako da nastane struktura prikazana iznad
(`fakead_model/` kao folder, `baseline.joblib` kao fajl pored `app.py`), pa:

```bash
cd "inteligentni sistemi"
pip install -r requirements.txt
streamlit run app.py
```

Aplikacija se pokreće na `http://localhost:8501`. Nalepi tekst oglasa i klikni **Analiziraj** — dobijaš
verovatnoću prevare oba modela i, ako uneseš Groq ključ u bočnoj traci, listu crvenih signala sa
objašnjenjem.

> Aplikacija se mora pokrenuti **iz** foldera `inteligentni sistemi`, jer `app.py` učitava modele preko
> relativnih putanja.

## Ograničenja

- Dataset je na **engleskom** jeziku i star nekoliko godina, dok aplikacija odgovara na srpskom — model nije
  validiran na oglasima sa domaćeg tržišta.
- Evaluacija je rađena na jednom fiksnom test skupu; unakrsna validacija bi dala pouzdaniju procenu varijanse.
- Treniranje DistilBERT-a nema fiksiran `seed`, pa se rezultati između pokretanja razlikuju za ~0,01 po metrici.
- Prag odlučivanja je ostavljen na podrazumevanih 0,5 iako se preko PR krive može bolje kalibrisati.
- LLM sloj nije kvantitativno evaluiran — ocenjen je samo kvalitativno, na pojedinačnim primerima.
- Model uči i neke artefakte dataset-a (npr. imena konkretnih kompanija među najuticajnijim tokenima).

## Mogući nastavak rada

Kalibracija praga preko PR krive, dodavanje strukturiranih atributa (`has_company_logo`, `telecommuting`,
`salary_range`) uz tekstualne, poređenje sa modelom većeg kapaciteta (RoBERTa) i objašnjivost putem
SHAP/LIME umesto oslanjanja isključivo na LLM.
