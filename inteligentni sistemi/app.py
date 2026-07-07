import os, re, json
import torch
import torch.nn.functional as F
import joblib
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq

st.set_page_config(page_title="FakeAd", page_icon="🚩", layout="centered")

# ---------- Učitavanje modela (kešira se, učita se samo jednom) ----------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("fakead_model")
    bert = AutoModelForSequenceClassification.from_pretrained("fakead_model").eval()
    baseline = joblib.load("baseline.joblib")
    return tokenizer, bert, baseline

tokenizer, bert, baseline = load_models()

def clean_html(x):
    x = re.sub(r"<[^>]+>", " ", str(x))
    return re.sub(r"\s+", " ", x).strip()

def bert_predict(text):
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = bert(**inputs).logits
    return F.softmax(logits, dim=1)[0, 1].item()

def baseline_predict(text):
    X = baseline["vectorizer"].transform([text])
    return baseline["clf"].predict_proba(X)[0, 1]

def llm_explain(text, prob_fake, api_key):
    client = Groq(api_key=api_key)
    system = ("Ti si analitičar za detekciju prevara u oglasima za posao. "
              "Identifikuješ konkretne 'crvene signale' (nerealna zarada, hitnost, "
              "traženje uplate ili ličnih podataka, šturi opis, sumnjiv kontakt). "
              "Odgovaraš ISKLJUČIVO na srpskom i ISKLJUČIVO validnim JSON objektom.")
    user = f"""Model je dao verovatnoću prevare {prob_fake:.0%}.

Oglas:
\"\"\"{text[:3000]}\"\"\"

Vrati JSON:
{{"procena": "...", "crveni_signali": ["..."], "objasnjenje": "..."}}"""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ---------- UI ----------
st.title("🚩 FakeAd")
st.caption("AI sistem za detekciju lažnih oglasa za posao")

with st.sidebar:
    st.header("Podešavanja")
    api_key = st.text_input("Groq API ključ", type="password",
                            help="Za objašnjenje (LLM sloj). Nabavi na console.groq.com/keys")
    use_llm = st.checkbox("Uključi LLM objašnjenje", value=True)

text = st.text_area("Nalepi tekst oglasa:", height=220,
                    placeholder="Unesi opis posla, uslove, benefite...")

if st.button("Analiziraj", type="primary"):
    if not text.strip():
        st.warning("Unesi tekst oglasa.")
        st.stop()

    clean = clean_html(text)
    p_bert = bert_predict(clean)
    p_base = baseline_predict(clean)

    col1, col2 = st.columns(2)
    col1.metric("DistilBERT", f"{p_bert:.0%}", help="Verovatnoća da je lažno")
    col2.metric("Baseline (TF-IDF)", f"{p_base:.0%}")

    if p_bert >= 0.5:
        st.error(f"⚠️ Sumnjiv oglas — verovatnoća prevare {p_bert:.0%}")
    else:
        st.success(f"✅ Verovatno legitiman — verovatnoća prevare {p_bert:.0%}")

    if use_llm:
        if not api_key:
            st.info("Unesi Groq API ključ u sidebar-u za objašnjenje.")
        else:
            with st.spinner("LLM analizira..."):
                try:
                    r = llm_explain(clean, p_bert, api_key)
                    st.subheader("🔍 Analiza")
                    st.write(f"**Procena:** {r['procena']}")
                    st.write("**Crveni signali:**")
                    for s in r["crveni_signali"]:
                        st.write(f"- 🚩 {s}")
                    st.write(f"**Objašnjenje:** {r['objasnjenje']}")
                except Exception as e:
                    st.error(f"LLM greška: {e}")