import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="centered")

col1, col2 = st.columns([0.1,1])
with col1:
    print('\n')
    print('\n')
    st.image('Bupa Logo.png', width = 90)

with col2:
    st.title('KurumMapping')

st.write('\n')
st.write('\n')

st.markdown("""
    <div style="font-size: 18px; font-weight: bold;">
        Maplemenin yapılabılmesi için yüklenen dosya aşağıdaki gibi olmalıdır:
        <ul>
            <li>xls, xlsx veya csv dosyası.</li>
            <li>Sadece bir tane sütun (A1 den başlayarak).</li>
            <li>Bu sütun kurum isimlerini içermelidir.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.write('\n')

@st.cache_resource
def load_references():
    return joblib.load('reference_data.pkl')

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def matched_function(row):

    hc_group = hc_groups[row['kurum_adi']]
    return row['kurum_adi'], 100, hc_group

# def unmatched_function(row):  

#     input_embedding = model.encode(row["kurum_adi"], convert_to_tensor=True)
#     cosine_scores = util.cos_sim(input_embedding, reference_embeddings)

#     best_idx = cosine_scores.argmax().item()
#     best_score = cosine_scores[0, best_idx].item() * 100  
#     best_match = reference_list[best_idx]
#     hc_group = hc_groups.get(best_match, None)

#     return best_match, best_score, hc_group

def unmatched_function(row):  

    if "YEDITEPE" in row["kurum_adi"]:
        filtered_references = [ref for ref in reference_list if "YEDITEPE" in ref]
        filtered_embeddings = model.encode(filtered_references, convert_to_tensor=True)
    elif "ACIBADEM" in row["kurum_adi"]:
        filtered_references = [ref for ref in reference_list if "ACIBADEM" in ref]
        filtered_embeddings = model.encode(filtered_references, convert_to_tensor=True)
    # elif "ECZANESI" in row["kurum_adi"]:
    #     filtered_references = [ref for ref in reference_list if "ECZANESI" in ref]
    #     filtered_embeddings = model.encode(filtered_references, convert_to_tensor=True)
    else:
        filtered_references = reference_list
        filtered_embeddings = reference_embeddings

    input_embedding = model.encode(row["kurum_adi"], convert_to_tensor=True)

    cosine_scores = util.cos_sim(input_embedding, filtered_embeddings)

    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[0, best_idx].item() * 100  
    best_match = filtered_references[best_idx]

    hc_group = hc_groups.get(best_match, None)

    return best_match, best_score, hc_group


def create_input_df(original_input):

    df = (
        original_input
        .assign(
            kurum_adi=lambda x: x[original_input.columns[0]]
            .str.strip()
            .str.upper()
            .str.replace('(eski Adı Biruni Üniv.sağ.eğitimi Uygulama Ve Araş. Merk)','')
            .str.replace("İ", "I")
            .str.replace("Ö", "O")
            .str.replace("Ü", "U")
            .str.replace("Ç", "C")
            .str.replace("Ş", "S")
            .str.replace("?", "")
            .str.replace("!", "")
            .str.replace(" N.HAST", " FLORENCE NIGHTINGALE HASTANESI")
            .str.replace("HASTANESI", "HOSPITAL") 
            .str.replace("HASTANE", "HOSPITAL")
            .str.replace("LIV HASTANESI", "LIV HOSPITAL")
            .str.replace("(ACIBADEM POLIKLINIKLERI A.S.)", "") 
            .str.replace("HİZ.TİC.A.Ş", "") 
            .str.replace("HIZ.TIC", "") 
            .str.replace(" HIZ.", "")
            .str.replace(" TIC.", "")
            .str.replace(" AS.", "")
            .str.replace(" SAG ", " SAGLIK ")
            .str.replace(" SAG.", " SAGLIK ")
            .str.replace("ORTOPEDI", "ORT.")
            .str.replace("OZEL ","")
            .str.replace(" OZEL ","")
            #.str.replace(" ECZANE", " ECZANESI")

            #.str.replace(" HAST.", " HASTANESI")
            .str.replace(" ECZANE ", " ECZANESI ")
            #.str.replace(" HAST. ", " HASTANESI ")
            .str.replace("ECZANE ", "ECZANESI ")
            #.str.replace("HAST. ", "HASTANESI ")

            .str.replace(" HAST.", " HOSPITAL")
            .str.replace(" HAST. ", " HOSPITAL ")
            .str.replace("HAST. ", "HOSPITAL ")

        )
)
    return df

# BUNA BAK
#df = create_input_df(pd.read_excel('ornek_excel.xlsx'))

st.markdown("""
    <label style="font-size: 18px; font-weight: bold;">
        Kurum Adlarını Buraya Yükleyebilirsiniz:
    </label>
""", unsafe_allow_html=True)

input_file = st.file_uploader('', type=['xls', 'xlsx', 'csv'])

if input_file is not None:

    if input_file.name.endswith('.csv'):
        original_input = pd.read_csv(input_file)
    else:
        original_input = pd.read_excel(input_file)

    st.write('Yüklemiş Olduğunuz Dosya')
    st.write('\n')
    st.dataframe(original_input)

    with st.spinner("Mapleme Yapılıyor"):

        df = create_input_df(original_input)
        model = load_model()
        hc_groups, reference_embeddings = load_references()

        reference_list = list(hc_groups.keys())

        df[["HEALTHCENTERDESC", "similarity_score", "HC_GROUP"]] = df.apply(
                lambda row: pd.Series(
                    matched_function(row) if row["kurum_adi"] in reference_list else unmatched_function(row)
                ),
                axis=1
            )
        
    st.success("Mapleme Tamamlandı")
        
    final = (
            df
            #.drop(columns = 'kurum_adi')
            .rename(columns = {
                'HEALTHCENTERDESC':'MAPLENDIGI_KURUM_ADI',
                'HC_GROUP':'MAPLENDIGI_KURUM_TIPI',
                'similarity_score': 'YAKINLIK_SKORU'})
            .assign(MANUEL_KONTROL = lambda x: np.where(x['YAKINLIK_SKORU'] <80, 'EVET', 'HAYIR'))
            [[
                df.columns[0],
                'kurum_adi',
                'MAPLENDIGI_KURUM_ADI',
                'MAPLENDIGI_KURUM_TIPI',
                'YAKINLIK_SKORU',
                'MANUEL_KONTROL'
            ]]
                
    )

    st.dataframe(final)