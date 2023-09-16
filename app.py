from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import joblib as jl
from joblib import  load

icon = Image.open('fav.png')
st.set_page_config(page_title='DeepAce', page_icon = icon)

import zipfile
with zipfile.ZipFile('./models/rf1.zip', 'r') as zip_ref:
    zip_ref.extractall('./models/')

def encodeSeq(seq):
    encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    encSeq = [0 for x in range(41)]
    i = 0
    ra = len(seq)
    for i in range(ra):
        value = encoder.index(seq[i])
        encSeq[i] = value
    seqArray=np.asarray(encSeq)
    return seqArray.reshape(1,41)


def modelLoader():
    myLSTM = load_model("./models/lstm_model_dropout.h5")
    myRF = jl.load("./models/random_forest.joblib")
    return myLSTM, myRF


def seqValidator(seq):
    checkSet = {'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    if set(seq).issubset(checkSet):
            return True
    return False


final_df = pd.DataFrame(columns=['Specie Name', 'Sub Sequence','Label'])
seq = ""
len_seq = 0
image = Image.open('c.jpg')
st.subheader("""DeepAce""")
st.image(image)
st.sidebar.subheader(("Input Sequence(s) (FASTA FORMAT ONLY)"))
fasta_string  = st.sidebar.text_area("Sequence Input", height=200)       
st.subheader("Click the Example Button for Sample Data")


if st.button('Example'):
    Ecoli = "SAEGFNFIGTGVSGGEEGALKGPSIMPGGQKEAYELVAPILSLLTWGDHELEAAGRLTQPLKYDAVSDCYKPLSWQQAFDEI" #10
    Bvelezensis = "YPDTTSLIPQESKTEIVVNTKEFLQAIDRASLLAREGRNNVELPYQKPVLISGTDGVGTKLKLAFSMDKHDTIGVDAVAMCV" #10
    Bsubtilis = "DLSVVIRGNSMSDVARFVSDKLSTLDSVVSTTTHFILKKYKLLLLMTIAALFAAAFAKKRAKMAVYGIIILSSLFVANSYQK" #10
    Cglutamicum = "DATLNEHNRPEGSVRLLPVTKFHPVEDIKILQEFGVTAVGEFVEIKPFHRVLHFIRVSDKDKVQGILAQAAHVDSSGLKVTN" #10
    Gkaustophilus = "GLVLFTYLHLAADPELTRVLKESGVIAIAYETVQVGRTLPLDVPDVIEIGFEQGVPVTLNGKAYPLAQLILELNALAGKHGV" #10
    Mtuber = "FGVPTVWSRVAADQAAAGALKPARLLVSGSAALPVPVFDKLVYSQRNRSACVRIPITGSNPKAKRLEFRSPDSSGNPYLAFS" #10
    Styphimurium= "ELRHEVTPKNILMIGPTGVGKTEIARRLAKLANAPFIKVEALRKATLDTLALYLACGIDPEKSTIFVQSHVPEHAQLGWALN" #10
    Seriocheiris= "LDIIDFERGTKLSGSRFIIYKGLGARLERAIINLMLDEHLKPDLDIEEDSTAFLSYQGPNQKLADLERFKTYLASDYLSLNT" #10
    st.code(">Escherichia_coli (strain K12) \n"+Ecoli, language="markdown")
    st.code(">Bacillus_velezensis (UCMB5033)	\n"+Bvelezensis, language="markdown")
    st.code(">Bacilus_subtilis (strain 168)	\n"+Bsubtilis, language="markdown")
    st.code(">Corynebacterium_glutamicum (strain ATCC 13032) \n"+Cglutamicum, language="markdown")
    st.code(">Geobacillus_kaustophilus (strain HTA426) \n"+Gkaustophilus, language="markdown")
    st.code(">Mycobac_tuber (strain ATCC 25618 / H37Rv) \n"+Mtuber, language="markdown")
    st.code(">Spiroplasma_typhimurium (strain LT2) \n"+Styphimurium, language="markdown")
    st.code(">Spiroplasma_eriocheiris (CCTCC M 207170) \n"+Seriocheiris, language="markdown")

if st.sidebar.button("SUBMIT"):
    if(fasta_string==""):
        st.info("Please input the sequence first.")
    fasta_io = StringIO(fasta_string) 
    records = SeqIO.parse(fasta_io, "fasta") 
    for rec in records:
        seq_id = str(rec.id)
        seq=str(rec.seq).upper()
        if(seqValidator(seq)):
            # seq = "XXXXXXXXXXXXXXXXXXXX"+seq+"XXXXXXXXXXXXXXXXXXXX"
            seqLen = len(seq)
            print(seqLen)
            for i in range(0,seqLen+1):
                if i==0:
                   continue
                if i % 41 == 0 :
                    sub_seq = seq[i-41:i]
                    print(sub_seq)
                    df_temp = pd.DataFrame([[seq_id, sub_seq,str(i+1-20),'None']], columns=['Specie Name', 'Sub Sequence','Label'] )
                    final_df = pd.concat([final_df,df_temp], ignore_index=True)
        else:
            st.info("Sequence with Specie Name: " + str(seq_id) + " is invalid, containing letters other than standard amino acids")
    fasta_io.close()
    if(final_df.shape[0]!=0):
        myLSTM, myRF = modelLoader()
        sc=load('./models/std_scaler.bin')
        for iter in range(final_df.shape[0]):
            tempSeq =  final_df.iloc[iter, 1]
            seqArray = encodeSeq(tempSeq)
            fvArray = myLSTM.predict(seqArray)
            icu =sc.transform(fvArray)
            score = myRF.predict(icu)
            pred_label = np.round_(score, decimals=0, out=None)
            print(pred_label)
            if(pred_label==1):
                pred_label=" K-ACE site"
            else:
                pred_label=" Non K-ACE Site"
            final_df.iloc[iter, 3] = str(pred_label)
    st.dataframe(final_df)

