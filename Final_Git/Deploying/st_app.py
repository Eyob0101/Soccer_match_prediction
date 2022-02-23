from unittest import result
import streamlit as st
import pickle
import pandas as pd
from PIL import Image


RF_model = pickle.load(open('pipeline_RF.pickle', 'rb'))

img = Image.open('EPL1.png')
image = Image.open('EPL2.png')
 
def Predict(HTP,ATP,HM1,HM2,HM3,AM1,AM2,AM3,HTGD,ATGD,LP,DiffFormPts,B365H,B365D,B365A,NGR_HT,GSR_HT,NGR_AT,GSR_AT):
    X = pd. DataFrame([[HTP,ATP,HM1,HM2,HM3,AM1,AM2,AM3,HTGD,ATGD,LP,DiffFormPts,B365H,B365D,B365A,NGR_HT,GSR_HT,NGR_AT,GSR_AT]],
     columns = ['HTP','ATP','HM1','HM2','HM3','AM1','AM2','AM3','HTGD','ATGD','DiffLP','DiffFormPts','B365H','B365D','B365A','NGR(HT)','GSR(HT)','NGR(AT)','GSR(AT)'])
    prediction = RF_model.predict_proba(X)
    return prediction




def main():
    st.set_page_config(page_icon=img)
    st.image(image, caption='English Premier League Players')

    st.title("EPL Soccer Match Predicter")

    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color: white;text-align:center;"> See If Your Favourite EPL Team Can win The next Match </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Menu = ['Home', 'About']
    choice = st.sidebar.selectbox("Menu", Menu)
    if choice == 'Home':
        st.subheader("Lets fill in the informations")

    
        with st.form(key='Soccerform'):
            col1, col2, col3 = st.columns([3,3,3])
            with col1:
                home_team = st.text_input("Home Team")
                HTP = st.number_input("Home Team Current Point")
                HM1 = st.selectbox("Home Team Last Match Result",['W','D','L'])
                HM2 = st.selectbox("Home Team Last Second Match Result",['W','D','L'])
                HM3 = st.selectbox("Home Team Last Third Match Result",['W','D','L'])
                HTGD = st.number_input("Net Goal Of Home Team(Goal Scored - Goal Conceded)")
                B365H = st.number_input("Bet365 odd for the Home Team")
                NGR_HT = st.number_input("Net Goal Rank of Home Team")
                GSR_HT = st.number_input("Total Goal Scored Rank Of Home Team")
            with col2:
                Away_team = st.text_input("Away Team")
                ATP = st.number_input("Away Team Current Point")
                AM1 = st.selectbox("Away Team Last Match Result",["W", "D", "L"])
                AM2 = st.selectbox("Away Team Last Second Match Result",["W", "D", "L"])
                AM3 = st.selectbox("Away Team Last Third Match Result",["W", "D", "L"])
                ATGD = st.number_input("Net Goal Of Away Team(Goal Scored - Goal Conceded)")
                B365A = st.number_input("BET365 odd for the Away Team")
                NGR_AT = st.number_input("Net Goal Rank of Away Team")
                GSR_AT = st.number_input("Total Goal Scored Rank Of Away Team")
            with col3:
                LP = st.number_input("Diffrence of Home team and Away Team Last Season Positions(Home Team LP - Away Team LP)")
                DiffFormPts = st.number_input("Difrence of the teams last five matchs points")
                B365D = st.number_input("BET365 odd for Draw ")


            Result=""

            if st.form_submit_button(label="Predict"):
                Result = Predict(float(HTP), float(ATP),HM1, HM2, HM3, AM1, AM2, AM3, float(HTGD), float(ATGD), float(LP), float(DiffFormPts), float(B365H),float(B365D), float(B365A), float(NGR_HT), float(GSR_HT), float(NGR_AT), float(GSR_AT))
                st.success(f"The probability of {home_team} winning the Match is {round((Result.flat[0]) * 100)}%")
    else:
        st.subheader("About")
        st.write("""This app is a soccer match predicter which takes a row information in and
        spits out te probability that the home team will win the game. the app uses a 
        trained random forest machine learning model which shows 72 persent accuracy of  on 
        a test
        
        Developer :- Eyob Bekele
        Email :- Eyobahede@gmail.com""")

if __name__=='__main__':
    main()

