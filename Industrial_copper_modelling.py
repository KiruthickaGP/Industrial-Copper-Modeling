
#Part 1

import numpy as np
import streamlit as st
import re
from PIL import Image
from streamlit_option_menu import option_menu
import warnings
import pickle
warnings.filterwarnings("ignore")

#Part 2

# Open the image file for the page icon
icon = Image.open("E:\Guvidatascience\Projects\Industrial_Copper_Modeling\Industrial_copper_modelling.png")
# Set page configurations with background color
st.set_page_config(
    page_title="Industrial Copper Modeling  | By Kiruthicka",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': """# This app is created by *Kiruthicka!*"""})



# Add background color using CSS
background_color = """
<style>
    body {
        background-color: #F7EBED;  /* Set background color to #F7EBED*/            #AntiqueWhite color
    }
    .stApp {
        background-color: #F7EBED; /* Set background-color for the entire app */
    }
</style>
"""
#AntiqueWhite color #F7EBED
st.markdown(background_color, unsafe_allow_html=True)




# CREATING OPTION MENU
with st.sidebar:
    selected = option_menu(None,["Home","Predictive analysis"],
        icons=["house-fill","tools"],
        default_index=0,
        orientation="Vertical",
        styles={
            "nav-link": {
                "font-size": "30px",
                "font-family": "Fira Sans",
                "font-weight": "Bold",
                "text-align": "left",
                "margin": "10px",
                "--hover-color": "#964B00"#Brown
            },
            "icon": {"font-size": "30px"},
            "container": {"max-width": "6000px"},
            "nav-link-selected": {
                "background-color": "#CD7F32", #Bronze
                "color": "Bronze",
            }
        }
    )

#Part3
# HOME PAGE
# Open the image file for the YouTube logo
logo = Image.open("E:\\Guvidatascience\\Projects\\Final_project\\Finalproject.png")

# Define a custom CSS style to change text color
custom_style = """
<style>
    .black-text {
        color: black; /* Change text color to black */
    }
</style>
"""

   
# Apply the custom style
st.markdown(custom_style, unsafe_allow_html=True)

if selected == "Home":
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open("E:\Guvidatascience\Projects\Industrial_Copper_Modeling\Industrial_copper_modelling.png")
        st.image(image, width=700,  output_format='PNG', use_column_width=False)
        

    with col2:
        st.markdown("## :green[**Technologies Used :**]")
        st.markdown("### Python: The core programming language for data analysis, machine learning, and web application development.")
        st.markdown("### Pandas, NumPy, Matplotlib, Seaborn: Libraries for data manipulation, numerical operations, and data visualization.")
        st.markdown("### Scikit-learn: A machine learning library for building and evaluating regression and classification models.")
        st.markdown("### Streamlit: A Python library for creating interactive web applications with minimal code.")


        st.markdown("## :green[**Overview :**]")
        st.markdown("### Industrial Copper Modeling is a comprehensive project focusing on data analysis, machine learning, and web development. The project involves Python scripting for data preprocessing, exploratory data analysis (EDA), and building machine learning models for regression and classification. The Streamlit framework is used to create an interactive web page allowing users to input data and obtain predictions for selling price or lead status. ")

        
#Part 5

if selected == "Predictive analysis":
    

    st.write("""
    <div style='text-align:center'>
        <h1 style='color:#CD7F32;'>Industrial Copper Modeling Application</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])
    with tab1:
        # Define the possible values for the dropdown menus
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                        'Offerable']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                            79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
                '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
                '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
                '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
                '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        # Define the widgets for user input
        with st.form("my_form"):
            col1, col2, col3 = st.columns([5, 2, 5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options, key=1)
                item_type = st.selectbox("Item Type", item_type_options, key=2)
                country = st.selectbox("Country", sorted(country_options), key=3)
                application = st.selectbox("Application", sorted(application_options), key=4)
                product_ref = st.selectbox("Product Reference", product, key=5)
            with col3:
                st.write(
                    f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>',
                    unsafe_allow_html=True)
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: #CD7F32;
                            color: white;
                            width: 100%;
                        }
                        </style>
                    """, unsafe_allow_html=True)

            flag = 0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons, thickness, width, customer]:
                if re.match(pattern, i):
                    pass
                else:
                    flag = 1
                    break

        if submit_button and flag == 1:
            if len(i) == 0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ", i)

        if submit_button and flag == 0:
            import pickle

            with open(r"E:\Guvidatascience\Projects\Industrial_Copper_Modeling\model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r'E:\Guvidatascience\Projects\Industrial_Copper_Modeling\scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)
            with open(r"E:\Guvidatascience\Projects\Industrial_Copper_Modeling\t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)
            with open(r"E:\Guvidatascience\Projects\Industrial_Copper_Modeling\s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                    country, float(customer), int(product_ref), item_type, status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, ]], new_sample_ohe, new_sample_be), axis=1)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

with tab2:
    with st.form("my_form1"):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

        cflag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
            if re.match(pattern, k):
                pass
            else:
                cflag = 1
                break

    if csubmit_button and cflag == 1:
        if len(k) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", k)

    if csubmit_button and cflag == 0:
        

        with open(r"E:\Guvidatascience\Projects\Industrial_Copper_Modeling\clsmodel.pkl", 'rb') as file:
            cloaded_model = pickle.load(file)

        with open(r"E:\Guvidatascience\Projects\Industrial_Copper_Modeling\cscaler.pkl", 'rb') as f:
            cscaler_loaded = pickle.load(f)

        with open(r"E:\Guvidatascience\Projects\Industrial_Copper_Modeling\ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)

        # Predict the status for a new sample
        # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
        new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(product_ref),
                                citem_type]])
        new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
        new_sample = cscaler_loaded.transform(new_sample)
        new_pred = cloaded_model.predict(new_sample)
        if new_pred == 1:
            st.write('## :green[The Status is Won] ')
        else:
            st.write('## :red[The status is Lost] ')
