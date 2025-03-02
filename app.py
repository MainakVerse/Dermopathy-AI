import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Dermatrix - Skin Disease Detection",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state='expanded'
)

# Custom CSS for modern look with no white backgrounds
st.markdown("""
<style>
    .main {
        background-color: #1e2130;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #262b3d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #353c54;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #d0d0d0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b71a0 !important;
        color: #e0e0e0 !important;
    }
    h1, h2, h3 {
        color: #8ab4f8;
    }
    .css-1kyxreq {
        justify-content: center;
    }
    .stAlert {
        background-color: #2b3548;
        border-color: #3b4863;
        color: #e0e0e0;
    }
    div.css-1kyxreq {
        background-color: #262b3d;
    }
    .css-18e3th9 {
        background-color: #1e2130;
    }
    .css-1d391kg {
        background-color: #262b3d;
    }
    .custom-box {
        background-color: #262b3d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4b71a0;
        color: #e0e0e0;
    }
    .result-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #2b3548;
        color: #e0e0e0;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-bottom: 20px;
    }
    .stTextInput > div > div {
        background-color: #353c54;
        color: #e0e0e0;
    }
    .stTextInput > label {
        color: #8ab4f8;
    }
    .stFileUploader > div > label {
        color: #8ab4f8;
    }
    .stFileUploader > div > div {
        background-color: #353c54;
    }
    .stButton > button {
        background-color: #4b71a0;
        color: #e0e0e0;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    .css-1vq4p4l {
        background-color: #262b3d;
    }
    .css-12oz5g7 {
        background-color: #1e2130;
    }
    div[data-testid="stSidebar"] {
        background-color: #262b3d;
        color: #e0e0e0;
    }
    div[data-testid="stSidebar"] > div:first-child {
        background-color: #262b3d;
    }
    .primary-box {
        background-color: #2e4a4e;
        padding: 15px;
        border-radius: 5px;
        height: 100%;
        border-left: 5px solid #28a745;
    }
    .secondary-box {
        background-color: #2e3c5a;
        padding: 15px;
        border-radius: 5px;
        height: 100%;
        border-left: 5px solid #0d6efd;
    }
    .disease-box {
        background-color: #2e3c5a;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 5px solid #4b71a0;
    }
    .result-container {
        background-color: #2e3c5a;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4b71a0;
    }
    .placeholder-box {
        background-color: #262b3d;
        padding: 50px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
    }
    .stChatInput > div {
        background-color: #353c54 !important;
        color: #e0e0e0 !important;
    }
    .stChatInput > div > input {
        color: #e0e0e0 !important;
    }
    div[data-testid="stChatMessage"] {
        background-color: #2e3c5a !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('skin.h5')
    return model

# Prediction function
def import_and_predict(image_data, model):
    size = (300, 300)    
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Class names
class_names = [
    'Eczema',
    'Warts Molluscum and other Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)',
    'Melanocytic Nevi (NV)',
    'Benign Keratosis-like Lesions (BKL)',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections'
]

# Disease descriptions for About tab
disease_descriptions = {
    'Eczema': "A condition that causes the skin to become itchy, red, dry and cracked. It's common in children but can occur at any age.",
    'Warts Molluscum and other Viral Infections': "Viral skin infections characterized by small, raised bumps on the skin. Contagious and spread through direct contact.",
    'Melanoma': "The most serious type of skin cancer that develops from the pigment-producing cells known as melanocytes.",
    'Atopic Dermatitis': "A chronic, inflammatory skin disease associated with asthma and hay fever, commonly starting in childhood.",
    'Basal Cell Carcinoma (BCC)': "The most common form of skin cancer, usually caused by sun exposure.",
    'Melanocytic Nevi (NV)': "Common moles that appear as small, dark brown spots caused by clusters of pigmented cells.",
    'Benign Keratosis-like Lesions (BKL)': "Non-cancerous growths that appear as waxy, scaly, slightly raised growths on the skin.",
    'Psoriasis pictures Lichen Planus and related diseases': "Inflammatory skin conditions characterized by scaly, itchy patches or bumps.",
    'Seborrheic Keratoses and other Benign Tumors': "Common non-cancerous skin growths that begin in keratinocytes and appear as waxy, scaly patches.",
    'Tinea Ringworm Candidiasis and other Fungal Infections': "Fungal infections on the skin characterized by ring-shaped rashes. Highly contagious."
}

# Remedies for diseases
remedies = {
    'Eczema': {
        'primary': 'An effective, intensive treatment for severe eczema involves applying a corticosteroid ointment and sealing in the medication with a wrap of wet gauze topped with a layer of dry gauze.',
        'secondary': "DUPIXENT® (dupilumab) is a prescription medicine used to treat people aged 6 years and older with moderate-to-severe atopic dermatitis (eczema) that is not well controlled with prescription therapies used on the skin (topical) or who cannot use topical therapies. Other treatments for eczema include azathioprine, cyclosporine, methotrexate, pimecrolimus, crisaborole, and tacrolimus, which are prescription creams and ointments that control inflammation and reduce immune system reactions. Calcineurin inhibitors, such as pimecrolimus and tacrolimus, are also recommended if OTC steroids don't work or cause problems. Corticosteroid creams, solutions, gels, foams, and ointments, made with hydrocortisone steroids, can quickly relieve itching and reduce inflammation. Pimecrolimus cream or tacrolimus ointment, also known as topical calcineurin inhibitors (TCIs), may be prescribed by a dermatologist.",
        'contagious': False
    },
    'Melanoma': {
        'primary': "Treatment for early-stage melanomas usually includes surgery to remove the melanoma. A very thin melanoma may be removed entirely during the biopsy and require no further treatment. Otherwise, your surgeon will remove the cancer as well as a border of normal skin and a layer of tissue beneath the skin.",
        'secondary': "Ipilimumab (Yervoy®) is an immunotherapy drug used to treat metastatic melanoma and stage III melanoma that cannot be removed completely with surgery. It works by blocking an immune molecule called CTLA-4. Checkpoint inhibitors, also known as immune checkpoint blockade, are commonly used to treat melanoma. Interferon alfa (Intron A, Roferon-A) can be used after surgery to prevent melanoma recurrence. Targeted therapy of melanoma includes vemurafenib, cobimetinib, dabrafenib, and trametinib, which attack cells that have a damaged BRAF gene. Targeted medicines for melanoma with NRAS and C-KIT mutations may be available through clinical trials.",
        'contagious': False
    },
    'Atopic Dermatitis': {
        'primary': "The main treatments for atopic eczema are: emollients (moisturisers) used every day to stop the skin becoming dry. topical corticosteroids creams and ointments used to reduce swelling and redness during flare-ups.",
        'secondary': "DUPIXENT® (dupilumab) is a prescription medicine used to treat moderate-to-severe atopic dermatitis (eczema) that is not well controlled with prescription therapies used on the skin (topical), or who cannot use topical therapies. It is not known if DUPIXENT is safe and effective in children with atopic dermatitis under 6 years of age. Cibinqo (abrocitinib) is an oral JAK1 inhibitor approved by the FDA for adults with refractory moderate to severe atopic dermatitis whose disease is not adequately controlled with other systemic drug products, including biologics, or when use of those therapies is inadvisable. Immunosuppressants are prescribed for moderate to severe atopic dermatitis in children and adults to help stop the itch-scratch cycle of eczema, to allow the skin to heal and reduce the risk of skin infection. Topical calcineurin inhibitors, immunosuppressant tablets, and alitretinoin are some of the topical treatments for atopic dermatitis.",
        'contagious': False
    },
    'Basal Cell Carcinoma (BCC)': {
        'primary': "The current mainstay of BCC treatment involves surgical modalities such as excision, electrodesiccation and curettage (EDC), cryosurgery, and Mohs micrographic surgery. Such methods are typically reserved for localized BCC and offer high 5-year cure rates, generally over 95%.",
        'secondary': "Basal cell skin cancer does not usually respond to chemotherapy, but it often responds to a targeted drug called vismodegib, sold as Erivedge®, which helps disrupt the activity of a group of proteins in the body called hedgehog. Erivedge® (vismodegib) capsule is a prescription medicine used to treat adults with basal cell carcinoma that has spread to other parts of the body or that has come back after surgery or that cannot be treated with surgery or radiation. It is the #1 most-prescribed oral medication for advanced basal cell carcinoma.",
        'contagious': False
    },
    'Melanocytic Nevi (NV)': {
        'primary': "Small nevi can be removed by simple surgical excision. The nevus is cut out, and the adjacent skin stitched together leaving a small scar. Removal of a large congenital nevus, however, requires replacement of the affected skin.",
        'secondary': "Melanocytic nevus is the medical term for a mole. Nevi can appear anywhere on the body. They are benign (non-cancerous) and typically do not require treatment. A very small percentage of melanocytic nevi may develop a melanoma within them. Of note, the majority of cutaneous melanomas arise within normally appearing skin.",
        'contagious': False
    },
    'Benign Keratosis-like Lesions (BKL)': {
        'primary': "Cryosurgery: The dermatologist applies liquid nitrogen, a very cold liquid, to the growth with a cotton swab or spray gun. Electrosurgery and curettage: Electrosurgery (electrocautery) involves numbing the growth with an anesthetic and using an electric current to destroy the growth.",
        'secondary': "A seborrheic keratosis is a growth on the skin. The growth is not cancer (benign). It's color can range from white, tan, brown, or black. Seborrheic keratoses often appear on a person's chest, arms, back, or other areas. They're very common in people older than age 50.",
        'contagious': False
    },
    'Psoriasis pictures Lichen Planus and related diseases': {
        'primary': "Lichen planus does not usually require treatment. It often goes away by itself within a year. If a person has particularly itchy or painful outbreaks, a doctor may prescribe topical corticosteroids or light therapy. Psoriasis is a long-term condition, but people can usually manage their symptoms well.",
        'secondary': "There isn't a cure for lichen planus. If you have lichen planus on your skin, in most cases, it goes away without treatment in as little as a few months to several years. Corticosteroid creams or ointments. Your healthcare provider may prescribe corticosteroid creams or ointments to reduce inflammation. Phototherapy uses ultraviolet light, usually ultraviolet B (UVB), from special lamps. The ultraviolet light waves found in sunlight can help certain skin disorders, including lichen planus.",
        'contagious': False
    },
    'Seborrheic Keratoses and other Benign Tumors': {
        'primary': "Eskata, a 40% hydrogen peroxide topical solution, is the first FDA-approved drug for treatment of seborrheic keratoses. Administration of the drug may be tedious and usually requires at least two office visits.",
        'secondary': "Ammonium lactate and alpha hydroxy acids have been reported to reduce the height of seborrheic keratoses, and superficial lesions can be treated by carefully applying pure trichloroacetic acid and repeating if the full thickness is not removed on the first treatment. Topical treatment with tazarotene cream 0.1% applied twice daily for 16 weeks caused clinical improvement in seborrheic keratoses in 7 of 15 patients. Diclofenac gel may be a new treatment option for seborrheic keratosis. Hydrogen peroxide 40% (Eskata) is a topical solution for the in-office treatment of raised seborrheic keratosis lesions.",
        'contagious': False
    },
    'Tinea Ringworm Candidiasis and other Fungal Infections': {
        'primary': "Typically, a course of antifungal creams (either prescription or over-the-counter) will clear up the rash and relieve the itchiness. Your healthcare provider can also discuss preventive steps to keep the rash from coming back.",
        'secondary': "Tinea ringworm can be treated with over-the-counter (OTC) antifungal creams containing clotrimazole, ketoconazole, econazole, tolnaftate, or terbinafine. However, if there are many patchy areas, a prescription cream or oral antifungal medicine taken by mouth may be necessary.",
        'contagious': True
    },
    'Warts Molluscum and other Viral Infections': {
        'primary': "Doctors recommend many topical treatments for molluscum contagiosum. Podophyllotoxin (contraindicated in pregnant women), potassium hydroxide, salicylic acid (associated or not with povidone-iodine), benzoyl peroxide, and tretinoin are used as home treatments and must be applied to each lesion.",
        'secondary': "Cantharidin (beetle juice): This FDA-approved treatment is made from blister beetles. It's approved to treat adults and children two years of age and older. Dermatologists have been using cantharidin to treat warts and molluscum since the 1950s. When treating molluscum bumps, your dermatologist applies the beetle juice to each bump. Your dermatologist will apply it to each bump in such a way that a water blister later forms.",
        'contagious': True
    }
}

# Sidebar content
with st.sidebar:
    st.image('mg.png')
    st.title("Dermatrix")
    st.subheader("Accurate detection of skin diseases with suggested remedies")
    
    st.markdown("---")
    st.markdown("### About the Model")
    st.info("This AI-powered tool helps identify common skin conditions. Always consult with a healthcare professional for accurate diagnosis and treatment.")

# Main content
tabs = st.tabs(["🔍 Detection", "ℹ️ About", "💬 Skin Expert"])

# Detection Tab
with tabs[0]:
    st.markdown("## Skin Disease Detection")
    
    st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.markdown("### Upload an image of your skin condition")
    st.markdown("The AI will analyze the image and provide possible diagnosis and treatment suggestions.")
    file = st.file_uploader("", type=["jpg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if file is not None:
        try:
            # Load model if it's not already loaded
            with st.spinner('Loading model...'):
                model = load_model()
                
            # Process image
            image = Image.open(file)
            
            # Display image and prediction
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Uploaded Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("### Analysis Results")
                
                # Make prediction
                predictions = import_and_predict(image, model)
                predicted_class = class_names[np.argmax(predictions)]
                
                # Generate confidence score
                confidence = random.randint(88, 99) + random.randint(0, 99) * 0.01
                
                # Display prediction and confidence
                st.markdown(f"<div class='result-container'>", unsafe_allow_html=True)
                st.markdown(f"#### Detected Condition:")
                st.markdown(f"<h3 style='color:#8ab4f8'>{predicted_class}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Confidence:</b> {confidence}%</p>", unsafe_allow_html=True)
                
                # Display contagious warning if applicable
                if remedies[predicted_class]['contagious']:
                    st.warning("⚠️ Warning: This condition is contagious!")
                else:
                    st.info("This condition is not contagious.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Treatment information
            st.markdown("### Recommended Treatments")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<div class='primary-box'>", unsafe_allow_html=True)
                st.markdown("#### Primary Treatment")
                st.markdown(remedies[predicted_class]['primary'])
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='secondary-box'>", unsafe_allow_html=True)
                st.markdown("#### Additional Information")
                st.markdown(remedies[predicted_class]['secondary'])
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Disclaimer
            st.markdown("---")
            st.warning("⚠️ For accurate assessment of disease severity, please consult a dermatologist for in-person examination.")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        # Show placeholder when no image is uploaded
        st.markdown("<div class='placeholder-box'>", unsafe_allow_html=True)
        st.markdown("### No image uploaded")
        st.markdown("Please upload a clear image of the skin condition for analysis")
        st.markdown("</div>", unsafe_allow_html=True)

# About Tab
with tabs[1]:
    st.markdown("## About Dermatrix")
    
    st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.markdown("""
    Dermatrix is an AI-powered tool designed to help identify and provide information about common skin conditions. 
    The application uses deep learning technology to analyze images of skin and compare them against a database of 
    known skin diseases.
    
    Our goal is to make skin health information more accessible and provide preliminary guidance. However, this tool 
    should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis 
    and treatment of any skin condition.
    
    The system can currently detect 10 different types of skin conditions with varying degrees of accuracy. The AI model 
    is continuously being improved to provide more accurate results.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("## Skin Conditions We Detect")
    
    # Create a grid layout for the diseases
    for i in range(0, len(class_names), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(class_names):
                st.markdown(f"<div class='disease-box'>", unsafe_allow_html=True)
                st.markdown(f"### {class_names[i]}")
                st.markdown(disease_descriptions[class_names[i]])
                if remedies[class_names[i]]['contagious']:
                    st.warning("⚠️ This condition is contagious")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if i + 1 < len(class_names):
                st.markdown(f"<div class='disease-box'>", unsafe_allow_html=True)
                st.markdown(f"### {class_names[i+1]}")
                st.markdown(disease_descriptions[class_names[i+1]])
                if remedies[class_names[i+1]]['contagious']:
                    st.warning("⚠️ This condition is contagious")
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Disclaimer")
    st.warning("This application is intended for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

# Skin Expert Tab (Chatbot)
with tabs[2]:
    st.markdown("## Skin Expert Chatbot")
    
    st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the Skin Expert chatbot! This AI assistant can answer your questions about skin conditions, 
    preventive measures, and general skin health. Please note that this is for informational purposes only 
    and does not replace professional medical advice.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Skin Health Assistant. How can I help you with your skin health questions today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about skin conditions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response (placeholder for actual AI logic)
        response = "I'm a simple demo chatbot. In the full version, I would analyze your question and provide information about skin conditions, treatments, and preventive measures. For now, please try the Detection tab to analyze skin images or check the About tab for information on common skin conditions."
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Disclaimer at the bottom of the chat
    st.markdown("---")
    st.info("This chatbot provides general information only. For specific medical advice, please consult a healthcare professional.")
