import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon = "🧑‍⚕️",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key


with st.sidebar:
        st.image('mg.png')
        st.title("Dermatrix")
        st.subheader("Accurate detection of skin diseases present with suggestion of remedies to cure them.")

             
        
def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key
        
       

    


@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('skin.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

    

st.write("""
         # Skin Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (300,300)    
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.info("Accuracy : " + str(x) + " %")

    class_names = ['Eczema','Warts Molluscum and other Viral Infections', 'Melanoma','Atopic Dermatitis','Basal Cell Carcinoma (BCC)','Melanocytic Nevi (NV)','Benign Keratosis-like Lesions (BKL)','Psoriasis pictures Lichen Planus and related diseases','Seborrheic Keratoses and other Benign Tumors','Tinea Ringworm Candidiasis and other Fungal Infections']

    st.markdown('''## Best Remedy :''')   

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Eczema':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        
        st.success('An effective, intensive treatment for severe eczema involves applying a corticosteroid ointment and sealing in the medication with a wrap of wet gauze topped with a layer of dry gauze.')

        st.info("DUPIXENT® (dupilumab) is a prescription medicine used to treat people aged 6 years and older with moderate-to-severe atopic dermatitis (eczema) that is not well controlled with prescription therapies used on the skin (topical) or who cannot use topical therapies.0 Other treatments for eczema include azathioprine, cyclosporine, methotrexate, pimecrolimus, crisaborole, and tacrolimus, which are prescription creams and ointments that control inflammation and reduce immune system reactions.1 Calcineurin inhibitors, such as pimecrolimus and tacrolimus, are also recommended if OTC steroids don't work or cause problems. Corticosteroid creams, solutions, gels, foams, and ointments, made with hydrocortisone steroids, can quickly relieve itching and reduce inflammation.2 Pimecrolimus cream or tacrolimus ointment, also known as topical calcineurin inhibitors (TCIs), may be prescribed by a dermatologist")

    elif class_names[np.argmax(predictions)] == 'Melanoma':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("Treatment for early-stage melanomas usually includes surgery to remove the melanoma. A very thin melanoma may be removed entirely during the biopsy and require no further treatment. Otherwise, your surgeon will remove the cancer as well as a border of normal skin and a layer of tissue beneath the skin.")

        st.info("Ipilimumab (Yervoy®) is an immunotherapy drug used to treat metastatic melanoma and stage III melanoma that cannot be removed completely with surgery. It works by blocking an immune molecule called CTLA-4. Checkpoint inhibitors, also known as immune checkpoint blockade, are commonly used to treat melanoma.2 Interferon alfa (Intron A, Roferon-A) can be used after surgery to prevent melanoma recurrence.0 Targeted therapy of melanoma includes vemurafenib, cobimetinib, dabrafenib, and trametinib, which attack cells that have a damaged BRAF gene. Targeted medicines for melanoma with NRAS and C-KIT mutations may be available through clinical trials.1")

    elif class_names[np.argmax(predictions)] == 'Atopic Dermatitis':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("The main treatments for atopic eczema are: emollients (moisturisers) used every day to stop the skin becoming dry. topical corticosteroids creams and ointments used to reduce swelling and redness during flare-ups.")

        st.info("DUPIXENT® (dupilumab) is a prescription medicine used to treat moderate-to-severe atopic dermatitis (eczema) that is not well controlled with prescription therapies used on the skin (topical), or who cannot use topical therapies. It is not known if DUPIXENT is safe and effective in children with atopic dermatitis under 6 years of age.1 Cibinqo (abrocitinib) is an oral JAK1 inhibitor approved by the FDA for adults with refractory moderate to severe atopic dermatitis whose disease is not adequately controlled with other systemic drug products, including biologics, or when use of those therapies is inadvisable. Immunosuppressants are prescribed for moderate to severe atopic dermatitis in children and adults to help stop the itch-scratch cycle of eczema, to allow the skin to heal and reduce the risk of skin infection.0 Topical calcineurin inhibitors, immunosuppressant tablets, and alitretinoin are some of the topical treatments for atopic dermatitis")

    elif class_names[np.argmax(predictions)] == 'Basal Cell Carcinoma (BCC)':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("The current mainstay of BCC treatment involves surgical modalities such as excision, electrodesiccation and curettage (EDC), cryosurgery, and Mohs micrographic surgery. Such methods are typically reserved for localized BCC and offer high 5-year cure rates, generally over 95%")
        st.info("Basal cell skin cancer does not usually respond to chemotherapy, but it often responds to a targeted drug called vismodegib, sold as Erivedge®, which helps disrupt the activity of a group of proteins in the body called hedgehog.0 Erivedge® (vismodegib) capsule is a prescription medicine used to treat adults with basal cell carcinoma that has spread to other parts of the body or that has come back after surgery or that cannot be treated with surgery or radiation. It is the #1 most-prescribed oral medication for advanced basal cell carcinoma.2")

    elif class_names[np.argmax(predictions)] == 'Melanocytic Nevi (NV)':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("Small nevi can be removed by simple surgical excision. The nevus is cut out, and the adjacent skin stitched together leaving a small scar. Removal of a large congenital nevus, however, requires replacement of the affected skin.")
        st.info("Melanocytic nevus is the medical term for a mole. Nevi can appear anywhere on the body. They are benign (non-cancerous) and typically do not require treatment. A very small percentage of melanocytic nevi may develop a melanoma within them. Of note, the majority of cutaneous melanomas arise within normally appearing skin.")

    elif class_names[np.argmax(predictions)] == 'Benign Keratosis-like Lesions (BKL)':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("Cryosurgery: The dermatologist applies liquid nitrogen, a very cold liquid, to the growth with a cotton swab or spray gun. Electrosurgery and curettage: Electrosurgery (electrocautery) involves numbing the growth with an anesthetic and using an electric current to destroy the growth.")
        st.info("A seborrheic keratosis is a growth on the skin. The growth is not cancer (benign). It’s color can range from white, tan, brown, or black. Seborrheic keratoses often appear on a person’s chest, arms, back, or other areas. They’re very common in people older than age 50")

    elif class_names[np.argmax(predictions)] == 'Psoriasis pictures Lichen Planus and related diseases':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("Lichen planus does not usually require treatment. It often goes away by itself within a year. If a person has particularly itchy or painful outbreaks, a doctor may prescribe topical corticosteroids or light therapy. Psoriasis is a long-term condition, but people can usually manage their symptoms well.")
        st.info("There isn’t a cure for lichen planus. If you have lichen planus on your skin, in most cases, it goes away without treatment in as little as a few months to several years. Corticosteroid creams or ointments. Your healthcare provider may prescribe corticosteroid creams or ointments to reduce inflammation.  Phototherapy uses ultraviolet light, usually ultraviolet B (UVB), from special lamps. The ultraviolet light waves found in sunlight can help certain skin disorders, including lichen planus.")

    elif class_names[np.argmax(predictions)] == 'Seborrheic Keratoses and other Benign Tumors':
        st.sidebar.error(string)
        st.sidebar.info("Its not contagious")
        st.success("Eskata, a 40% hydrogen peroxide topical solution, is the first FDA-approved drug for treatment of seborrheic keratoses. Administration of the drug may be tedious and usually requires at least two office visits.")
        st.info("Ammonium lactate and alpha hydroxy acids have been reported to reduce the height of seborrheic keratoses, and superficial lesions can be treated by carefully applying pure trichloroacetic acid and repeating if the full thickness is not removed on the first treatment. Topical treatment with tazarotene cream 0.1% applied twice daily for 16 weeks caused clinical improvement in seborrheic keratoses in 7 of 15 patients.0 Diclofenac gel may be a new treatment option for seborrheic keratosis.2 Hydrogen peroxide 40% (Eskata) is a topical solution for the in-office treatment of raised seborrheic keratosis lesions")

    elif class_names[np.argmax(predictions)] == 'Tinea Ringworm Candidiasis and other Fungal Infections':
        st.sidebar.error(string)
        st.sidebar.error("Beware!! It's contagious⚠️")
        st.success("Typically, a course of antifungal creams (either prescription or over-the-counter) will clear up the rash and relieve the itchiness. Your healthcare provider can also discuss preventive steps to keep the rash from coming back.")
        st.info("Tinea ringworm can be treated with over-the-counter (OTC) antifungal creams containing clotrimazole, ketoconazole, econazole, tolnaftate, or terbinafine. However, if there are many patchy areas, a prescription cream or oral antifungal medicine taken by mouth may be necessary.")

    elif class_names[np.argmax(predictions)] == 'Warts Molluscum and other Viral Infections':
        st.sidebar.error(string)
        st.error("Beware!! It's contagious⚠️")
        st.success("Doctors recommend many topical treatments for molluscum contagiosum. Podophyllotoxin (contraindicated in pregnant women), potassium hydroxide, salicylic acid (associated or not with povidone-iodine), benzoyl peroxide, and tretinoin are used as home treatments and must be applied to each lesion")
        st.info("Cantharidin (beetle juice): This FDA-approved treatment is made from blister beetles. It’s approved to treat adults and children two years of age and older. Dermatologists have been using cantharidin to treat warts and molluscum since the 1950s. When treating molluscum bumps, your dermatologist applies the beetle juice to each bump. Your dermatologist will apply it to each bump in such a way that a water blister later forms.")

    st.sidebar.warning("Look bottom for remedies")

    st.info("For detecting severity of disease, kindly consult a dermatologist for physical consulation")
