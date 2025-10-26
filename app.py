import streamlit as st
from urllib.parse import unquote
from predict_email import extract_entities,predict

h=250
st.title("📧 📬 Spam Screener")
# Read email from text input
email_text = st.text_area("Paste your email content here:",width=h*4,height=h)

button_pressed=st.button("predict email")
if button_pressed:
    if not email_text:
        email_text = st.text_area("Paste your email content here:")

    if email_text:

        if predict(str(email_text))==1:
            st.warning('SPAM')
        else:
            st.success('Secure ✅')
        st.subheader("📍 Named Entities Found")
        entities = extract_entities(unquote(email_text))

        if entities:
            for text, label in entities:
                st.markdown(f"- **{label}**: `{text}`")

        else:
            st.info("No named entities found.")
