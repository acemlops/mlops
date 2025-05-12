import streamlit as st

st.set_page_config(page_title="About Us", layout="centered")

st.title("👥 About Us")

st.markdown("Team GreenShield")

# Create columns for team members
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Team Member 1
with col1:
    st.image("/Users/atharvasatishchougale/Downloads/WhatsApp Image 2025-05-13 at 00.28.14.jpeg", caption="Rameshwari Badarkhe", width=200)
    st.markdown("""
    **📧** badarkherame18@gmail.com 
    **📱** +91-8767452181 
    """)

# Team Member 2
with col2:
    st.image("/Users/atharvasatishchougale/Library/CloudStorage/GoogleDrive-atharvachougale99@gmail.com/My Drive", caption="Atharva Chougale", width=200)
    st.markdown("""
    **📧** atharvachougale99@gmail.com 
    **📱** +91-9156523207  
    """)

# Team Member 3
with col3:
    st.image("/Users/atharvasatishchougale/Downloads/WhatsApp Image 2025-05-12 at 23.58.05.jpeg", caption="Raghav Dashrath", width=200)
    st.markdown("""
    **📧** [carol@example.com](mailto:carol@example.com)  
    **📱** +91-7058524533 
    """)

# Team Member 4
with col4:
    st.image("/Users/atharvasatishchougale/Downloads/WhatsApp Image 2025-05-12 at 23.56.45.jpeg", caption="Atharva Bhaleghare", width=200)
    st.markdown("""
    **📧** atharvbhaleghare@gmail.com 
    **📱** +91-9322617537
    """)

st.markdown("---")
st.markdown("🛠️ Built with ❤️ by the GreenShield Team")
