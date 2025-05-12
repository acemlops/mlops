import streamlit as st

st.set_page_config(page_title="About Us", layout="centered")

st.title("👥 About Us")

st.markdown("### Meet Our Team")

# Create columns for team members
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Team Member 1
with col1:
    st.image("https://example.com/photo1.jpg", caption="Alice Johnson", width=200)
    st.markdown("""
    **📧** [alice@example.com](mailto:alice@example.com)  
    **📱** +91-1234567890  
    **🌐** [LinkedIn](https://www.linkedin.com/in/alicejohnson)  
    """)

# Team Member 2
with col2:
    st.image("https://example.com/photo2.jpg", caption="Bob Smith", width=200)
    st.markdown("""
    **📧** [bob@example.com](mailto:bob@example.com)  
    **📱** +91-2345678901  
    **🌐** [LinkedIn](https://www.linkedin.com/in/bobsmith)  
    """)

# Team Member 3
with col3:
    st.image("https://example.com/photo3.jpg", caption="Carol Singh", width=200)
    st.markdown("""
    **📧** [carol@example.com](mailto:carol@example.com)  
    **📱** +91-3456789012  
    **🌐** [LinkedIn](https://www.linkedin.com/in/carolsingh)  
    """)

# Team Member 4
with col4:
    st.image("https://example.com/photo4.jpg", caption="David Kumar", width=200)
    st.markdown("""
    **📧** [david@example.com](mailto:david@example.com)  
    **📱** +91-4567890123  
    **🌐** [LinkedIn](https://www.linkedin.com/in/davidkumar)  
    """)

st.markdown("---")
st.markdown("🛠️ Built with ❤️ by the GreenShield Team")
