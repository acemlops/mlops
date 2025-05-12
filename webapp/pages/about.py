import streamlit as st

st.set_page_config(page_title="About Us", layout="centered")

st.title("👥 About Us")
st.markdown("### Team GreenShield")

# Create columns for team members
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Team Member 1
with col1:
    st.image(
        "https://raw.githubusercontent.com/acemlops/mlops/e8ef49126d0940d0ed874f7d543cc0a6664765bd/Rami.jpeg",
        caption="Rameshwari Badarkhe",
        width=200
    )
    st.markdown("""
    📧 [badarkherame18@gmail.com](mailto:badarkherame18@gmail.com)  
    📱 [📞 +91-8767452181](tel:+918767452181)
    """)

# Team Member 2
with col2:
    st.image(
        "https://raw.githubusercontent.com/acemlops/mlops/e8ef49126d0940d0ed874f7d543cc0a6664765bd/AtharvaC.jpg",
        caption="Atharva Chougale",
        width=200
    )
    st.markdown("""
    📧 [atharvachougale99@gmail.com](mailto:atharvachougale99@gmail.com)  
    📱 [📞 +91-9156523207](tel:+919156523207)
    """)

# Team Member 3
with col3:
    st.image(
        "https://raw.githubusercontent.com/acemlops/mlops/e8ef49126d0940d0ed874f7d543cc0a6664765bd/Raghav.jpeg",
        caption="Raghav Dashrath",
        width=200
    )
    st.markdown("""
    📧 [raghavdashrath10@gmail.com](mailto:raghavdashrath10@gmail.com)  
    📱 [📞 +91-7058524533](tel:+917058524533)
    """)

# Team Member 4
with col4:
    st.image(
        "https://raw.githubusercontent.com/acemlops/mlops/e8ef49126d0940d0ed874f7d543cc0a6664765bd/AtharvaB.jpeg",
        caption="Atharva Bhaleghare",
        width=200
    )
    st.markdown("""
    📧 [atharvbhaleghare@gmail.com](mailto:atharvbhaleghare@gmail.com)  
    📱 [📞 +91-9322617537](tel:+919322617537)
    """)

st.markdown("---")
st.markdown("🛠️ Built with ❤️ by the GreenShield Team")
