import streamlit as st
from welcome.wel import display_welcome_page
from prediction.pred import display_prediction_page
from curing import display_curing_page  # Import the curing page logic
from dashboard.dash import display_dashboard_page  # Import the dashboard logic

# Page navigation logic
def main():
    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "Welcome"  # Default to Welcome page

    # Create a navigation bar
    st.sidebar.title("Navigation")
    pages = ["Welcome", "Prediction", "Dashboard"]  # Add Dashboard to pages
    st.session_state.page = st.sidebar.radio(
        "Go to:",
        options=pages,
        index=pages.index(st.session_state.page)
    )

    # Render selected page
    if st.session_state.page == "Welcome":
        display_welcome_page()
    elif st.session_state.page == "Prediction":
        display_prediction_page()
    elif st.session_state.page == "Dashboard":
        display_dashboard_page()  # Call the dashboard page function

if __name__ == "__main__":
    main()