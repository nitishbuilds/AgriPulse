import streamlit as st


def render_contact_us():
    if st.session_state.get("show_contact_us", False):
        @st.dialog("Contact Us")
        def contact_us_dialog():
            st.write("AgriPulse team se sampark karne ke liye form bharein.")
            contact_name = st.text_input("Your name", key="contact_name")
            contact_email = st.text_input("Email address", key="contact_email")
            contact_subject = st.selectbox(
                "Subject",
                ["General Enquiry", "Technical Support", "Crop / Pest Assistance", "Feedback"],
                key="contact_subject",
            )
            contact_message = st.text_area("Your message", key="contact_message")

            if st.button("Submit", type="primary", use_container_width=True, key="contact_submit"):
                if contact_name.strip() and contact_email.strip() and contact_message.strip():
                    st.success("Form validated. Email/database delivery abhi configure nahi hai.")
                else:
                    st.warning("Please name, email aur message fill karein.")

            if st.button("Close", use_container_width=True, key="contact_close"):
                st.session_state["show_contact_us"] = False
                st.rerun()

        # Consume the one-shot open request. Unrelated full-app reruns (e.g. theme toggle)
        # must not reopen this dialog.
        st.session_state["show_contact_us"] = False
        contact_us_dialog()
