import streamlit as st
from streamlit_option_menu import option_menu
import home
import detection
import visualization

# Set page configuration with wide layout, page title, and icon
st.set_page_config(layout="wide", page_title="AI Image Detector", page_icon="📸")

# Custom CSS to disable scrolling in the sidebar and reduce empty space
st.markdown("""
            <style>
            .css-1iyw2u1 {
                display: none;
            }
                
            /* Import Montserrat font */
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap');
            
            /* Apply Montserrat font to titles */
            .title-font {
                font-family: 'Montserrat', sans-serif;
                font-size: 2em;
                color: white;
            }
            
            .sub-title {
                font-family: 'Montserrat', sans-serif;
                font-size: 1.6em;
                color: white;
            }
            
            .custom-font{
                font-family: 'Montserrat', sans-serif;
                font-size: 1.5em;
                color: white;
            }
            
            * {
                font-family: 'Montserrat', sans-serif;
            }
            
            /* Adjust sidebar width */
            .css-1d391kg {
                width: 500px; /* Set desired width */
            }
            
            /* Optionally, you can set a minimum width */
            .css-1d391kg {
                min-width: 300px; /* Set minimum width */
            }
            
            .main {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border-radius: 15px;
                padding: 30px;
                max-width: 1600px;
                margin-bottom: 32px;
                margin-left: auto;
                margin-right: auto;
                font-family: 'Montserrat', sans-serif;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            }
            
            .main h2 {
                color: #ff2546;
                font-family: 'Montserrat', sans-serif;
            }
            .main p {
                font-size: 18px;
                font-family: 'Montserrat', sans-serif;
                text-align: justify;
            }

            /* Feature box styling */
            .feature-box {
                background-color: #212121;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            
            }
            
            .stTabs [data-baseweb="tab"] {
                color: #ffffff;
                font-size: 32px; /* Maximize font size */
                font-weight: bold; /* Make the font bold */
                padding: 15px 25px; /* Increase padding for better spacing */
                margin-right: 10px; /* Space between tabs */
                cursor: pointer; /* Pointer cursor for better UX */
                font-family: 'Montserrat', sans-serif;
            }

            /* Style for the active tab */
            .stTabs [aria-selected="true"] {
                color: #ff2546; /* Active tab text color */
                font-size: 50px;
                font-family: 'Montserrat', sans-serif;
            }

            /* Hover effect for inactive tabs */
            .stTabs [data-baseweb="tab"]:hover {
                color: #b82d6a;
                font-size: 32px;
                font-family: 'Montserrat', sans-serif;
            }
            
            .dmain {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border-radius: 10px;
                padding: 20px;
                max-width: 1200px;
                margin-bottom: 20px;
                margin-left: auto;
                margin-right: auto;
                margin-top: 10px;
                font-family: 'Montserrat', sans-serif;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            }
            
            .fake-result{
                text-align: center; 
                background-color: #ffcccc; 
                padding: 10px; 
                border-radius: 5px; 
                font-family: 'Montserrat', sans-serif;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            
            .real-result{
                text-align: center; 
                background-color: #ccffcc; 
                padding: 10px; 
                border-radius: 10px; 
                font-family: 'Montserrat', sans-serif;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            
            /* Expander styling */
            .streamlit-expanderHeader {
                background-color: #ff2546; /* Header background color */
                color: #ffffff; /* Header text color */
                font-weight: bold; /* Header text weight */
                font-family: 'Montserrat', sans-serif;
                padding: 10px; /* Padding for the header */
                border-radius: 10px; /* Rounded corners for the header */
                border: 1px solid white; /* Border color for the header */
            }
        
            .streamlit-expanderContent {
                background-color: #ffffff; /* Content background color */
                color: #333333; /* Content text color */
                font-family: 'Montserrat', sans-serif;
                padding: 15px; /* Padding for the content */
                border-radius: 10px; /* Rounded corners for the content */
                border: 1px solid white; /* Border color for the content */
            }
        
            .streamlit-expanderContent ul {
                list-style-type: disc; /* Bullet style for lists */
                margin-left: 20px; /* Indentation for lists */
                font-family: 'Montserrat', sans-serif;
            }
            
            .metric-card {
                background-color: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .reason-box {
                background-color: rgba(220, 57, 131, 0.1);
                border-left: 4px solid #ff2546;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            
            </style>
            """, 
            unsafe_allow_html=True
            )

# MultiApp class to manage multiple applications
class MultiApp:
    def __init__(self):
        self.apps = []

    # Use for add a new application to the list
    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    # Run the selected application
    def run(self):
        with st.sidebar:
            selected_app = option_menu(
                menu_title="📸 AI Image Detector",
                options=[app["title"] for app in self.apps],
                icons=["house", "camera", "bar-chart-line"],
                menu_icon="",
                default_index=0, # Default selected app
                styles={
                    "container": {
                        "padding": "10px 5px", 
                        "background-color": "rgba(255, 255, 255, 0.1)",
                        "backdrop-filter": "blur(10px)",
                        "border-radius": "10px",
                        "border": "2px solid white",
                        "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2)",
                        "font-family": "'Montserrat', sans-serif",
                    },
                    "icon": {
                        "color": "#ffffff",  
                        "font-size": "20px",  
                        "margin-right": "10px",
                    },
                    "menu-icon": {
                        "display": "none",
                    },
                    "nav-link": {
                        "font-size": "18px",
                        "color": "#fff",
                        "--hover-color": "rgba(255, 255, 255, 0.1)",
                        "margin-bottom": "15px",
                    },
                    "nav-link-selected": {
                        "background-color": "#ff2546",
                        "color": "white",
                    },
                    
                }
            )
            # Give padding for the main content container
            st.markdown('<style>div.block-container{padding-top:3.2rem;}</style>', unsafe_allow_html=True)
            
            # Add container for display the social media links
            with st.container():
                st.markdown(
                    """
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; margin-top: 40px; border-radius: 10px; border: 2px solid white; backdrop-filter: blur(10px); box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); font-family: 'Montserrat', sans-serif;">
                    <p><strong>Connect with me:</strong></p>
                    <ul style="list-style-type: none; padding-left: 0; align-item: center;">
                        <li>
                        <div style="display: inline-block; background-color: rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 5px; margin-left: -10px;">
                            <img src="https://cdn.icon-icons.com/icons2/936/PNG/512/github-logo_icon-icons.com_73546.png" alt="GitHub" 
                            style="width: 20px; height: 20px; vertical-align: middle;">
                            <a href="https://github.com/KithminiRanasinghe" target="_blank" style= "color: #ffffff;"> @KithminiRanasinghe (GitHub)</a>
                        </div>
                        </li>
                        <li style="margin-top: 20px;">
                        <div style="display: inline-block; background-color: rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 5px; margin-left: -12px;">
                            <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/linkedin_logo_icon_170234.png" alt="LinkedIn" 
                            style="width: 20px; height: 20px; vertical-align: middle;">
                            <a href="https://www.linkedin.com/in/kithmini-ranasinghe-866a99253/" target="_blank" style= "color: #ffffff;"> Kithmini Ranasinghe (LinkedIn)</a>
                        </div>
                        </li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display the version of the web-app
                st.markdown(
                    """
                    <div style="text-align: center; padding: 10px 0; margin-bottom: -100px; font-family: 'Montserrat', sans-serif;">
                    <p style="color: #0F0F0F; background-color: rgba(255, 255, 255, 0.9); padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 40px; backdrop-filter: blur(10px); box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);">
                        AI Image Detector - Version 1.0</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

        # Run the selected app from the sidebar menu
        for app in self.apps:
            if app["title"] == selected_app:
                app["function"]()

# Add the individual apps to the MultiApp instance
app = MultiApp()
app.add_app("Home", home.app)
app.add_app("Detection", detection.app)
app.add_app("Visualization", visualization.app)
app.run()