from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

def get_openai_model():
    # Try to get API key from environment (already set by main.py)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found, try Streamlit secrets as fallback
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                api_key = st.secrets.get("OPENAI_API_KEY")
        except ImportError:
            pass
    
    # If still not found, try loading from .env
    if not api_key:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please check your configuration.")
        
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7
    )
    return llm