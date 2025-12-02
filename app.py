import streamlit as st
import pandas as pd
import google.generativeai as genai
import json

# =====================
# 🇯🇵 JLPT AI Sentence Generator (Python SDK Version)
# =====================
st.set_page_config(layout="wide", page_title="JLPT Vocabulary AI Sentence Generator", page_icon="🇯🇵")

# --- Constants ---
FILE_PATH = "jlpt_vocab.csv" 
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# =====================
# Load Data
# =====================
@st.cache_data
def load_vocab():
    """Loads and returns the vocabulary DataFrame."""
    try:
        df = pd.read_csv(FILE_PATH)
        df.rename(columns={'Original': 'Word (Kanji)', 'Furigana': 'Reading (Kana)', 'English': 'Meaning'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{FILE_PATH}' was not found. Please ensure it is uploaded.")
        return pd.DataFrame()
    except KeyError:
        st.error("Error: CSV column names do not match expected format (Original, Furigana, English, JLPT Level).")
        return pd.DataFrame()

# =====================
# Gemini API Logic
# =====================
def generate_sentences(word, api_key):
    """
    Calls the Gemini API to generate MULTIPLE example sentences.
    """
    try:
        genai.configure(api_key=api_key)
        
        # Updated Schema to accept an ARRAY of examples
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "examples": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "sentence_kanji": { "type": "STRING" },
                            "sentence_kana": { "type": "STRING" },
                            "english": { "type": "STRING" },
                            "context": { "type": "STRING", "description": "Brief context tag (e.g., Formal, Casual, Written)" }
                        },
                        "required": ["sentence_kanji", "sentence_kana", "english", "context"]
                    }
                }
            },
            "required": ["examples"]
        }

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction="""
            You are a strict Japanese language teacher. 
            Generate 3 DISTINCT, natural N2-level example sentences for the target word.
            - Vary the context (e.g., one formal, one casual, one business/written).
            - The 'sentence_kanji' MUST use appropriate Kanji.
            - The 'sentence_kana' must be the exact reading.
            """,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )

        prompt = f"Generate 3 example sentences for the word: '{word}'"
        response = model.generate_content(prompt)
        return json.loads(response.text)

    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# =====================
# Main App
# =====================
def app():
    df = load_vocab()
    if df.empty:
        return

    st.title("🇯🇵 JLPT Vocabulary AI Study Tool")
    st.markdown("Select a word, generate sentences, and test your reading skills!")

    # Sidebar
    st.sidebar.header("Configuration")
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password",
        value=st.session_state['api_key'],
        help="Paste your Gemini API key here."
    )
    st.session_state['api_key'] = api_key 

    # Level Select
    levels = sorted(df['JLPT Level'].unique().tolist())
    selected_level = st.sidebar.selectbox(
        "Choose Level:",
        options=levels,
        index=levels.index('N2') if 'N2' in levels else 0
    )
    
    # Filter
    filtered_df = df[df['JLPT Level'] == selected_level].copy()
    
    # Layout: Table on top (collapsed)
    with st.expander(f"📚 View {selected_level} Word List ({len(filtered_df)} words)"):
        st.dataframe(
            filtered_df[['Word (Kanji)', 'Reading (Kana)', 'Meaning']], 
            use_container_width=True, 
            hide_index=True
        )

    st.divider()

    # Main Interaction Area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1️⃣ Select Word")
        word_options = ["— Select a Word —"] + filtered_df['Word (Kanji)'].tolist()
        selected_word = st.selectbox("Choose a target word:", options=word_options, index=0, label_visibility="collapsed")
        
        # Display basic info about selected word immediately if selected
        if selected_word != "— Select a Word —":
            word_info = filtered_df[filtered_df['Word (Kanji)'] == selected_word].iloc[0]
            st.caption(f"Dictionary Meaning: {word_info['Meaning']}")

    with col2:
        st.subheader("2️⃣ Generate Context")
        generate_btn = st.button(
            f"✨ Generate Sentences", 
            type="primary", 
            disabled=(selected_word == "— Select a Word —"),
            use_container_width=True
        )

    # Output Area
    if generate_btn:
        if not api_key:
            st.error("🔑 Please enter your API Key in the sidebar.")
        elif selected_word == "— Select a Word —":
            st.warning("👆 Please select a word first.")
        else:
            # Modern loading status
            with st.status("🤖 AI is crafting multiple examples...", expanded=True) as status:
                result = generate_sentences(selected_word, api_key)
                status.update(label="✅ Sentences Ready!", state="complete", expanded=False)
            
            if result and 'examples' in result:
                st.divider()
                st.markdown(f"### 🎯 Target: :green[{selected_word}]")
                
                # Create Tabs for the different examples
                examples = result['examples']
                tabs = st.tabs([f"Example {i+1} ({ex.get('context', 'General')})" for i, ex in enumerate(examples)])
                
                for i, tab in enumerate(tabs):
                    ex = examples[i]
                    with tab:
                        with st.container(border=True):
                            # Main Kanji Sentence
                            st.success(f"**{ex['sentence_kanji']}**", icon="🇯🇵")
                            
                            # Hidden Details
                            with st.expander("👀 Reveal Reading & Translation", expanded=False):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.info(f"**Reading:**\n\n{ex['sentence_kana']}")
                                with c2:
                                    st.warning(f"**Meaning:**\n\n{ex['english']}")

if __name__ == "__main__":
    app()