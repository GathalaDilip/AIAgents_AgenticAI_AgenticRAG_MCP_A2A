# chatbot.py
import streamlit as st
from retriever import FaissRetriever
from generator import LocalGenerator
from agent import build_agent
import json, os, csv, datetime
import google.genai.errors

# --- Config ---
INDEX_PATH = "index.faiss"
META_PATH = "meta.json"
IMAGE_DIR = "extracted_images"
FEEDBACK_CSV = "feedback.csv"

# --- Helpers ---
def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (user, bot, sources)
    if "retriever" not in st.session_state:
        st.session_state.retriever = FaissRetriever(index_path=INDEX_PATH, meta_path=META_PATH)
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(st.session_state.retriever)
    if "exited" not in st.session_state:
        st.session_state.exited = False
    if "memory" not in st.session_state:
        st.session_state.memory = []  # simple session memory (list of user queries)

def save_feedback(query, answer, sources, rating):
    exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "query", "answer", "sources", "rating"])
        writer.writerow([datetime.datetime.utcnow().isoformat(), query, answer, json.dumps(sources, ensure_ascii=False), rating])

def display_sources(sources):
    for s in sources:
        with st.expander(f"Page {s['page']} (score {s.get('rerank_score', 0):.3f})"):
            st.write(s["text"][:1000])
            image_paths = s.get("image_paths")
            if image_paths:
                for img_path in image_paths:
                    abs_path = img_path
                    if not os.path.isabs(img_path):
                        abs_path = os.path.join(IMAGE_DIR, os.path.basename(img_path))
                    if abs_path and os.path.exists(abs_path):
                        st.image(abs_path, caption=f"Image on page {s['page']}", use_container_width=True)



# --- Modern Streamlit UI ---

st.set_page_config(page_title="AmpD Enertainer Chatbot", layout="wide")
init_session()

# --- Welcome Message (show only if not exited) ---
if not st.session_state.exited:
    st.markdown("<h2 style='color:#0366d6'>Welcome to the AMPD Chatbot!</h2>", unsafe_allow_html=True)
    st.markdown("<b>Ask a question about the internal document (type 'exit' to quit):</b>", unsafe_allow_html=True)

# Sidebar branding and instructions
with st.sidebar:
    st.image(r".\logo.png", use_container_width=True)
    st.markdown("## AmpD Enertainer Chatbot")
    st.markdown("Ampd is electrifying construction. We replace diesel with smart battery storage and digital solutions to create automated, emission-free, and highly efficient worksites for a sustainable future.")
    st.markdown("---")
    st.markdown("**Feedback:** If you spot issues or want to suggest improvements, please contact support@ampd.energy.")


# Main UI
if st.session_state.exited:
    st.markdown("<h3 style='color:#6f42c1'>Goodbye!</h3>", unsafe_allow_html=True)
    st.stop()

st.markdown("### Ask a question about the manual:")
col_input, col_send = st.columns([0.85, 0.15])
with col_input:
    user_query = st.text_input("Type your question", key="user_query", label_visibility="collapsed")
with col_send:
    send_clicked = st.button("Send", use_container_width=True)

# Example questions (clickable)
st.markdown("#### Quick Example Questions")
example_questions = [
    "How do I connect the Enertainer to a mains supply?",
    "What is the emergency shutdown procedure?",
    "What does error code E12 mean?"
]
ex_col1, ex_col2, ex_col3 = st.columns(3)
for i, q in enumerate(example_questions):
    if [ex_col1, ex_col2, ex_col3][i].button(q, key=f"ex_{i}"):
        user_query = q
        send_clicked = True

# Handle query and show spinner
if user_query and send_clicked:
    if user_query.strip().lower() == "exit":
        st.session_state.exited = True
        st.markdown("<h3 style='color:#6f42c1'>Goodbye!</h3>", unsafe_allow_html=True)
        st.stop()
    st.session_state.memory.append(user_query)
    sources = st.session_state.retriever.retrieve(user_query, top_k=10, rerank_k=6)
    context = "\n\n".join([f"[Page {s['page']}] {s['text']}" for s in sources])
    agent_input = f"{user_query}\n\nContext:\n{context}"
    with st.spinner("Getting answer from Agent..."):
        try:
            answer = st.session_state.agent.invoke({"input": agent_input})
        except google.genai.errors.ClientError as e:
            if "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                answer = "[Gemini API quota exceeded. Please wait a minute and try again, or upgrade your plan for higher limits.]"
            else:
                answer = f"[Error: {e}]"
    st.session_state.history.append({"role": "user", "text": user_query})
    st.session_state.history.append({"role": "bot", "text": answer, "sources": sources})

# Show latest response in card style
if st.session_state.history:
    last_turn = st.session_state.history[-1]
    if last_turn["role"] == "bot":
        st.markdown("---")
        # Show only output if response is a dict with 'output' key
        bot_response = last_turn['text']
        if isinstance(bot_response, dict) and 'output' in bot_response:
            bot_response = bot_response['output']
        st.markdown(
            f"<div style='background-color:#f6f8fa; border-radius:8px; padding:16px; margin-bottom:8px;'>"
            f"<b>Bot:</b> {bot_response}"
            f"</div>", unsafe_allow_html=True)
        if last_turn.get("sources"):
            display_sources(last_turn["sources"])
        # Feedback buttons
        fb_col1, fb_col2 = st.columns([0.1, 0.1])
        if fb_col1.button("👍", key=f"up_{len(st.session_state.history)}"):
            save_feedback(user_query, last_turn.get("text",""), [s['page'] for s in last_turn.get("sources",[])], "up")
            st.success("Thanks for your feedback!")
        if fb_col2.button("👎", key=f"down_{len(st.session_state.history)}"):
            save_feedback(user_query, last_turn.get("text",""), [s['page'] for s in last_turn.get("sources",[])], "down")
            st.success("Thanks for your feedback!")

# Scrollable conversation history
with st.expander("Show full conversation history"):
    st.markdown("<div style='max-height:350px;overflow-y:auto;'>", unsafe_allow_html=True)
    for turn in st.session_state.history:
        if turn["role"] == "user":
            st.markdown(f"<span style='color:#0366d6'><b>You:</b> {turn['text']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#6f42c1'><b>Bot:</b> {turn['text']}</span>", unsafe_allow_html=True)
            if turn.get("sources"):
                display_sources(turn["sources"])
    st.markdown("</div>", unsafe_allow_html=True)
