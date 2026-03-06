import os
import uuid
import streamlit as st
import pandas as pd
from RESEACH_AGENT import build_graph

# --- Streamlit UI Build ---
st.set_page_config(page_title="Deep Math Research Agent", page_icon="📝", layout="wide")

st.title("📚 Deep Math & Research Article Writer ✍️")
st.markdown("Enter a topic and provide your **Groq API key** in the sidebar. The AI agent will thoroughly research the topic online, draft an outline (which you can review and edit), and iteratively write an in-depth article with **deep mathematical formula support**.")

# Initialize Session States
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent_app" not in st.session_state:
    st.session_state.agent_app = build_graph()
if "phase" not in st.session_state:
    st.session_state.phase = "setup"
if "final_article" not in st.session_state:
    st.session_state.final_article = ""

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_model = st.selectbox(
        "Select Model",
        [
            "Gemini 3 Flash",
            "Gemini 2.5 Flash",
            "Gemini 2.5 Flash Lite",
            "Gemini 3.1 Flash Lite",
            "Groq openai gpt oss 120b"
        ]
    )
    
    api_key_label = "Groq API Key" if "Groq" in selected_model else "Gemini API Key"
    user_api_key = st.text_input(api_key_label, type="password", help=f"Provide your {api_key_label}")
    
    if user_api_key:
        if "Groq" in selected_model:
            os.environ["GROQ_API_KEY"] = user_api_key
        else:
            os.environ["GEMINI_API_KEY"] = user_api_key
            os.environ["GOOGLE_API_KEY"] = user_api_key
            
    st.session_state.selected_model = selected_model
        
    st.markdown("---")
    st.info("""
    **How this Agent works:**
    1. Generates 5 precise web search queries.
    2. Browses the web (via DuckDuckGo) aggregating information.
    3. Plans an extensive article outline.
    4. **(Human-in-the-loop)** You review, edit, or add sections to the outline on an interactive board!
    5. Iteratively writes each section using heavy mathematical depth and LaTeX formulas.
    6. Compiles an exhaustive final article ready to download!
    """)
    if st.button("Reset Session 🔄"):
        st.session_state.clear()
        st.rerun()

# --- App Logic based on Phase ---

if st.session_state.phase == "setup":
    topic_input = st.text_input("What topic would you like the agent to research?", placeholder="e.g. The detailed mathematics of the transformers attention mechanism")

    if st.button("Generate Outline 🧠", type="primary"):
        if not user_api_key:
            st.error("Please provide your API Key in the sidebar.")
        elif not topic_input:
            st.warning("Please enter a topic to research.")
        else:
            initial_state = {
                "topic": topic_input,
                "selected_model": st.session_state.selected_model,
                "research_queries": [],
                "gathered_info": [],
                "outline_sections": [],
                "current_section_idx": 0,
                "article_sections": [],
                "final_article": ""
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.status("Agent Pipeline Running...", expanded=True) as status:
                try:
                    # Stream until it hits the interrupt breakpoint (before write_section)
                    for step in st.session_state.agent_app.stream(initial_state, config):
                        for node_name, state_update in step.items():
                            if node_name == "generate_queries":
                                st.write(f"✅ Generated {len(state_update.get('research_queries', []))} deep search queries.")
                            elif node_name == "conduct_research":
                                st.write("✅ Conducted continuous web research and aggregated sources.")
                            elif node_name == "create_outline":
                                st.write(f"✅ Planned extensive article outline based on math and context.")
                    
                    status.update(label="Outline Ready for Review!", state="complete", expanded=False)
                    st.session_state.phase = "review"
                    st.rerun()
                except Exception as e:
                    status.update(label=f"An error occurred: {e}", state="error")
                    st.error(f"Error Details: {e}")

elif st.session_state.phase == "review":
    st.subheader("📋 Review and Edit Outline Board")
    st.markdown("Below is the planned outline structure. You can **edit section names directly in the table**, **add completely new sections** by clicking the bottom row, or hover and **delete sections**.")
    
    # Get current state from LangGraph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    current_state = st.session_state.agent_app.get_state(config).values
    current_outline = current_state.get("outline_sections", [])
    
    # Render Interactive DataFrame
    df = pd.DataFrame({"Section Title": current_outline})
    
    edited_df = st.data_editor(
        df, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=False
    )
    
    if st.button("Approve & Start Writing Article 🚀", type="primary"):
        updated_outline = edited_df["Section Title"].dropna().tolist()
        
        # Update the LangGraph state with user's modified outline!
        st.session_state.agent_app.update_state(config, {"outline_sections": updated_outline})
        st.session_state.phase = "writing"
        st.rerun()

elif st.session_state.phase == "writing":
    st.subheader("🖋️ Writing Article Sections...")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.status("Writing full mathematical article in depth...", expanded=True) as status:
        try:
            # Resume graph execution (streaming from none)
            final_state_snapshot = None
            for step in st.session_state.agent_app.stream(None, config):
                for node_name, state_update in step.items():
                    if node_name == "write_section":
                        idx = state_update.get('current_section_idx', 1)
                        latest_state = st.session_state.agent_app.get_state(config).values
                        total = len(latest_state.get('outline_sections', []))
                        section_name = latest_state['outline_sections'][idx - 1]
                        st.write(f"✏️ Wrote section {idx}/{total}: **{section_name}**")
                    elif node_name == "compile_article":
                        final_state_snapshot = state_update
                        st.write("🚀 Article compiled successfully into full Markdown!")
            
            if final_state_snapshot and "final_article" in final_state_snapshot:
                 st.session_state.final_article = final_state_snapshot["final_article"]
            else:
                 latest_state = st.session_state.agent_app.get_state(config).values
                 st.session_state.final_article = latest_state.get("final_article", "")
                 
            st.session_state.phase = "done"
            status.update(label="Article Generation Complete! 🎉", state="complete", expanded=False)
            st.rerun()
            
        except Exception as e:
            status.update(label=f"An error occurred: {e}", state="error")
            st.error(f"Error Details: {e}")

elif st.session_state.phase == "done":
    st.success("Your article is ready!")
    st.markdown("---")
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    latest_state = st.session_state.agent_app.get_state(config).values
    topic_input = latest_state.get("topic", "Generated_Article")
    
    if st.button("Reset Session 🔄"):
        st.session_state.clear()
        st.rerun()
        
    # Download button
    st.download_button(
        label="📥 Download Final Article as Markdown",
        data=st.session_state.final_article,
        file_name=f"{topic_input.replace(' ', '_')}_article.md",
        mime="text/markdown"
    )

    # Display the preview below
    st.markdown("### Preview")
    st.markdown(st.session_state.final_article)
