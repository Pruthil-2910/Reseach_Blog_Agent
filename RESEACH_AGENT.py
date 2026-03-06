import os
import json
from typing import List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# 1. Define the state for the graph
class AgentState(TypedDict):
    topic: str
    research_queries: List[str]
    gathered_info: List[str]
    outline_sections: List[str]
    current_section_idx: int
    article_sections: List[str]
    final_article: str

# 2. Add node functions
def generate_queries(state: AgentState):
    topic = state['topic']
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
    prompt = f"You are an expert researcher. Generate 5 distinct search queries to deeply research the topic: '{topic}'. Return only the queries separated by newlines, no numbers."
    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    return {"research_queries": queries}

def conduct_research(state: AgentState):
    ddg = DuckDuckGoSearchAPIWrapper(max_results=5)
    gathered = []
    for query in state['research_queries']:
        try:
            results = ddg.run(query)
            gathered.append(f"### Query: {query}\n{results}")
        except Exception as e:
            pass
    return {"gathered_info": state.get('gathered_info', []) + gathered}

def create_outline(state: AgentState):
    topic = state['topic']
    info = "\n\n".join(state["gathered_info"])
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
    
    prompt = f"""Based on the following research about '{topic}', create a comprehensive outline for a Medium article that takes 10+ minutes to read.
    The article requires EXTREME technical and mathematical depth. Break it down into 6-8 comprehensive sections.
    Include sections that explicitly dive into the underlying mathematical formulas, models, proofs, or complex implementations algorithmically where applicable.
    
    Output the outline as a valid JSON list of strings. Example: ["Introduction", "Mathematical Foundations", "Algorithmic Implementation", "Conclusion"]
    
    Research material:
    {info[:15000]}
    """
    response = llm.invoke(prompt)
    try:
        content = response.content.replace('```json', '').replace('```', '').strip()
        sections = json.loads(content)
    except Exception:
        sections = [s for s in response.content.split('\n') if s.strip()]
    return {"outline_sections": sections, "current_section_idx": 0, "article_sections": []}

def write_section(state: AgentState):
    idx = state["current_section_idx"]
    section_topic = state["outline_sections"][idx]
    info = "\n\n".join(state["gathered_info"])
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)
    
    prompt = f"""You are an expert technical writer drafting a comprehensive Medium article piece by piece.
    Topic of the whole article: {state['topic']}
    Please write the complete content for the following section: "{section_topic}"
    
    REQUIREMENTS:
    - Make this section incredibly detailed, informative, and engaging (around 400-800 words).
    - INCORPORATE DEEP MATHEMATICS: Where relevant, use mathematical formulas (format them using LaTeX/Markdown syntax like $$ formula $$ or inline $ formula $). Explain the math clearly in-depth.
    - Dive into deep technical algorithmic implementations and complex theory if it applies to this section.
    - Use clear markdown formatting (bolding, bullet points where necessary).
    - Do not include main title header (H1) since this is just a section. Use H2 (##) for the section header.
    
    Use the following research context to inform your writing:
    {info[:20000]}
    
    Output ONLY THE MARKDOWN CONTENT for this specific section, do not add introductory or concluding sentences unless it is the intro/conclusion section.
    """
    response = llm.invoke(prompt)
    new_sections = state.get("article_sections", []) + [response.content]
    return {"article_sections": new_sections, "current_section_idx": idx + 1}

def should_continue_writing(state: AgentState):
    if state["current_section_idx"] < len(state["outline_sections"]):
        return "write_section"
    return "compile_article"

def compile_article(state: AgentState):
    topic = state["topic"]
    title = f"# {topic.title()}\n\n"
    full_article = title + "\n\n".join(state["article_sections"])
    return {"final_article": full_article}

def review_outline(state: AgentState):
    # This acts purely as a pause breakpoint for the human-in-the-loop to edit the outline
    pass

# 3. Build Graph
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("conduct_research", conduct_research)
    workflow.add_node("create_outline", create_outline)
    workflow.add_node("review_outline", review_outline)
    workflow.add_node("write_section", write_section)
    workflow.add_node("compile_article", compile_article)

    workflow.set_entry_point("generate_queries")
    workflow.add_edge("generate_queries", "conduct_research")
    workflow.add_edge("conduct_research", "create_outline")
    workflow.add_edge("create_outline", "review_outline")
    workflow.add_edge("review_outline", "write_section")
    
    workflow.add_conditional_edges(
        "write_section",
        should_continue_writing,
        {
            "write_section": "write_section",
            "compile_article": "compile_article"
        }
    )
    workflow.add_edge("compile_article", END)
    
    # Add checkpointer and interrupt before REVIEWING the outline, not writing!
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["review_outline"])
