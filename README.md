# Deep Math Research Agent 📚✍️

A LangGraph Multi-Agent system that researches a given topic extensively on the web and creates an intelligent outline. After confirmation/adjustment from the user through an interactive UI, it then generates a highly detailed Medium-styled technical article with deep mathematical formulas to support it!

## Features

- **Model Selection UI**: Choose between powerful LLM models directly in the sidebar!
  - `Gemini 3 Flash`
  - `Gemini 2.5 Flash`
  - `Gemini 2.5 Flash Lite`
  - `Gemini 3.1 Flash Lite`
  - `Groq openai/gpt-oss-120b`
- **Dynamic API Key Input**: Securely insert your Google GenAI or Groq key depending on the model you want to run.
- **Deep Mathematical Emphasis**: Hard-prompts the AI to utilize LaTeX and Markdown to deeply explain formulas related to the research topic.
- **Human-in-the-Loop Outline Review**: You have complete control to add, edit, or remove sections from the planned outline before generation begins.
- **Downloadable Markdowns**: Safely compile your generation into a ready-to-go `.md` file!
