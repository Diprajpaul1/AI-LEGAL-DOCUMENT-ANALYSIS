from crewai import Agent, LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0
)

agent = Agent(
    role= "template Agent",
    goal="",
    backstory="",
    llm= llm,
    tools=[],
    verbose=True
)
