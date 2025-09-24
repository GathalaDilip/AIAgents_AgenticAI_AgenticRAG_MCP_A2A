# agent_langgraph.py
from langchain.agents import create_react_agent
from langchain.tools import BaseTool
from retriever import FaissRetriever
from generator import LocalGenerator
from google import genai
from langchain.llms.base import LLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from pydantic import Field
from dotenv import load_dotenv
import os
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


class RetrieverTool(BaseTool):
    """Wrap the FaissRetriever as a LangChain tool."""
    name: str = "retriever"
    description: str = "Retrieves relevant passages from the AmpD manual."

    def __init__(self, retriever: FaissRetriever):
        super().__init__()
        self._retriever = retriever

    def _run(self, query: str):
        hits = self._retriever.retrieve(query, top_k=10, rerank_k=5)
        contexts = []
        for h in hits:
            contexts.append(f"[Page {h['page']}] {h['text']}")
        return "\n\n".join(contexts)


def build_agent(retriever: FaissRetriever):
    """
    Build a LangChain REACT agent using Gemini LLM via google-genai.
    """
    tool = RetrieverTool(retriever)

    client = genai.Client(api_key=gemini_api_key)

   

    
    class GeminiLLM(LLM):
        """Custom LangChain LLM wrapper for Gemini via google-genai."""
        client: object = Field(...)
        model: str = Field(default="gemini-2.5-flash")

        def __init__(self, client, model="gemini-2.5-flash"):
            super().__init__(client=client, model=model)

        def _call(self, prompt, stop=None):
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text

        @property
        def _llm_type(self):
            return "gemini"

    llm = GeminiLLM(client)

    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question and elaborate if needed

        Question: {input}
        {agent_scratchpad}"""
    )

    agent = create_react_agent(llm=llm, tools=[tool], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=True)
    return agent_executor
