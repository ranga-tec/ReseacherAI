import os
import json
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fpdf import FPDF
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field
from typing import Type, List

from langchain.agents import Tool, AgentExecutor
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

load_dotenv()

openai_api_key = os.getenv("openai_api_key")
browserless_api_key = os.getenv("browserless_api_key")
serper_api_key = os.getenv("serper_api_key")

# Tool for search
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    search_results = response.json()
    print(search_results)
    return search_results

# Tool for scraping
def scrape_website(url: str):
    print("Scraping website...")
    headers = {'Cache-Control': 'no-cache', 'Content-Type': 'application/json'}
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    print(response)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        print("CONTENT:", text)
        return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return ""

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for {objective}. The text is scraped data from a website so 
    will have a lot of useless information that doesn't relate to this topic, links, other news stories, etc. 
    Only summarize the relevant info and try to keep as much factual information intact:
    "{text}"
    DETAILED SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    output = summary_chain.run(input_documents=docs, objective=objective)
    print(output)
    return output

def save_summary_as_pdf(summaries: List[str], filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for summary in summaries:
        pdf.multi_cell(0, 10, summary)
        pdf.ln()
    pdf.output(filename)
    print(f"Saved summary to {filename}")

class ScrapeWebsiteInput(BaseModel):
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The URL of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Useful when you need to get data from a website URL, passing both URL and objective to the function; DO NOT make up any URL, the URL should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

# Create Langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="Useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a world-class researcher, who can do detailed research on any topic and produce facts-based results;
        you do not make things up, you will try as hard as possible to gather facts & data to back up the research
        
        Please make sure you complete the objective above with the following rules:
        1/ You should do enough research to gather as much information as possible about the objective
        2/ If there are URLs of relevant links & articles, you will scrape it to gather more information
        3/ After scraping & search, you should think "is there any new things I should search & scrape based on the data I collected to increase research quality?" If the answer is yes, continue; But don't do this more than 3 iterations
        4/ You should not make things up, you should only write facts & data that you have gathered
        5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        """),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
} | prompt | llm_with_tools | OpenAIToolsAgentOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

# API endpoint via FastAPI
app = FastAPI()

class Query(BaseModel):
    query: str

# Use Streamlit to create a web app
def main():
    st.set_page_config(page_title="AI Research Agent", page_icon=":bird:")
    st.header("AI Research Agent :bird:")
    query = st.text_input("Research goal")
    if query:
        st.write("Doing research for ", query)
        search_results = agent_executor({"input": query})
        
        # Print search results to understand its structure
        st.write("Search Results:", search_results)

        # Check the structure of search results and handle accordingly
        try:
            search_output = json.loads(search_results['output'])
            st.write("Parsed Search Output:", search_output)
        except Exception as e:
            st.error(f"Error parsing search results: {e}")
            return

        # Continue only if search results are structured correctly
        if isinstance(search_output, dict) and 'results' in search_output:
            st.info("Search completed. Scraping and summarizing content...")

            summaries = []
            for result in search_output['results']:
                url = result['link']
                content = scrape_website(url)
                summary_text = summary(query, content)
                summaries.append(summary_text)

            pdf_filename = "summary.pdf"
            save_summary_as_pdf(summaries, pdf_filename)
            st.success(f"Summary saved to {pdf_filename}")
        else:
            st.error("Unexpected structure in search results. Please check the output.")

if __name__ == '__main__':
    main()
