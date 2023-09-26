from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
# load_dotenv()
# import os
# serper_api_key = os.getenv('SERPAPI_API_KEY')
# brwoserless_api_key = os.getenv('BROWSERLESS_API_KEY')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# print(OPENAI_API_KEY)
# 1. Tool for search


import streamlit as st
st.title("Search for the VC funds for your company")
st.info("Just put in your industry an I will find you a bunch of vcs and investors in the industry")
with st.expander('Input your keys:'):
    browserless_api_key = st.text_input('BROWSERLESS, go to https://www.browserless.io/ to get a key')
    OPENAI_API_KEY = st.text_input('OPENAI KEY, you know it already, or go to playground.openai.com')
    serper_api_key = st.text_input('SERPER API KEY, can be found here: https://serpapi.com/search-api')
    
def init():


    def search(query):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })

        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)

        return response.text

    import base64
    # 2. Tool for scraping
    def scrape_website(objective: str, url: str, BROWSERLESS_API=browserless_api_key):
        # scrape website, and also will summarize the content based on objective if the content is too large
        # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

        print("Scraping website...")
        # Define the headers for the request
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }
        encoded_token = base64.b64encode(BROWSERLESS_API.encode()).decode()

        # Define the data to be sent in the request
        data = {
            "url": url
        }

        # Convert Python object to JSON string
        data_json = json.dumps(data)

        # Send the POST request
        post_url = "https://chrome.browserless.io/content"
        headers['Authorization'] = f'Basic {encoded_token}'
        response = requests.post(post_url, headers=headers, data=data_json)

        # Check the response status code
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            body_tag = soup.find('body')
            if body_tag:
                body_text = body_tag.get_text()
            else:
                return
            output = split_and_summarize(body_text[:9000], objective)
            return output
        else:
            print('HTTP RESPONSE FAILED WITH ERROR CODE:', response.status_code)


    def split_and_summarize(text, objective):
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)

        # Split the content into smaller chunks
        chunks = custom_chunk_splitter(text, 3000)
        # Initialize an empty list to store summary chunks
        summary_chunks = []

        # Process each chunk
        for chunk in chunks:
            # Generate a summary for the chunk
            summary_chunk = generate_summary(llm, objective, chunk)

            # Append the summary chunk to the list
            summary_chunks.append(summary_chunk)

        # Combine the summary chunks into a single summary
        full_summary = "\n".join(summary_chunks)

        return full_summary
    def custom_chunk_splitter(text, max_chunk_length):
        """
        Split a text into smaller chunks based on a maximum character length.

        Args:
            text (str): The input text to be split.
            max_chunk_length (int): The maximum character length for each chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        current_chunk = ""

        for word in text.split():
            if len(current_chunk) + len(word) <= max_chunk_length:
                current_chunk += word + " "
            else:
                # Add the current chunk to the list of chunks
                chunks.append(current_chunk.strip())
                # Start a new chunk with the current word
                current_chunk = word + " "

        # Add the last chunk (if any)
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    def generate_summary(llm, objective, text_chunk):

        map_prompt = """
        Write a short 600 word summary of the following text for {objective}:
        "{text_chunk}"
        SUMMARY:
        """
        prompt = PromptTemplate(template=map_prompt, input_variables=["objective","text_chunk"])
        # Generate the summary using the model
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({"objective":objective,"text_chunk":text_chunk})

        # Extract and return the summary from the response

        return response




    class ScrapeWebsiteInput(BaseModel):
        """Inputs for scrape_website"""
        objective: str = Field(
            description="The objective & task that users give to the agent")
        url: str = Field(description="The url of the website to be scraped")


    class ScrapeWebsiteTool(BaseTool):
        name = "scrape_website"
        description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
        args_schema: Type[BaseModel] = ScrapeWebsiteInput

        def _run(self, objective: str, url: str):
            return scrape_website(objective, url)

        def _arun(self, url: str):
            raise NotImplementedError("error here")


    # 3. Create langchain agent with the tools above
    tools = [
        Tool(
            name="Search",
            func=search,
            description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
        ),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
                you do not make things up, you will try as hard as possible to gather facts & data to back up the research
                
                Please make sure you complete the objective above with the following rules:
                1/ You should do enough research to gather as much information as possible about the objective
                2/ If there are url of relevant links & articles, you will scrape it to gather more information
                3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 5 iteratins
                4/ You should not make things up, you should only write facts & data that you have gathered
                5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
                6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", openai_api_key=OPENAI_API_KEY)
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    return agent


if not browserless_api_key or not OPENAI_API_KEY or not serper_api_key:
    st.write('INPUT KEYS YOU FOOL')
else:
    query = st.text_input('Type in what industry you are focusing on', placeholder='Create a table of vc funds and investors that focus on robotics and prosthetics with as much data about them as possible. YOU MUST INCLUDE THEIR EMAILS')

    if query:
        agent = init()
        with st.spinner('loading, pls wait...'):
            content = agent({"input": query})
            actual_content = content['output']
            st.write(actual_content)