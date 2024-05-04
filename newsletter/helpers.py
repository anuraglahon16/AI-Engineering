import os
from dotenv import find_dotenv, load_dotenv
import openai
import json
import requests
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.utilities import GoogleSerperAPIWrapper

openai.api_key = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERPER_API_KEY")

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()

def search_serp(query, start_date, end_date):
    search = GoogleSerperAPIWrapper(k=5, type="search")
    date_range = f"date:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    response_json = search.results(f"{query} {date_range}")
    print(f"Response=====>, {response_json}")
    return response_json

def pick_best_articles_urls(response_json, query, start_date, end_date):
    response_str = json.dumps(response_json)
    
    llm = ChatOpenAI(temperature=0.7)
    template = """ 
      You are a world-class journalist, researcher, tech, Software Engineer, Developer, and an online course creator.
      You are amazing at finding the most interesting, relevant, and useful articles on certain topics within a specific date range.
      
      QUERY RESPONSE:{response_str}
      
      Above is the list of search results for the query {query} between {start_date} and {end_date}.
      
      Please choose the best 3 articles from the list and return ONLY an array of the URLs.  
      Do not include anything else - return ONLY an array of the URLs. 
      Make sure the articles are within the specified date range.
      If the file or URL is invalid, show www.google.com.
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query", "start_date", "end_date"],
        template=template
    )
    article_chooser_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )
    urls = article_chooser_chain.run(response_str=response_str,
                                     query=query,
                                     start_date=start_date,
                                     end_date=end_date)
    
    url_list = json.loads(urls)
    return url_list

def extract_content_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(data)
    
    if not docs:
        raise ValueError("No documents found in the provided URLs.")
    
    db = FAISS.from_documents(docs, embeddings)
    
    return db

def summarizer(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7)
    template = """
       {docs}
        As a world-class journalist, researcher, article, newsletter, and blog writer, 
        you will summarize the text above in order to create a 
        newsletter around {query}.
        This newsletter will be sent as an email. The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest, and understand
        6/ The content needs to give the audience actionable advice & insights including resources and links if necessary
        
        SUMMARY:
    """
    prompt_template = PromptTemplate(input_variables=["docs", "query"],
                                     template=template)
    
    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    
    response = summarizer_chain.run(docs=docs_page_content, query=query)
    
    return response.replace("\n", "")

def generate_newsletter(summaries, query):
    summaries_str = str(summaries)
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                     temperature=.7)
    template = """
    {summaries_str}
        As a world-class journalist, researcher, article, newsletter, and blog writer, 
        you'll use the text above as the context about {query}
        to write an excellent newsletter to be sent to subscribers about {query}.
        
        This newsletter will be sent as an email. The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Make sure to write it informally - no "Dear" or any other formalities. Start the newsletter with
        `Hi All!
          Here is your weekly dose of the Tech Newsletter, a list of what I find interesting
          and worth exploring.`
          
        Make sure to also write a backstory about the topic - make it personal, engaging, and lighthearted before
        going into the meat of the newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest, and understand
        6/ The content needs to give the audience actionable advice & insights including resources and links if necessary.
        
        If there are books or products involved, make sure to add Amazon links to the products or just a link placeholder.
        
        As a sign-off, write a clever quote related to learning, general wisdom, living a good life. Be creative with this one - and then,
        
        -Anurag
        
        NEWSLETTER-->:
    """
    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], 
                                     template=template)
    news_letter_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True)
    news_letter = news_letter_chain.predict(
        summaries_str=summaries_str,
        query=query)
    
    return news_letter
# ... (Previous code remains the same)

def pick_best_articles_urls(response_json, query, start_date, end_date, num_articles):
    response_str = json.dumps(response_json)
    
    llm = ChatOpenAI(temperature=0.7)
    template = """ 
      You are a world-class journalist, researcher, tech, Software Engineer, Developer, and an online course creator.
      You are amazing at finding the most interesting, relevant, and useful articles on certain topics within a specific date range.
      
      QUERY RESPONSE:{response_str}
      
      Above is the list of search results for the query {query} between {start_date} and {end_date}.
      
      Please choose the best {num_articles} articles from the list and return ONLY an array of the URLs.  
      Do not include anything else - return ONLY an array of the URLs. 
      Make sure the articles are within the specified date range.
      If the file or URL is invalid, show www.google.com.
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query", "start_date", "end_date", "num_articles"],
        template=template
    )
    article_chooser_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )
    urls = article_chooser_chain.run(response_str=response_str,
                                     query=query,
                                     start_date=start_date,
                                     end_date=end_date,
                                     num_articles=num_articles)
    
    url_list = json.loads(urls)
    return url_list

# ...

def generate_newsletter(summaries, query, custom_sections):
    summaries_str = str(summaries)
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                     temperature=.7)
    template = """
    {summaries_str}
        As a world-class journalist, researcher, article, newsletter, and blog writer, 
        you'll use the text above as the context about {query}
        to write an excellent newsletter to be sent to subscribers about {query}.
        
        This newsletter will be sent as an email. The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Make sure to write it informally - no "Dear" or any other formalities. Start the newsletter with
        `Hi All!
          Here is your weekly dose of the Tech Newsletter, a list of what I find interesting
          and worth exploring.`
          
        Make sure to also write a backstory about the topic - make it personal, engaging, and lighthearted before
        going into the meat of the newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest, and understand
        6/ The content needs to give the audience actionable advice & insights including resources and links if necessary.
        
        If there are books or products involved, make sure to add Amazon links to the products or just a link placeholder.
        
        Include the following custom sections in the newsletter, if provided:
        {custom_sections}
        
        As a sign-off, write a clever quote related to learning, general wisdom, living a good life. Be creative with this one - and then,
        Sign with "Paulo 
          - Learner and Teacher"
        
        NEWSLETTER-->:
    """
    prompt_template = PromptTemplate(input_variables=["summaries_str", "query", "custom_sections"], 
                                     template=template)
    news_letter_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True)
    news_letter = news_letter_chain.predict(
        summaries_str=summaries_str,
        query=query,
        custom_sections=custom_sections)
    
    return news_letter