from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel, Field
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import requests

search_tool = TavilySearchResults()


def read_pdf(file_path) -> str:
    output = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                output.append(page.extract_text())

        return output
        
    except Exception as e:
        print(f"Error reading PDF file: {e}")


# Scrape menu data
class ScrapeInput(BaseModel):
    url: str = Field(description="the URL of the menu page")
    url: str = Field(description="the URL of the menu page")

@tool("scrape_pdf", args_schema=ScrapeInput, return_direct=True)
def scrape_pdf(url: str):
    """Scrape a webpage that may include links to a restaurants current menu and return the links"""
    pdf_links = []
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        pdf_links = [link.get('href') for link in links if link.get('href').endswith('.pdf')]
    except Exception as e:
        print(f"failed to scrape {url} ERROR: {e}")
    return pdf_links


@tool("scrape_text", args_schema=ScrapeInput, return_direct=True)
def scrape_text(url: str):
    """Scrape the text directly from a website"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text
        text = soup.get_text(separator=' ')
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        print(f"failed to scrape {url} ERROR: {e}")
        return None
    return text


class DownloadInput(BaseModel):
    download_url: str = Field(description="the URL of the menu pdf")
    filename: str = Field(description="the name of the restaurants")

@tool("download-pdf", args_schema=DownloadInput, return_direct=True)
def download_pdf(download_url: str, filename: str):
    """Download a pdf file from a given url and filename"""
    response = urlopen(download_url)
    file = open("static/"+filename+".pdf", 'wb')
    file.write(response.read())
    file.close()


# Tokenize the pdf 
class upsertInput(BaseModel):
    file_path: str = Field(description="The filepath to the pdf in the format of pdf/filename.pdf")
    restaurant_name: str = Field(description="The name of the restaurant")
    location: str = Field(description="The location of the restaurant")


@tool("upsert-pdf", args_schema=upsertInput, return_direct=True)
def upsert_pdf(file_path, restaurant_name, location):
    """Upsert the pdf_file with metadata for vector search"""
    # Create the text splitter
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    
    # Split the input text
    file = read_pdf(file_path)
    docs = text_splitter.create_documents(file)
    
    # Add metadata 
    metadata = {
        "name": restaurant_name,
        "location": location,
    }
    
    for doc in docs:
        doc.metadata = metadata
        
    # Point to our Pinecone index
    pc = Pinecone()
    index = pc.Index("paprika-ragu")
    
    # Upsert the data
    PineconeVectorStore.from_documents(docs, embeddings, index_name="paprika-ragu")

tools = [search_tool, scrape_pdf, scrape_text, download_pdf, upsert_pdf]
