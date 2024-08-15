from langchain.tools import tool, Tool
from langchain_community.tools import TavilySearchResults
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel, Field
import pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import requests

"""Perform a web search using Tavily API and return the results."""
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
        return None

class ScrapeInput(BaseModel):
    url: str = Field(description="The URL of the menu page")

@tool("scrape_pdf", args_schema=ScrapeInput, return_direct=True)
def scrape_pdf(url: str):
    """Scrape a webpage that may include links to a restaurant's current menu and return the links."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href').endswith('.pdf')]
        return pdf_links
    except Exception as e:
        return f"Failed to scrape {url}. ERROR: {e}"

@tool("scrape_text", args_schema=ScrapeInput, return_direct=True)
def scrape_text(url: str):
    """Scrape the text directly from a website."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = '\n'.join(chunk for chunk in (phrase.strip() for line in (line.strip() for line in soup.get_text(separator=' ').splitlines()) for phrase in line.split("  ")) if chunk)
        return text
    except Exception as e:
        return f"Failed to scrape {url}. ERROR: {e}"

class DownloadInput(BaseModel):
    download_url: str = Field(description="The URL of the menu pdf")
    filename: str = Field(description="The name of the restaurant")

@tool("download_pdf", args_schema=DownloadInput, return_direct=True)
def download_pdf(download_url: str, filename: str):
    """Download a pdf file from a given url with the specified filename."""
    try:
        response = requests.get(download_url)
        with open(f"static/{filename}.pdf", 'wb') as file:
            file.write(response.content)
        return f"Downloaded {filename}.pdf successfully."
    except Exception as e:
        return f"Failed to download PDF from {download_url}. ERROR: {e}"

class UpsertInput(BaseModel):
    file_path: str = Field(description="The filepath to the pdf in the format of pdf/filename.pdf")
    restaurant_name: str = Field(description="The name of the restaurant")
    location: str = Field(description="The location of the restaurant")

@tool("upsert_pdf", args_schema=UpsertInput, return_direct=True)
def upsert_pdf(file_path: str, restaurant_name: str, location: str):
    """Upsert the PDF file with metadata for vector search."""
    try:
        embeddings = OpenAIEmbeddings()
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        
        file_content = read_pdf(file_path)
        if not file_content:
            return "Failed to read PDF content."

        docs = text_splitter.create_documents(file_content)
        metadata = {"name": restaurant_name, "location": location}
        
        for doc in docs:
            doc.metadata = metadata
        
        pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
        PineconeVectorStore.from_documents(docs, embeddings, index_name="paprika-ragu")
        return "Successfully upserted PDF content to Pinecone."
    except Exception as e:
        return f"Failed to upsert PDF. ERROR: {e}"

tools = [search_tool, scrape_pdf, scrape_text, download_pdf, upsert_pdf]