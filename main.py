import requests
import pandas as pd
import yfinance as yf
import nltk
nitk.download('punkt')

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.llms.openai import OpenAI
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.agent import ReActAgent
import matplotlib.pyplot as plt
import os
import tiktoken
import pandas as pd
from datetime import datetime, timedelta
import re
import streamlit as st
from pinecone import Pinecone,ServerlessSpec
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext
from llama_index.core.schema import TransformComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext,Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import openai
from bs4 import BeautifulSoup

def get_stock_ticker(company_name):
    """
    Fetches the stock  symbol for a given company using OpenAI's GPT-4 API.

    Args:
        company_name (str): The name of the company for which the stock is needed.

    Returns:
        str: The stock  symbol for the specified company.

    
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": f"What is the stock ticker for {company_name}?"}
        ]
    )
    
    # Extract the relevant response
    ticker_info = response.choices[0].message.content
    return ticker_info



def get_basic_info(ticker):
    """Get basic company information."""
    try:
        ticker = yf.Ticker(ticker)
        info = ticker.info
        return {
            'Company Name': info.get('longName'),
            'Industry': info.get('industry'),
            'Sector': info.get('sector'),
            'Country': info.get('country'),
            'Website': info.get('website'),
            'Business Summary': info.get('longBusinessSummary')
        }
    except Exception as e:
        return f"Error fetching basic info: {str(e)}"

def get_financial_metrics(ticker):
    """Get key financial metrics."""
    try:
        ticker = yf.Ticker(ticker)
        info = ticker.info
        return {
            'Market Cap': info.get('marketCap'),
            'Forward P/E': info.get('forwardPE'),
            'Trailing P/E': info.get('trailingPE'),
            'Price/Book': info.get('priceToBook'),
            'Enterprise Value': info.get('enterpriseValue'),
            'Enterprise To Revenue': info.get('enterpriseToRevenue'),
            'Enterprise To EBITDA': info.get('enterpriseToEbitda'),
            'Revenue': info.get('totalRevenue'),
            'Gross Profits': info.get('grossProfits'),
            'Profit Margins': info.get('profitMargins'),
            'Operating Margins': info.get('operatingMargins'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'Revenue Per Share': info.get('revenuePerShare'),
            'EBITDA': info.get('ebitda'),
            'Debt To Equity': info.get('debtToEquity'),
            'Current Ratio': info.get('currentRatio'),
            'Book Value': info.get('bookValue'),
            'Free Cash Flow': info.get('freeCashflow')
        }
    except Exception as e:
        return f"Error fetching financial metrics: {str(e)}"
def get_trading_info(ticker):
    """Get current trading information."""
    try:
        ticker = yf.Ticker(ticker)
        info = ticker.info
        return {
            'Current Price': info.get('currentPrice'),
            'Target High Price': info.get('targetHighPrice'),
            'Target Low Price': info.get('targetLowPrice'),
            'Target Mean Price': info.get('targetMeanPrice'),
            'Previous Close': info.get('previousClose'),
            'Open': info.get('open'),
            'Day Low': info.get('dayLow'),
            'Day High': info.get('dayHigh'),
            '52 Week Low': info.get('fiftyTwoWeekLow'),
            '52 Week High': info.get('fiftyTwoWeekHigh'),
            'Volume': info.get('volume'),
            'Avg Volume': info.get('averageVolume'),
            'Market Cap': info.get('marketCap'),
            'Beta': info.get('beta'),
            'PE Ratio': info.get('trailingPE'),
            'EPS': info.get('trailingEps'),
        }
    except Exception as e:
        return f"Error fetching trading info: {str(e)}"
    
def get_financial_statements(ticker):
    """Get financial statements for a stock."""
    try:
        ticker = yf.Ticker(ticker)
        return {
            'Balance Sheet': ticker.balance_sheet,
            'Income Statement': ticker.income_stmt,
            'Cash Flow': ticker.cashflow
        }
    except Exception as e:
        return f"Error fetching financial statements: {str(e)}"

def get_summarized_market_news(query="stock market"):
    """
    Retrieves the latest market news articles, extracts the full text, and summarizes each article.

    Parameters:
        query (str): The query term for fetching news articles.
        api_key_news (str): Your News API key.
        api_key_openai (str): Your OpenAI API key.

    Returns:
        str: A formatted string containing the summaries of the latest news articles for users.
    """
    api_key = "1fc561f0a48d4ae19bdd2e2473b74ed8"
    # Retrieve articles from News API
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data["status"] != "ok":
        return "Error retrieving data."

    # Prepare summaries
    summarized_news = []
    for article in data["articles"][:5]:  
        title = article["title"]
        article_url = article["url"]
        
        # Fetch full content by scraping the article URL
        article_response = requests.get(article_url)
        soup = BeautifulSoup(article_response.text, "html.parser")
        
        # Extract the main content - this may vary by website structure
        paragraphs = soup.find_all("p")
        full_content = " ".join(p.get_text() for p in paragraphs)
        
        # Summarize with OpenAI
        prompt = f"Summarize the following article in 150 words, highlighting the main points:\n{full_content}"
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
          
        )
        
        # Extract summary
        summary =response.choices[0].message.content
        summarized_news.append(f"Title: {title}\nSummary: {summary}\nRead more: {article_url}\n")

    return "\n\n".join(summarized_news)


def get_stock_market_from_last_30_days(symbol):
    """
    Downloads the last 30 days of stock market data for a given symbol using yfinance.
    
    Parameters:
        symbol (str): The stock symbol to query.

    Returns:
        pd.DataFrame: A DataFrame containing the stock market data (Open, High, Low, 
                      Close, Volume) or an error message if the data could not be retrieved.
    """
    stock_data = yf.download(symbol, period='1mo', interval='1d')
    if not stock_data.empty:
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return stock_data
    else:
        return "Error retrieving data"

def calculate_SMA(symbol, window):
    """
    Calculates the Simple Moving Average (SMA) for a given stock symbol and window size.
    
    Parameters:
        symbol (str): The stock symbol to calculate SMA for.
        window (int): The number of periods to calculate the SMA.

    Returns:
        str: The latest SMA value or a message indicating insufficient data.
    """
    data = yf.Ticker(f"{symbol}").history(period='1y').Close
    
    if len(data) < window:
        return "Not enough data to calculate SMA"
    
    sma_value = data.rolling(window=window).mean().iloc[-1]
    return str(sma_value)

def calculate_EMA(symbol, window):
    """Calculate Exponential Moving Average for a given stock symbol and window size.
    Parameters:-
       symbol (str): The stock symbol to calculate EMA for.
        window (int): The number of periods to calculate the EMA.
    Returns string  latest exponential moving average
    """
    data = yf.Ticker(f"{symbol}").history(period='1mo').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def get_stock_info(ticker_symbol):
    """
    Retrieve comprehensive stock financial metrics  for the given ticker symbol
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol 
    
    Returns:
    dict: Dictionary containing various stock metrics and information
    """
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker_symbol)
        
        # Get basic info
        info = stock.info
        
        # Get historical data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        hist = stock.history(start=start_date, end=end_date)
        
        # Compile relevant information
        stock_data = {
            'Company_Name': info.get('longName', 'N/A'),
            'Current_Price': info.get('currentPrice', 'N/A'),
            'Previous_Close': info.get('previousClose', 'N/A'),
            'Open': info.get('open', 'N/A'),
            'Day_High': info.get('dayHigh', 'N/A'),
            'Day_Low': info.get('dayLow', 'N/A'),
            'Volume': info.get('volume', 'N/A'),
            'Market_Cap': info.get('marketCap', 'N/A'),
            '52_Week_High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_Week_Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'PE_Ratio': info.get('trailingPE', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Dividend_Yield': info.get('dividendYield', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Historical_Data': hist.to_dict()
        }
        
        return stock_data
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

def get_income_statement(ticker_symbol, quarterly=False):
    """
    Get detailed income statement with calculated metrics
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    quarterly (bool): If True, get quarterly statements instead of annual
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Get income statement
        income_stmt = ticker.income_stmt if not quarterly else ticker.quarterly_income_stmt
        
        # Calculate additional metrics
        if isinstance(income_stmt, pd.DataFrame):
            # Profitability Metrics
            income_stmt.loc['Gross Margin %'] = (income_stmt.loc['Gross Profit'] / income_stmt.loc['Total Revenue']) * 100
            income_stmt.loc['Operating Margin %'] = (income_stmt.loc['Operating Income'] / income_stmt.loc['Total Revenue']) * 100
            income_stmt.loc['Net Profit Margin %'] = (income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue']) * 100
            
            # Growth Metrics (Year over Year)
            for item in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']:
                if item in income_stmt.index:
                    income_stmt.loc[f'{item} YoY Growth %'] = income_stmt.loc[item].pct_change(-1) * 100
            
            # Expense Ratios
            if 'Total Revenue' in income_stmt.index and 'Operating Expenses' in income_stmt.index:
                income_stmt.loc['Operating Expense Ratio %'] = (income_stmt.loc['Operating Expenses'] / income_stmt.loc['Total Revenue']) * 100
            
        return income_stmt
    except Exception as e:
        return f"Error fetching income statement: {str(e)}"
def get_balance_sheet(ticker_symbol, quarterly=False):
    """
    Get detailed balance sheet with calculated metrics
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    quarterly (bool): If True, get quarterly statements instead of annual
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Get balance sheet
        balance_sheet = ticker.balance_sheet if not quarterly else ticker.quarterly_balance_sheet
        
        if isinstance(balance_sheet, pd.DataFrame):
            # Liquidity Ratios
            balance_sheet.loc['Current Ratio'] = balance_sheet.loc['Total Current Assets'] / balance_sheet.loc['Total Current Liabilities']
            
            if 'Inventory' in balance_sheet.index:
                balance_sheet.loc['Quick Ratio'] = (balance_sheet.loc['Total Current Assets'] - balance_sheet.loc['Inventory']) / balance_sheet.loc['Total Current Liabilities']
            
            # Solvency Ratios
            if 'Total Liabilities Net Minority Interest' in balance_sheet.index:
                balance_sheet.loc['Debt to Equity Ratio'] = balance_sheet.loc['Total Liabilities Net Minority Interest'] / balance_sheet.loc['Total Stockholders Equity']
            
            # Asset Utilization
            if 'Total Assets' in balance_sheet.index:
                balance_sheet.loc['Asset to Equity Ratio'] = balance_sheet.loc['Total Assets'] / balance_sheet.loc['Total Stockholders Equity']
            
            # Working Capital
            balance_sheet.loc['Working Capital'] = balance_sheet.loc['Total Current Assets'] - balance_sheet.loc['Total Current Liabilities']
            
        return balance_sheet
    except Exception as e:
        return f"Error fetching balance sheet: {str(e)}"

os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI"]
pc = Pinecone(api_key="264040b3-b298-4918-9d56-b31134d5ba48")
index = pc.Index("capitalbot")
st.title("capitalbot")
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


get_income_statement_tool = FunctionTool.from_defaults(fn=get_income_statement)
get_balance_sheet_tool = FunctionTool.from_defaults(fn= get_balance_sheet)
get_basic_info_tool = FunctionTool.from_defaults(fn=get_basic_info)
get_trading_info_tool = FunctionTool.from_defaults(fn=get_trading_info)
get_financial_metrics_tool = FunctionTool.from_defaults(fn=get_financial_metrics)
calculate_EMA_tools = FunctionTool.from_defaults(fn=calculate_EMA)
calculate_sma_tools = FunctionTool.from_defaults(fn=calculate_SMA)
get_stock_market_from_last_30_days_tools = FunctionTool.from_defaults(fn=get_stock_market_from_last_30_days)
get_latest_market_data_tool = FunctionTool.from_defaults(fn=get_summarized_market_news)
get_stock_info_tool = FunctionTool.from_defaults(fn=get_stock_info)
stock_symbol_tools = FunctionTool.from_defaults(fn=get_stock_ticker)
system_prompt = {
    "role": "system",
    "content": (
        "You are a financial assistant capable of providing stock market insights based on  financial data and news. "
        "Analyze the financial data and company information and  balance sheet  , and present the findings. "
        "Provide information that can help the user make informed decisions for buying stocks"
    )
}
llm = OpenAI(model="gpt-4o",temperature=0.2)
agent = ReActAgent.from_tools(
    tools=[
        get_balance_sheet_tool,
        get_income_statement_tool,
        stock_symbol_tools,
        get_basic_info_tool,
        get_trading_info_tool,
        get_financial_metrics_tool,
        get_stock_info_tool,
        get_stock_market_from_last_30_days_tools,
        get_latest_market_data_tool,
    ],
    llm=llm,
    verbose=True,
    messages=[system_prompt] 
)

if prompt := st.chat_input("Ask about cryptocoins?"):
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = agent.chat(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
