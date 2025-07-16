import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY', './data_storage/')
CONDA_PATH = os.getenv('CONDA_PATH', '/opt/anaconda3/')
CONDA_ENV = os.getenv('CONDA_ENV', 'base')
CHROMEDRIVER_PATH = os.getenv('CHROMEDRIVER_PATH', '/Users/nisargmehta/Downloads/chromedriver-mac-x64/chromedriver')