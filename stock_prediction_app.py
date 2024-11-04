from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools import Tool
from dataclasses import dataclass
from tools.stock_price_tool import StockPredictor
# from tools.fetch_stock_info import get_stock_ticker

import requests
import streamlit as st
import plotly.graph_objects as go
# from stock_price_tool import StockPredictor # Import the StockPredictor class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class StockPrediction:
    date: datetime
    price: float
    change: float

class ChatBot:
    def __init__(self, model: str, api_key: str):
        self.client = openai.Client(api_key=api_key)
        self.model = model
        
    def get_response(self, question: str, predictions: List[StockPrediction], metrics: Dict, symbol: str) -> str:
        """Generates response to user questions about stock predictions."""
        try:
            # Create context from predictions and metrics
            context = {
                "predictions": [
                    {
                        "date": p.date.strftime("%Y-%m-%d"),
                        "price": f"${p.price:.2f}",
                        "change": f"{p.change:+.2f}%"
                    } for p in predictions
                ],
                "model_metrics": metrics
            }
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst providing clear, accurate information."},
                    {"role": "user", "content": f"Context: Analysis for {symbol} stock\n{context}\nQuestion: {question}"}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"I apologize, but I encountered an error processing your question. Please try again."



class PricePredictor:
    def __init__(self):
        self.predictor = None
        
    def setup_predictor(self, symbol: str):
        """Initialize or update the StockPredictor with new symbol"""
        self.predictor = StockPredictor(symbol)

    def get_financial_statements(self, symbol:str):
        return self.predictor.get_financial_statements(symbol)
        
    def predict_prices(self, days_ahead: int) -> List[StockPrediction]:
        """Generate predictions for specified number of days"""
        try:
            if not self.predictor:
                raise ValueError("Predictor not initialized")
                
            # Train the model
            self.predictor.fit()
            
            # Get predictions and changes
            predictions, changes = self.predictor.predict(days_ahead)
            current_price = self.predictor.get_current_price()
            
            # Create list of StockPrediction objects
            prediction_list = []
            for day, (pred, change) in enumerate(zip(predictions, changes), 1):
                prediction_date = datetime.now() + timedelta(days=day)
                prediction_list.append(StockPrediction(
                    date=prediction_date,
                    price=pred,
                    change=change
                ))
            
            return prediction_list
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Failed to generate predictions: {e}")
            
    def get_model_metrics(self) -> Dict:
        """Get model performance metrics"""
        if self.predictor:
            return self.predictor.get_model_metrics()
        return {}
    
    def get_current_price(self) -> float:
        """Get current stock price"""
        if self.predictor:
            return self.predictor.get_current_price()
        return 0.0


def create_stock_analysis_agent(api_key: str) -> AgentExecutor:
    """Create a LangChain agent to analyze stock predictions."""
    # Initialize the OpenAI model
    llm = ChatOpenAI(api_key=api_key, model_name="gpt-4-turbo-preview", temperature=0.1)

    # Define the tools with proper typing and error handling

    def create_prediction_chart(predictions: List[StockPrediction], symbol: str):
        """Create and display prediction chart."""
        if predictions:
            predictor = PricePredictor()
            fig = go.Figure()
            
            # Price prediction line
            dates = [p.date for p in predictions]
            prices = [p.price for p in predictions]
            changes = [p.change for p in predictions]
            
            # Add current price point
            current_price = predictor.get_current_price()
            dates.insert(0, datetime.now())
            prices.insert(0, current_price)
            
            # Create the line plot
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines+markers',
                name='Stock Price',
                line=dict(color='#00ff00'),
                marker=dict(size=8)
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Predicted Stock Prices for {symbol}",
                xaxis_title="Date",
                yaxis_title="Stock Price (USD)",
                template="plotly_dark",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True) 

    def get_stock_prediction(symbol: str) -> str:
        # , days_ahead: str
        """Get stock price predictions for the specified symbol and time period."""
        try:
            days_ahead = 7 #int(days_ahead)
            predictor = PricePredictor()
            predictor.setup_predictor(symbol)
            predictions = predictor.predict_prices(days_ahead)

            st.session_state.predictions = predictions
            st.session_state.current_symbol = symbol
            create_prediction_chart(predictions,symbol)
            # Format predictions into a readable string
            st.subheader("Detailed Predictions")
            prediction_data = {
                "Date": [p.date.strftime("%Y-%m-%d") for p in predictions],
                "Predicted Price": [f"${p.price:.2f}" for p in predictions],
                "Expected Change": [f"{p.change:+.2f}%" for p in predictions]
            }
            st.table(pd.DataFrame(prediction_data))
        except Exception as e:
            return f"Error generating predictions: {str(e)}"
    
    def get_financial_statements(symbol:str) -> str:
        predictor = PricePredictor()
        return get_financial_statements(symbol)


    def get_stock_ticker(query: str):
        functions = [
            {
                "type": "function",
                "function":{
                "name": "get_company_stock_ticker",
                "description": "Retrieve the NASDAQ stock ticker for a company and the number of days ahead for stock price prediction. The 'days_ahead' parameter should be an integer representing the number of days, defaulting to 1 if not provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_symbol": {
                            "type": "string",
                            "description": "The stock symbol of the company.",
                        },
                        "company_name": {
                            "type": "string",
                            "description": "The name of the company from the query.",
                        },
                        "days_ahead": {
                            "type": "integer",
                            "description": "The number of days ahead for stock price prediction.",
                            "default": 1
                        }
                    },
                    "required": ["company_name", "ticker_symbol","days_ahead"],
                    "additionalProperties": False,
                },
            }
                
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the company name, stock ticker, and prediction days from the user query: {query}"
                }
            ],
            tools=functions
        )
        
        # Assuming the API now returns structured data directly
        
        # stool_call = response.choices[0].message.tool_calls[0]
        response_dict = response.model_dump() 
        print(response_dict["choices"][0]["message"]["content"])
        # arguments = json.loads(response['choices'][0]['message']['content'])
        # print(arguments)
        
        # if arguments:
        #     # Load arguments JSON safely
        #     arguments = json.loads(arguments)
        #     company_name = arguments.get("company_name")
        #     symbol = arguments.get("ticker_symbol")
        #     days_ahead = arguments.get("days_ahead", 1)
        #     return company_name, symbol, days_ahead
        # else:
        #     raise ValueError("Function call did not return arguments as expected.")


    tools = [
        Tool(
            name = "get_financial_statements",
            func = get_financial_statements,
            description = "Get financial statments of the given company",
            return_direct=True
        ),
        Tool(
            name="get_stock_ticker",
            func=get_stock_ticker,
            description="Get current stock information for a given symbol, company name and days ahead.",
            return_direct=True
        ),
        Tool(
            name="get_stock_prediction",
            func=get_stock_prediction,
            description="Get stock price predictions for a given symbol and number of days ahead. Input should be a stock symbol (e.g., 'AAPL') and number of days (e.g., 7).",
            return_direct=True
        )
    ]

    # Create a more specific system prompt
    system_prompt = """You are a financial analysis assistant that helps users understand stock predictions and market data. 
    When users ask about stocks, first get the current stock information using get_stock_ticker, then generate predictions using get_stock_prediction.
    Always explain the predictions in a clear, concise manner and highlight important trends or patterns, if you need any financial statments you can use get_financial_statements .
    If you encounter any errors, inform the user and suggest alternative approaches."""

    # Fix: Use the correct message format
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create the agent with proper error handling
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

class StockAnalysisApp:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.agent = create_stock_analysis_agent(self.api_key)
        self.symbol = "AAPL"
        self.days_ahead = 7
        self.setup_session_state()
        self.chatbot = ChatBot("gpt-4-turbo-preview", self.api_key)
        self.initialize_components()
        self.setup_session_state()

    def initialize_components(self):
        """Initialize main components of the application."""
        self.predictor = PricePredictor()
        self.chatbot = ChatBot("gpt-4-turbo-preview", self.api_key)

    def setup_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = ""
   
    
    def run(self):
        st.set_page_config(layout="wide")
        st.title("Stock Price Prediction & Analysis Assistant")
        st.write("AI-powered predictions using machine learning and real-time market data.")
                # Create main layout
        col1, col2 = st.columns(spec=2, gap="medium")
        with col1:
            # Input fields for prediction settings
            self.symbol = st.text_input("Enter Stock Symbol:", value=self.symbol)
            self.days_ahead = st.text_input("Days to Predict(1 to 30):", value=self.days_ahead)

            if st.button("Generate Predictions"):
                try:
                    # Update: Format the input as a dictionary with required keys
                    query = f"Get predictions for {self.symbol} stock for the next {self.days_ahead} days"
                    print(query)
                    response = self.agent.invoke({
                        "input": query,
                        "agent_scratchpad": [],  # Initialize empty scratchpad,
                        "chat_history":[],
                    })
                    
                    if response and "output" in response:
                        st.success("Predictions generated successfully!")
                        st.write(response["output"])
                    else:
                        st.error("No predictions were generated. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    logger.error(f"Agent error: {str(e)}", exc_info=True)
        with col2:
            self.render_chat_section()
            # user_query = st.text_input("Ask a question about the predictions:")
            # if user_query:
            #     response = self.agent.invoke({
            #         "input": user_query,
            #         "agent_scratchpad": [],  # Initialize empty scratchpad,
            #         "chat_history":[],
            #         })
            #     st.write("Response:", response['output'])

    def render_chat_section(self):
        """Render the chat interface."""
        st.subheader("Ask Questions")
        st.write("Ask questions about the predictions and analysis.")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the predictions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                if st.session_state.predictions:
                    response = self.chatbot.get_response(
                        prompt,
                        st.session_state.predictions,
                        self.predictor.get_model_metrics(),
                        st.session_state.current_symbol
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.markdown("Please generate stock predictions first to enable the chat feature.")

    
if __name__ == "__main__":
    app = StockAnalysisApp()
    app.run()
