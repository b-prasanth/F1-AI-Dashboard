import mysql.connector
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, FunctionMessage
from langchain.tools import Tool
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import json
from langchain_core.messages import HumanMessage

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DatabaseAgent:
    def __init__(self):
        """Initialize the F1 Database Agent with LangChain SQL capabilities"""
        self.db = None
        self.agent_executor = None
        history = ChatMessageHistory()
        self.memory = ConversationBufferWindowMemory(
            chat_memory=history,
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        self._setup_database_connection()
        self._setup_llms()
        self._setup_database_agent()
    
    def _setup_database_connection(self):
        """Set up database connection using LangChain SQLDatabase"""
        try:
            # Create connection string
            db_config = {
                'host': os.getenv('DB_HOST'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'database': os.getenv('F1_DATABASE')
            }
            
            connection_string = (
                f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}/{db_config['database']}"
            )
            
            self.db = SQLDatabase.from_uri(
                connection_string,
                include_tables=[
                    'race_results', 
                    'qualifying_results', 
                    'drivers_championship',
                    'constructors_championship',
                    'f1_calendar',
                    'sprint_race_results',
                    'sprint_qualifying_results'
                ],
                sample_rows_in_table_info=3
            )
            
            logger.info("Database connection established successfully")
            logger.info(f"Available tables: {self.db.get_usable_table_names()}")
            
        except Exception as e:
            logger.error(f"Error setting up database connection: {str(e)}")
            raise
    
    def _setup_llms(self):
        """Set up LLMs with Google Gemini as primary option"""
        try:
            if not os.getenv("LLM_API_KEY"):
                raise ValueError("No valid API key found for Google Gemini LLM")
            
            # Database LLM
            self.database_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                temperature=0,
                google_api_key=os.getenv("LLM_API_KEY")
            )
            logger.info("Using Google Gemini 2.0 Flash as database LLM")

            # User LLM
            self.user_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                temperature=0.7,  # Higher temperature for more creative responses
                google_api_key=os.getenv("LLM_API_KEY")
            )
            logger.info("Using Google Gemini 2.0 Flash as user LLM")
                
        except Exception as e:
            logger.error(f"Error setting up LLMs: {str(e)}")
            raise
    
    def _setup_database_agent(self):
        """Set up the SQL agent with custom tools and prompts"""
        try:
            # Create custom tools
            tools = self._create_custom_tools()
            
            # Define system prompt for tool-calling agent
            database_system_prompt = """
            You are a Formula 1 expert database analyst. 
            Your role is to help users get accurate information about Formula 1 from the database and respond in a user friendly way.

            Below is the Information regarding the tables you have access to.

                1. `constructors_championship`:
                - id (int, PK, auto_increment)
                - year (int)
                - position (int)
                - constructor (varchar(100))
                - constructor_id (varchar(50))
                - nationality (varchar(50), nullable)
                - points (decimal(8,2), nullable)
                - wins (int, nullable, default 0)
                Description: Stores the championship standings for constructors (teams) each year, including their position, points, and wins.

                2. `drivers_championship`:
                - id (int, PK, auto_increment)
                - year (int)
                - position (int)
                - driver (varchar(100))
                - driver_id (varchar(50))
                - constructors (text, nullable)
                - points (decimal(8,2), nullable)
                - wins (int, nullable, default 0)
                Description: Stores the championship standings for drivers each year, including their position, points, wins, and associated constructors.

                3. `f1_calendar`:
                - year (int, nullable)
                - round (int, nullable)
                - GrandPrix (varchar(255), nullable)
                - circuit (varchar(255), nullable)
                - date (date, nullable)
                - PoleSitter (varchar(255), nullable)
                - Winner (varchar(255), nullable)
                - status (varchar(255), nullable)
                Description: Contains the schedule of Grand Prix races, including the year, round, circuit, date, pole sitter, winner, and race status (e.g., "Completed").

                4. `qualifying_results`:
                - id (int, PK, auto_increment)
                - year (int)
                - round (int)
                - grand_prix (varchar(100))
                - circuit (varchar(100))
                - date (date, nullable)
                - driver (varchar(100))
                - driver_id (varchar(50))
                - constructor (varchar(100))
                - position (int, nullable)
                - q1_time (time(3), nullable)
                - q2_time (time(3), nullable)
                - q3_time (time(3), nullable)
                Description: Records qualifying session results, including driver positions and times for Q1, Q2, and Q3.

                5. `race_results`:
                - id (int, PK, auto_increment)
                - year (int)
                - round (int)
                - grand_prix (varchar(100))
                - circuit (varchar(100))
                - date (date, nullable)
                - driver (varchar(100))
                - driver_id (varchar(50))
                - constructor (varchar(100))
                - grid_position (int, nullable)
                - finish_position (int, nullable)
                - status (varchar(50), nullable)
                - points (decimal(8,2), nullable)
                - fastest_lap_rank (int, nullable)
                - fastest_lap_time (time(3), nullable)
                - fastest_lap_speed_kph (decimal(6,3), nullable)
                Description: Stores detailed race results, including grid and finish positions, points, and fastest lap details.

                6. `sprint_qualifying_results`:
                - id (int, PK, auto_increment)
                - year (int)
                - round (int)
                - grand_prix (varchar(100))
                - circuit (varchar(100))
                - date (date, nullable)
                - driver (varchar(100))
                - driver_id (varchar(50))
                - constructor (varchar(100))
                - position (int, nullable)
                - sprint_time (time(3), nullable)
                Description: Contains results for sprint qualifying sessions, including driver positions and times.

            IMPORTANT GUIDELINES:
            1. ONLY answer questions related to Formula 1 racing
            2. If asked about non-F1 topics, politely redirect to F1-related questions
            3. Always use the database to get current and accurate information
            4. When querying, consider these table relationships mentioned above.
            5. For recent/current season queries, focus on the latest year in the database. Check the current date and then respond accordingly.
            6. Always provide context about the year/season when giving results
            7. If data spans multiple years, summarize appropriately
            8. Handle driver name variations (nicknames, shortened names)
            9. Be specific about race names and circuits when relevant
            10. If the user says Hi or greets you in anyway, greet the user back. But answer questions only related to F1.
            11. Remember the last 10 messages with the user and respond accordingly. If the previous question was related to F1 use that and answer any followup questions.

            Remember: Always verify your SQL queries are correct before executing them. Use proper JOIN statements when needed to get complete information.
            
            If Conversation is unrelated to F1 then respond to user with "I'm a Formula 1 expert assistant. I can help you with questions about F1 races, drivers, "
                    "championships, qualifying results, and F1 history. Please ask me something about Formula 1!"
            """
            
            # Create prompt template for tool-calling agent
            database_prompt = ChatPromptTemplate.from_messages([
                ("system", database_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create tool-calling agent (compatible with Gemini)
            agent = create_tool_calling_agent(
                llm=self.database_llm,
                tools=tools,
                prompt=database_prompt
            )
            logger.info("Using tool-calling agent for Gemini")
            
            self.database_agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                return_intermediate_steps=True
            )
            
            logger.info("Agent setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise
    
    def _create_custom_tools(self):
        """Create custom tools for the agent"""
        tools = []
        
        # SQL Database Query Tool
        sql_query_tool = QuerySQLDatabaseTool(db=self.db)
        tools.append(sql_query_tool)
        
        # Database Schema Tool
        def get_schema_info(table_name: str) -> str:
            """Get schema information for a specific table"""
            try:
                if table_name in self.db.get_usable_table_names():
                    return self.db.get_table_info([table_name])
                else:
                    return f"Table '{table_name}' not found. Available tables: {', '.join(self.db.get_usable_table_names())}"
            except Exception as e:
                return f"Error getting schema info: {str(e)}"
        
        schema_tool = Tool(
            name="sql_db_schema",
            description="Get schema information for F1 database tables. Input should be a table name.",
            func=get_schema_info
        )
        tools.append(schema_tool)
        
        # List Tables Tool
        def list_tables() -> str:
            """List all available tables in the database"""
            try:
                tables = self.db.get_usable_table_names()
                return f"Available F1 database tables: {', '.join(tables)}"
            except Exception as e:
                return f"Error listing tables: {str(e)}"
        
        list_tables_tool = Tool(
            name="sql_db_list_tables",
            description="List all available tables in the F1 database",
            func=lambda x: list_tables()
        )
        tools.append(list_tables_tool)
        
        # Recent Season Info Tool
        def get_recent_season_info() -> str:
            """Get information about the most recent season in the database"""
            try:
                query = "SELECT MAX(year) as latest_year FROM race_results"
                result = self.db.run(query)
                return f"Most recent season data available: {result}"
            except Exception as e:
                return f"Error getting recent season info: {str(e)}"
        
        recent_season_tool = Tool(
            name="get_recent_season_info",
            description="Get information about the most recent F1 season available in the database",
            func=lambda x: get_recent_season_info()
        )
        tools.append(recent_season_tool)
        
        return tools
    
    def query(self, user_question: str) -> tuple[str, dict]:
        try:
            
            logger.info(f"Processing F1 query: {user_question}")

            # Use the agent executor with proper input format
            db_response = self.database_agent_executor.invoke({
                "input": user_question,
                "chat_history": self.memory.chat_memory.messages
            })
            
            agent_output = db_response.get("output", "I couldn't process your question. Please try rephrasing it.")

            # Create metadata
            metadata = {
                "type": "database_query",
                "timestamp": datetime.now().isoformat(),
                "intermediate_steps": len(db_response.get("intermediate_steps", [])),
                "memory_length": len(self.memory.chat_memory.messages)
            }
            
            logger.info(f"Query processed successfully")
            return agent_output, metadata

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_response = (
                "I encountered an error while processing your question. "
                "Please try rephrasing your question or ask about a different F1 topic."
            )
            error_metadata = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return error_response, error_metadata
 
        
    def get_database_stats(self) -> dict:
        """Get statistics about the database"""
        try:
            stats = {}
            
            for table in self.db.get_usable_table_names():
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {table}"
                    result = self.db.run(count_query)
                    stats[table] = {"record_count": result}
                    
                    # Get year range for tables with year column
                    if table in ['race_results', 'qualifying_results', 'drivers_championship', 
                               'constructors_championship', 'f1_calendar']:
                        year_query = f"SELECT MIN(year) as min_year, MAX(year) as max_year FROM {table}"
                        year_result = self.db.run(year_query)
                        stats[table]["year_range"] = year_result
                        
                except Exception as e:
                    stats[table] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")

# Global instance
f1_agent = None

def initialize_f1_agent():
    """Initialize the F1 database agent"""
    global f1_agent
    try:
        logger.info("Initializing F1 Database Agent...")
        f1_agent = F1DatabaseAgent()
        logger.info("F1 Database Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize F1 Database Agent: {str(e)}")
        return False

def query_f1_database(user_question: str) -> tuple[str, dict]:
    """
    Main function to query the F1 database using LangChain
    
    Args:
        user_question (str): The user's question about F1
        
    Returns:
        tuple: (response_text, metadata_dict)
    """
    global f1_agent
    
    if f1_agent is None:
        if not initialize_f1_agent():
            return (
                "Database agent is not available. Please try again later.",
                {"type": "error", "timestamp": datetime.now().isoformat()}
            )
    
    return f1_agent.query(user_question)

def get_database_statistics() -> dict:
    """Get database statistics"""
    global f1_agent
    
    if f1_agent is None:
        if not initialize_f1_agent():
            return {"error": "Database agent not available"}
    
    return f1_agent.get_database_stats()

def clear_conversation_memory():
    """Clear the conversation memory"""
    global f1_agent
    
    if f1_agent is not None:
        f1_agent.clear_memory()

# Example usage and testing
if __name__ == "__main__":
    # Initialize the agent
    if initialize_f1_agent():
        # Test queries
        test_queries = [
            "Who won the 2025 Monaco Grand Prix?",
            "What are Hamilton's race wins in 2023?",
            "Show me the top 5 drivers from the latest season",
            "Which constructor has the most wins?",
            "What's the weather like today?"  # Non-F1 question
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            response, metadata = query_f1_database(query)
            print(f"Response: {response}")
            print(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    else:
        print("Failed to initialize F1 Database Agent")