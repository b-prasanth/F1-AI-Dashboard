from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
import logging
from datetime import datetime
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend import rag as langchain_f1_agent  # Import the new LangChain agent

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangChain F1 Agent on startup
logger.info("Initializing F1 Database Agent...")
try:
    if langchain_f1_agent.initialize_f1_agent():
        logger.info("F1 Database Agent initialized successfully")
    else:
        logger.error("Failed to initialize F1 Database Agent")
except Exception as e:
    logger.error(f"Error during F1 Database Agent initialization: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check if the agent is properly initialized
    agent_status = "healthy" if langchain_f1_agent.f1_agent is not None else "agent_unavailable"
    
    return jsonify({
        'status': 'healthy',
        'agent_status': agent_status,
        'timestamp': datetime.now().isoformat(),
        'service': 'F1 LangChain SQL Chatbot'
    })

@app.route('/llm', methods=['POST'])
def llm():
    """Main chatbot endpoint using LangChain SQL agent"""
    try:
        logger.info('Received request')
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request format. Please provide a message field.'
            }), 400
        
        user_message = data['message'].strip()
        logger.info(f'Received message: {user_message}')
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty.'
            }), 400
        # Query the F1 database using LangChain agent
        response_text, metadata = langchain_f1_agent.query_f1_database(user_message)
        
        # Format the response
        response = {
            'response': response_text,
            'query_type': metadata.get('type', 'unknown'),
            'processing_steps': metadata.get('intermediate_steps', 0),
            'memory_context': metadata.get('memory_length', 0),
            'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
            'source': 'langchain_sql_agent'
        }
        
        # Add error details if present
        if 'error' in metadata:
            response['error_details'] = metadata['error']
        
        logger.info(f'Sending response: {response_text[:100]}...')
        return jsonify(response)
    
    except Exception as e:
        logger.error(f'Error processing request: {str(e)}')
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/database_stats', methods=['GET'])
def get_database_stats():
    """Get statistics about the F1 database"""
    try:
        logger.info('Database stats requested')
        
        stats = langchain_f1_agent.get_database_statistics()
        
        return jsonify({
            'database_statistics': stats,
            'timestamp': datetime.now().isoformat(),
            'agent_type': 'langchain_sql'
        })
    
    except Exception as e:
        logger.error(f'Error getting database stats: {str(e)}')
        return jsonify({
            'error': 'Failed to retrieve database statistics',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear the conversation memory of the agent"""
    try:
        logger.info('Memory clear requested')
        
        langchain_f1_agent.clear_conversation_memory()
        
        return jsonify({
            'message': 'Conversation memory cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f'Error clearing memory: {str(e)}')
        return jsonify({
            'error': 'Failed to clear conversation memory',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/agent_info', methods=['GET'])
def get_agent_info():
    """Get information about the current agent setup"""
    try:
        agent = langchain_f1_agent.f1_agent
        
        if agent is None:
            return jsonify({
                'status': 'agent_not_initialized',
                'message': 'Agent is not properly initialized',
                'timestamp': datetime.now().isoformat()
            })
        
        # Get available tables from the database
        try:
            tables = agent.db.get_table_names() if agent.db else []
        except:
            tables = []
        
        info = {
            'status': 'active',
            'agent_type': 'langchain_sql_agent',
            'llm_model': str(agent.llm.model if hasattr(agent.llm, 'model') else 'gemini-2.0-flash-001'),
            'available_tables': tables,
            'memory_type': 'ConversationBufferWindowMemory',
            'max_memory_exchanges': 10,
            'capabilities': [
                'Dynamic SQL query generation',
                'Multi-step reasoning',
                'Context-aware responses',
                'F1-specific knowledge',
                'Conversation memory',
                'Error handling and retry'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f'Error getting agent info: {str(e)}')
        return jsonify({
            'error': 'Failed to retrieve agent information',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test_query', methods=['POST'])
def test_query():
    """Test endpoint with predefined F1 queries"""
    test_queries = {
        'recent_champion': "Who won the most recent drivers' championship?",
        'constructor_standings': "Show me the latest constructor championship standings",
        'race_winner': "Who won the last race in the database?",
        'driver_stats': "What are Lewis Hamilton's career statistics?",
        'qualifying_performance': "Show me the best qualifying performances this season"
    }
    
    try:
        data = request.get_json() or {}
        query_type = data.get('query_type', 'recent_champion')
        
        if query_type not in test_queries:
            return jsonify({
                'error': f'Invalid query type. Available types: {list(test_queries.keys())}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        test_question = test_queries[query_type]
        logger.info(f'Testing with query: {test_question}')
        
        # Execute the test query
        response_text, metadata = langchain_f1_agent.query_f1_database(test_question)
        
        return jsonify({
            'test_query': test_question,
            'query_type': query_type,
            'response': response_text,
            'metadata': metadata,
            'available_test_types': list(test_queries.keys()),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f'Error in test query: {str(e)}')
        return jsonify({
            'error': 'Test query failed',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/reinitialize_agent', methods=['POST'])
def reinitialize_agent():
    """Reinitialize the LangChain agent (useful for debugging)"""
    try:
        logger.info('Agent reinitialization requested')
        
        # Clear existing agent
        langchain_f1_agent.f1_agent = None
        
        # Reinitialize
        success = langchain_f1_agent.initialize_f1_agent()
        
        if success:
            return jsonify({
                'message': 'Agent reinitialized successfully',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Failed to reinitialize agent',
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    except Exception as e:
        logger.error(f'Error reinitializing agent: {str(e)}')
        return jsonify({
            'error': 'Agent reinitialization failed',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'GET /health': 'Health check and agent status',
            'POST /llm': 'Main F1 chatbot endpoint',
            'GET /database_stats': 'Get F1 database statistics',
            'POST /clear_memory': 'Clear conversation memory',
            'GET /agent_info': 'Get agent configuration information',
            'POST /test_query': 'Test with predefined F1 queries',
            'POST /reinitialize_agent': 'Reinitialize the agent'
        },
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'The server encountered an unexpected error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)
