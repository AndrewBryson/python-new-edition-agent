# Flask web application for Azure AI Agent chat interface

from flask import Flask, render_template, request, jsonify, session
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai.types.responses.response_input_param import McpApprovalResponse, ResponseInputParam
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import secrets
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure secret key for sessions (set this as an environment variable in production)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

# Configure session cookies
app.config['SESSION_COOKIE_NAME'] = 'ai_agent_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

# Store thread IDs per session
session_clients = defaultdict(lambda: None)

# Azure AI configuration
myEndpoint = os.getenv('myEndpoint')
myAgent = os.getenv('myAgent')

# Configure logging
def setup_logging():
    log_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s (%(funcName)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation (10MB max, keep 5 backup files)
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Configure app logger
    app.logger.setLevel(log_level)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(error_handler)
    app.logger.addHandler(console_handler)
    
    # Reduce noise from Azure SDK
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
    
    app.logger.info('='*50)
    app.logger.info('Application starting...')
    app.logger.info('='*50)

# Initialize logging
setup_logging()

# Initialize Azure AI client
try:
    app.logger.info(f'Initializing Azure AI client with endpoint: {myEndpoint}')
    project_client = AIProjectClient(
        endpoint=myEndpoint,
        credential=DefaultAzureCredential(),
    )
    
    # Get the agent
    app.logger.info(f'Retrieving agent: {myAgent}')
    agent = project_client.agents.get(agent_name=myAgent)

    openai_client = project_client.get_openai_client()

    # Modify the tool approval settings to "never"
    updated_definition = agent.versions.latest.definition
    for tool in updated_definition.tools:
        tool.require_approval = "never"

    # Update the agent with the modified definition
    agent = project_client.agents.update(
        agent_name=myAgent, 
        definition=updated_definition
    )
    
    app.logger.info(f'Successfully initialized agent: {agent.name}')
except Exception as e:
    app.logger.error(f'Failed to initialize Azure AI client: {str(e)}', exc_info=True)
    raise

def get_or_create_session_id():
    """Get existing session ID or create a new one"""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_urlsafe(32)
        session.permanent = True
        app.logger.info(f'Created new session: {session["session_id"][:8]}...')
    return session['session_id']

# def get_or_create_responses_client(session_id):
#     """Get existing thread for session or create a new one"""
#     """
#         response = openai_client.responses.create(
#         input=input_list,
#         previous_response_id=response.id,
#         extra_body={"agent": {"name": agent.name, "type": "agent_reference"}, "thread_id": thread_id},
#     """
#     if session_clients[session_id] is None:
#         response = openai_client.responses.create()
#         session_clients[session_id] = response.id
#         app.logger.info(f'Created new Responses client {response.id} for session {session_id[:8]}...')
#     return session_clients[session_id]

@app.route('/')
def index():
    """Render the main chat interface and establish session"""
    session_id = get_or_create_session_id()
    app.logger.info(f'Index loaded for session: {session_id[:8]}...')
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        # Get or create session
        session_id = get_or_create_session_id()
        # responses_client_id = get_or_create_responses_client(session_id)

        data = request.json
        user_message = data.get('message', '').strip()
        
        app.logger.info(f'[{request_id}] Chat request from {request.remote_addr} (session: {session_id[:8]}..., Responses ID: {None})')
        app.logger.debug(f'[{request_id}] User message: {user_message[:100]}...' if len(user_message) > 100 else f'[{request_id}] User message: {user_message}')
        
        if not user_message:
            app.logger.warning(f'[{request_id}] Empty message received')
            return jsonify({'error': 'Empty message'}), 400
                
        # Send message to agent
        app.logger.info(f'[{request_id}] Sending message to agent')
        response = openai_client.responses.create(
            input=[{"role": "user", "content": user_message}],
            previous_response_id=None,
            extra_body={"agent": {"name": agent.name, "type": "agent_reference"}}
        )
        app.logger.debug(f'[{request_id}] Response ID: {response.id}')
        
        # # Process MCP approval requests
        # input_list: ResponseInputParam = []
        # for item in response.output:
        #     if item.type == "mcp_approval_request":
        #         app.logger.info(f'[{request_id}] MCP approval request detected: {item.id}')
        #         input_list.append(
        #             McpApprovalResponse(
        #                 type="mcp_approval_response",
        #                 approve=True,
        #                 approval_request_id=item.id,
        #             )
        #         )
        
        # If there were approval requests, send approval response
        # if input_list:
        #     app.logger.info(f'[{request_id}] Sending approval response')
        #     response = openai_client.responses.create(
        #         input=input_list,
        #         previous_response_id=response.id,
        #         extra_body={"agent": {"name": agent.name, "type": "agent_reference"}}
        #     )
        #     app.logger.debug(f'[{request_id}] Approval response ID: {response.id}')
        
        # Get agent's response
        agent_response = response.output_text
        
        app.logger.info(f'[{request_id}] Agent response received ({len(agent_response)} chars)')
        app.logger.debug(f'[{request_id}] Agent response: {agent_response}')
        app.logger.debug(f'[{request_id}] Agent response: {agent_response[:200]}...' if len(agent_response) > 200 else f'[{request_id}] Agent response: {agent_response}')
        
        return jsonify({'response': agent_response})
        
    except Exception as e:
        app.logger.error(f'[{request_id}] Chat error: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history for the current session"""
    try:
        session_id = get_or_create_session_id()
        
        if session_clients[session_id]:
            old_thread_id = session_clients[session_id]
            # Create a new thread for this session
            thread = openai_client.threads.create()
            session_clients[session_id] = thread.id
            app.logger.info(f'Conversation history cleared for session {session_id[:8]}... (old thread: {old_thread_id}, new thread: {thread.id})')
        
        return jsonify({'success': True})
        
    except Exception as e:
        app.logger.error(f'Clear history error: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    app.logger.warning(f'404 error: {request.url} from {request.remote_addr}')
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f'500 error: {str(e)}', exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)