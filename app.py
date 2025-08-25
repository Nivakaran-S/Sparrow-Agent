from flask import Flask, request, jsonify, render_template, session
import uuid
import logging
from datetime import datetime
import os
import sys


from src.graphs.finalAgentGraph import sparrowAgent
from langchain_core.messages import HumanMessage

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


conversations = {}

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'})
        
        # Get or create conversation thread
        thread_id = session.get('thread_id')
        if not thread_id:
            thread_id = str(uuid.uuid4())
            session['thread_id'] = thread_id
            conversations[thread_id] = {
                'messages': [],
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        
        conversation = conversations.get(thread_id, {
            'messages': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        })
        
        # Prepare the state for the Sparrow Agent
        # Add the new user message to the conversation
        conversation['messages'].append(HumanMessage(content=user_message))
        conversation['last_updated'] = datetime.now()
        conversations[thread_id] = conversation
        
        # Create the input state for Sparrow Agent
        sparrow_input = {
            'messages': conversation['messages'],
            'notes': [],
            'query_brief': '',
            'final_message': ''
        }
        
        logger.info(f"Processing message for thread {thread_id}: {user_message}")
        
        # Run the Sparrow Agent
        result = sparrowAgent.invoke(sparrow_input)
        
        # Extract the response
        response_message = ""
        status_info = ""
        
        # Get the final message or the last AI message
        if result.get('final_message'):
            response_message = result['final_message']
            status_info = "Task completed"
        elif result.get('messages'):
            # Find the last AI message
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and msg.content and msg.content != user_message:
                    response_message = msg.content
                    break
        
        if not response_message:
            response_message = "I'm processing your request. Could you provide more details?"
        
        # Add execution status information
        if result.get('execution_jobs'):
            status_info = f"Executed: {', '.join(result['execution_jobs'])}"
        elif result.get('notes'):
            # Get the last note as status
            status_info = result['notes'][-1] if result['notes'] else ""
        
        # Update conversation with the agent's response
        conversation['messages'] = result.get('messages', conversation['messages'])
        conversations[thread_id] = conversation
        
        logger.info(f"Response generated for thread {thread_id}: {response_message[:100]}...")
        
        return jsonify({
            'success': True,
            'response': response_message,
            'status': status_info,
            'thread_id': thread_id
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        })

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    """Start a new conversation thread"""
    # Clear the session thread
    session.pop('thread_id', None)
    return jsonify({'success': True, 'message': 'New conversation started'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_conversations': len(conversations)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Clean up old conversations periodically (simple cleanup)
    import threading
    import time
    
    def cleanup_conversations():
        while True:
            time.sleep(3600)  # Clean up every hour
            cutoff = datetime.now().timestamp() - 24 * 3600  # 24 hours ago
            
            threads_to_remove = []
            for thread_id, conv in conversations.items():
                if conv['last_updated'].timestamp() < cutoff:
                    threads_to_remove.append(thread_id)
            
            for thread_id in threads_to_remove:
                del conversations[thread_id]
                
            logger.info(f"Cleaned up {len(threads_to_remove)} old conversations")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_conversations, daemon=True)
    cleanup_thread.start()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Sparrow Agent Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)