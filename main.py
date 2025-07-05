import os
import uuid
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from openai import OpenAI
from langgraph.graph import StateGraph
from typing_extensions import Annotated, TypedDict

# Enhanced environment variable loading for Streamlit Cloud
def load_environment_variables():
    """Load environment variables from .env file or Streamlit secrets"""
    env_vars = {'OPENAI_API_KEY': None, 'MONGODB_URI': None}
    
    try:
        # Try to import streamlit for cloud deployment
        import streamlit as st
        # If we're in Streamlit, try to get secrets first
        if hasattr(st, 'secrets'):
            try:
                # Access secrets directly without .get() to see if they exist
                if 'OPENAI_API_KEY' in st.secrets:
                    env_vars['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                if 'MONGODB_URI' in st.secrets:
                    env_vars['MONGODB_URI'] = st.secrets['MONGODB_URI']
                    
                print("Successfully loaded secrets from Streamlit")
                print(f"OPENAI_API_KEY found: {'Yes' if env_vars['OPENAI_API_KEY'] else 'No'}")
                print(f"MONGODB_URI found: {'Yes' if env_vars['MONGODB_URI'] else 'No'}")
                return env_vars
            except Exception as e:
                print(f"Error loading Streamlit secrets: {e}")
                # Fall through to .env file loading
    except ImportError:
        print("Streamlit not available, using .env file")
    
    # Fallback to .env file for local development
    try:
        env_path = find_dotenv()
        if env_path:
            load_dotenv(env_path, override=True)
            print(f"Loaded .env file from: {env_path}")
        
        env_vars['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        env_vars['MONGODB_URI'] = os.getenv('MONGODB_URI')
    except Exception as e:
        print(f"Error loading .env file: {e}")
    
    return env_vars

# Load environment variables with better error handling
try:
    env_vars = load_environment_variables()
    print(f"Environment variables loaded: {list(env_vars.keys())}")
    for key, value in env_vars.items():
        if value:
            print(f"{key}: Found (length: {len(str(value))})")
        else:
            print(f"{key}: Not found")
except Exception as e:
    print(f"Critical error loading environment variables: {e}")
    # Provide fallback for basic functionality
    env_vars = {'OPENAI_API_KEY': None, 'MONGODB_URI': None}

# Set environment variables for the rest of the application
for key, value in env_vars.items():
    if value:
        os.environ[key] = value

api_key = env_vars.get('OPENAI_API_KEY')
mongodb_uri = env_vars.get('MONGODB_URI')

if api_key:
    print(f"OpenAI API key loaded successfully (length: {len(api_key)})")
else:
    print("Warning: OpenAI API key not loaded")

if mongodb_uri:
    print(f"MongoDB URI loaded successfully (length: {len(mongodb_uri)})")
else:
    print("Warning: MongoDB URI not loaded")

# ================== STATE DEFINITION ==================
class AgentState(TypedDict):
    input_type: str
    original_data: Any
    processed_text: str
    analysis_result: str
    questions: list
    user_responses: list
    session_id: str
    timestamp: str
    error_message: str
    current_question_index: int
    questions_answered: bool

# ================== MONGODB CONNECTION ==================
class DatabaseManager:
    def __init__(self):
        # Get MongoDB URI from multiple sources
        self.mongo_uri = None
        
        # Try environment variable first (set by load_environment_variables)
        self.mongo_uri = os.getenv('MONGODB_URI')
        
        # If not found, try Streamlit secrets directly
        if not self.mongo_uri:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'MONGODB_URI' in st.secrets:
                    self.mongo_uri = st.secrets['MONGODB_URI']
                    print("MongoDB URI loaded directly from Streamlit secrets")
            except Exception as e:
                print(f"Could not load MongoDB URI from Streamlit secrets: {e}")
        
        # If still not found, try global env_vars
        if not self.mongo_uri:
            self.mongo_uri = env_vars.get('MONGODB_URI')
        
        print(f"DatabaseManager init - MongoDB URI found: {'Yes' if self.mongo_uri else 'No'}")
        
        if not self.mongo_uri:
            error_msg = "MongoDB URI not found. Please configure MONGODB_URI in Streamlit secrets or .env file."
            print(f"DatabaseManager Error: {error_msg}")
            raise ValueError(error_msg)
        
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client.get_database('agent_system')
            self.collection = self.db.get_collection('sessions')
            # Test the connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise ValueError(f"Failed to connect to MongoDB: {e}")
    
    def store_session_data(self, session_data: dict) -> str:
        """Store complete session data in MongoDB"""
        try:
            # Import mongodb helpers with error handling
            try:
                from utils.mongodb_helpers import prepare_document_for_mongodb
            except ImportError:
                # Fallback if utils module is not available
                print("Warning: utils.mongodb_helpers not found, using basic sanitization")
                prepared_data = self._basic_sanitize(session_data)
            else:
                prepared_data = prepare_document_for_mongodb(session_data)
            
            # Check if we're updating an existing session or creating a new one
            session_id = prepared_data.get('session_id')
            
            if session_id:
                # Try to update existing session first
                existing_session = self.collection.find_one({"session_id": session_id})
                
                if existing_session:
                    # Update existing session
                    if '_id' in prepared_data:
                        del prepared_data['_id']
                    
                    result = self.collection.update_one(
                        {"session_id": session_id},
                        {"$set": prepared_data}
                    )
                    return str(existing_session['_id'])
                else:
                    # Create new session
                    if '_id' in prepared_data:
                        del prepared_data['_id']
                    result = self.collection.insert_one(prepared_data)
                    return str(result.inserted_id)
            else:
                # No session_id provided, create new
                if '_id' in prepared_data:
                    del prepared_data['_id']
                result = self.collection.insert_one(prepared_data)
                return str(result.inserted_id)
                
        except Exception as e:
            import traceback
            print(f"Database Error: {str(e)}")
            print(traceback.format_exc())
            raise Exception(f"Database storage error: {str(e)}")
    
    def _basic_sanitize(self, data):
        """Basic sanitization fallback when utils module is not available"""
        import json
        
        def sanitize_value(value):
            if hasattr(value, 'read'):  # File objects
                return str(value)
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)
        
        if isinstance(data, dict):
            return {k: sanitize_value(v) for k, v in data.items()}
        return sanitize_value(data)
    
    def close_connection(self):
        """Close MongoDB connection"""
        self.client.close()
    
    def get_session(self, session_id: str) -> dict:
        """Retrieve session data by ID"""
        try:
            result = self.collection.find_one({"session_id": session_id})
            return result
        except Exception as e:
            raise Exception(f"Database retrieval error: {str(e)}")
    
    def update_session_with_mom(self, session_id: str, mom_content: str) -> bool:
        """Update an existing session with generated MoM content"""
        try:
            result = self.collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "final_mom_content": mom_content,
                        "mom_generation_complete": True,
                        "mom_generated_timestamp": datetime.now().isoformat(),
                        "status": "completed_with_mom"
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            raise Exception(f"Database MoM update error: {str(e)}")
    
    def get_session_with_mom(self, session_id: str) -> dict:
        """Retrieve session data including MoM if available"""
        try:
            session = self.collection.find_one({"session_id": session_id})
            if session and session.get('final_mom_content'):
                session['has_mom'] = True
            else:
                session['has_mom'] = False
            return session
        except Exception as e:
            raise Exception(f"Database retrieval error: {str(e)}")
    
    def get_all_sessions(self, limit: int = 10) -> list:
        """Get all sessions with limit, enhanced with MoM status information"""
        try:
            # Get most recent sessions first with enhanced display info
            results = list(self.collection.find({}).sort("timestamp", -1).limit(limit))
            
            # Add MoM status to each session
            for session in results:
                session['has_mom'] = bool(session.get('final_mom_content'))
                session['mom_status'] = 'Generated' if session['has_mom'] else 'Not Generated'
            
            return results
        except Exception as e:
            raise Exception(f"Database retrieval error: {str(e)}")
    
    def search_sessions_by_company(self, company_name: str) -> list:
        """Search sessions by company name"""
        try:
            results = list(self.collection.find({
                "company_name": {"$regex": company_name, "$options": "i"}
            }).sort("timestamp", -1))
            return results
        except Exception as e:
            raise Exception(f"Database search error: {str(e)}")
    
    def search_sessions_by_attendee(self, attendee_name: str) -> list:
        """Search sessions by attendee name"""
        try:
            results = list(self.collection.find({
                "attendees": {"$regex": attendee_name, "$options": "i"}
            }).sort("timestamp", -1))
            return results
        except Exception as e:
            raise Exception(f"Database search error: {str(e)}")
    
    def check_duplicate_data(self, processed_text: str, threshold: float = 0.9) -> dict:
        """Check if similar data already exists in the database"""
        try:
            # Get recent sessions to check for duplicates
            recent_sessions = list(self.collection.find({}).sort("timestamp", -1).limit(50))
            
            for session in recent_sessions:
                existing_text = session.get('processed_text', '')
                
                if existing_text and processed_text:
                    # Simple similarity check based on text length and content overlap
                    similarity = self._calculate_similarity(processed_text, existing_text)
                    
                    if similarity > threshold:
                        return {
                            "is_duplicate": True,
                            "existing_session": session,
                            "similarity": similarity
                        }
            
            return {"is_duplicate": False}
            
        except Exception as e:
            # If check fails, proceed with storage
            return {"is_duplicate": False, "error": str(e)}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word-based similarity calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def update_session_field(self, session_id: str, field_name: str, field_value: Any) -> bool:
        """Update a single field in a session document more safely"""
        try:
            from utils.mongodb_helpers import sanitize_for_mongodb
            
            # Sanitize the value before storing
            safe_value = sanitize_for_mongodb(field_value)
            
            # Create update document with just this field
            update_doc = {field_name: safe_value}
            
            # Update only this field
            result = self.collection.update_one(
                {"session_id": session_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
        except Exception as e:
            import traceback
            print(f"Field update error: {str(e)}")
            print(traceback.format_exc())
            raise Exception(f"Database field update error: {str(e)}")
    
# ================== STARTER FUNCTION ==================
def starter_node(state: AgentState) -> AgentState:
    """
    Starter function that routes user input to appropriate processing.
    """
    try:
        input_type = state.get('input_type')
        original_data = state.get('original_data')
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Validate input type
        if input_type not in ['text', 'audio']:
            state['error_message'] = "Invalid input_type. Must be 'text' or 'audio'"
            return state
        
        if input_type == 'text':
            # Validate text data
            if not original_data or not isinstance(original_data, str):
                state['error_message'] = "Text data must be a non-empty string"
                return state
            
            # For text input, processed_text is the same as original_data
            state['processed_text'] = original_data
            
        elif input_type == 'audio':
            # Audio will be processed in the next node
            if not original_data:
                state['error_message'] = "Audio file data is required"
                return state
        
        state['session_id'] = session_id
        state['timestamp'] = timestamp
        
        return state
        
    except Exception as e:
        state['error_message'] = f"Starter error: {str(e)}"
        return state

# ================== AUDIO TRANSCRIBER ==================
def audio_transcriber_node(state: AgentState) -> AgentState:
    """
    Transcribes audio file using OpenAI Whisper
    """
    try:
        # Skip if not audio input
        if state.get('input_type') != 'audio':
            return state
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            state['error_message'] = "OpenAI API key not found in environment variables"
            return state
        
        client = OpenAI(api_key=api_key)
        audio_file = state.get('original_data')
        
        print(f"Debug - Audio file type: {type(audio_file)}")
        print(f"Debug - Audio file info: {audio_file}")
        
        # Handle different input types
        try:
            if isinstance(audio_file, str):
                # File path
                if not os.path.exists(audio_file):
                    state['error_message'] = f"Audio file not found: {audio_file}"
                    return state
                
                with open(audio_file, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",  # Correct model name for Whisper API
                        file=file,
                        language="en"  # Specify language if known
                    )
            else:
                # File object (from Streamlit)
                if hasattr(audio_file, 'seek'):
                    audio_file.seek(0)
                
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",  # Correct model name for Whisper API
                    file=audio_file,
                    language="en"  # Specify language if known
                )
            
            # Extract transcription text
            if hasattr(transcription, 'text'):
                transcription_text = transcription.text
            else:
                print(f"Debug - Transcription response: {transcription}")
                transcription_text = str(transcription)
            
            if not transcription_text or transcription_text.strip() == "":
                state['error_message'] = "No transcription generated"
                return state
            
            print(f"Debug - Transcription successful: {transcription_text[:100]}...")
            state['processed_text'] = transcription_text
            
        except Exception as e:
            print(f"Debug - Transcription error details: {str(e)}")
            state['error_message'] = f"Transcription failed: {str(e)}"
            return state
        
        return state
        
    except Exception as e:
        state['error_message'] = f"Audio transcription error: {str(e)}"
        return state

# ================== GENERAL AGENT ==================
def general_agent_node(state: AgentState) -> AgentState:
    """
    GeneralAgent that processes text using OpenAI GPT models
    """
    try:
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            state['error_message'] = "OpenAI API key not found"
            return state
        
        client = OpenAI(api_key=api_key)
        processed_text = state.get('processed_text')
        
        if not processed_text:
            state['error_message'] = "No processed text available for analysis"
            return state
        
        # Enhanced system prompt for call data analysis and labeling
        system_prompt = """
        You are a GeneralAgent AI assistant specialized in analyzing business call data and meeting transcripts. Your task is to:
        
        1. ANALYSIS PHASE:
           - Analyze the provided call/meeting content thoroughly
           - Extract key business insights, decisions made, and important discussion points
           - Identify main topics, challenges discussed, and action items mentioned
           - Summarize the overall purpose and outcomes of the call/meeting
           - Provide a comprehensive but concise analysis focusing on business value
        
        2. LABELING QUESTIONS:
           After your analysis, ask exactly these 3 essential labeling questions to properly categorize and store this data:
           - What is the company name discussed in this call/meeting?
           - Who were the attendees/participants in this meeting?
           - What was the duration of this meeting (approximate time in minutes)?
        
        Format your response EXACTLY as follows:
        
        ANALYSIS:
        [Provide detailed analysis of the call/meeting content, focusing on:
         - Main discussion topics and business context
         - Key decisions or outcomes
         - Important action items or next steps
         - Overall meeting purpose and value
         - Any challenges or opportunities discussed]
        
        LABELING QUESTIONS:
        1. What is the company name discussed in this call/meeting?
        2. Who were the attendees/participants in this meeting?
        3. What was the duration of this meeting (approximate time in minutes)?
        
        Keep your analysis professional, concise, and business-focused. The labeling questions are mandatory for data organization purposes.
        """
        
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyze this call/meeting data:\n\n{processed_text}"}
            ],
            temperature=0.7
        )
        
        analysis_result = response.choices[0].message.content
        
        # Parse analysis and questions
        if "LABELING QUESTIONS:" in analysis_result:
            parts = analysis_result.split("LABELING QUESTIONS:")
            analysis = parts[0].replace("ANALYSIS:", "").strip()
            questions_text = parts[1].strip()
            
            # Extract the 3 specific labeling questions
            questions = [
                "What is the company name discussed in this call/meeting?",
                "Who were the attendees/participants in this meeting?", 
                "What was the duration of this meeting (approximate time in minutes)?"
            ]
        else:
            # Fallback if format is not followed
            analysis = analysis_result
            questions = [
                "What is the company name discussed in this call/meeting?",
                "Who were the attendees/participants in this meeting?",
                "What was the duration of this meeting (approximate time in minutes)?"
            ]
        
        state['analysis_result'] = analysis
        state['questions'] = questions
        state['current_question_index'] = 0
        state['questions_answered'] = False
        state['user_responses'] = []
        
        return state
        
    except Exception as e:
        state['error_message'] = f"GeneralAgent processing error: {str(e)}"
        return state

def process_user_response(state: AgentState, response: str) -> AgentState:
    """Process user response to questions"""
    try:
        if not state.get('questions'):
            state['error_message'] = "No questions available"
            return state
            
        current_index = state.get('current_question_index', 0)
        user_responses = state.get('user_responses', [])
        
        # Add response to the current question
        user_responses.append({
            'question': state['questions'][current_index],
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        state['user_responses'] = user_responses
        
        # Move to next question or mark as completed
        if current_index + 1 < len(state['questions']):
            state['current_question_index'] = current_index + 1
        else:
            state['questions_answered'] = True
            
        return state
        
    except Exception as e:
        state['error_message'] = f"Response processing error: {str(e)}"
        return state

# ================== DATABASE STORAGE ==================
def database_storage_node(state: AgentState) -> AgentState:
    """
    Store complete session data in MongoDB with enhanced labeling
    """
    try:
        db_manager = DatabaseManager()
        
        # Extract labels from user responses
        company_name = ""
        attendees = ""
        duration = ""
        
        for response in state.get('user_responses', []):
            question = response.get('question', '').lower()
            answer = response.get('response', '')
            
            if 'company name' in question:
                company_name = answer
            elif 'attendees' in question or 'participants' in question:
                attendees = answer
            elif 'duration' in question:
                duration = answer
        
        # Check for duplicate data
        duplicate_check = db_manager.check_duplicate_data(state.get('processed_text', ''))
        
        if duplicate_check.get('is_duplicate'):
            existing_session = duplicate_check.get('existing_session', {})
            state['error_message'] = f"Duplicate data detected. Similar session already exists (ID: {existing_session.get('session_id')})."
            state['duplicate_session'] = existing_session
            return state
        
        # Prepare enhanced data for storage with labels
        session_data = {
            "session_id": state.get('session_id'),
            "timestamp": state.get('timestamp'),
            "input_type": state.get('input_type'),
            "original_data": str(state.get('original_data'))[:1000],  # Limit size for storage
            "processed_text": state.get('processed_text'),
            "analysis_result": state.get('analysis_result'),
            "questions": state.get('questions', []),
            "user_responses": state.get('user_responses', []),
            
            # Enhanced labeling fields for better retrieval
            "labels": {
                "company_name": company_name,
                "attendees": attendees,
                "duration": duration
            },
            
            # Search-friendly fields
            "company_name": company_name,
            "attendees": attendees,
            "meeting_duration": duration,
            
            "status": "completed" if not state.get('error_message') else "error",
            "error_message": state.get('error_message', None)
        }
        
        # Store in database
        document_id = db_manager.store_session_data(session_data)
        state['document_id'] = document_id
        
        db_manager.close_connection()
        
        return state
        
    except Exception as e:
        state['error_message'] = f"Database storage error: {str(e)}"
        return state

# ================== WORKFLOW DEFINITION ==================
def create_workflow():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("starter", starter_node)
    workflow.add_node("audio_transcriber", audio_transcriber_node)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("database_storage", database_storage_node)
    
    # Define edges
    workflow.set_entry_point("starter")
    workflow.add_edge("starter", "audio_transcriber")
    workflow.add_edge("audio_transcriber", "general_agent")
    workflow.add_edge("general_agent", "database_storage")
    workflow.set_finish_point("database_storage")
    
    return workflow.compile()

# ================== MAIN EXECUTION FUNCTION ==================
def process_user_input(input_type: str, data: Any) -> Dict:
    """
    Main function to process user input through the complete workflow
    
    Args:
        input_type (str): Either 'text' or 'audio'
        data: Text string or audio file path/object
    
    Returns:
        Dict: Result containing success/error status and additional information
    """
    try:
        # Validate OpenAI API key with more detailed error
        api_key = env_vars.get('OPENAI_API_KEY')
        if not api_key:
            return {
                "status": "error",
                "message": "OpenAI API key not found. Please configure OPENAI_API_KEY in Streamlit secrets or .env file.",
                "details": [
                    "For Streamlit Cloud:",
                    "1. Go to your app settings",
                    "2. Click on 'Secrets'",
                    "3. Add: OPENAI_API_KEY = \"your-openai-api-key\"",
                    "4. Redeploy your app",
                    "",
                    "For local development:",
                    "1. Create a .env file",
                    "2. Add: OPENAI_API_KEY=your-openai-api-key"
                ]
            }

        # Create workflow
        app = create_workflow()
        
        # Initial state
        initial_state = {
            "input_type": input_type,
            "original_data": data,
            "processed_text": "",
            "analysis_result": "",
            "questions": [],
            "user_responses": [],
            "session_id": "",
            "timestamp": "",
            "error_message": "",
            "current_question_index": 0,
            "questions_answered": False
        }
        
        # Run initial workflow up to questions
        result = app.invoke(initial_state)
        
        if result.get('error_message'):
            return {
                "status": "error",
                "message": result['error_message']
            }
        
        # Return analysis and first question
        return {
            "status": "success",
            "message": "Analysis completed",
            "session_id": result.get('session_id'),
            "analysis": result.get('analysis_result'),
            "current_question": result['questions'][0],
            "question_index": 0,
            "total_questions": len(result['questions'])
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Workflow execution error: {str(e)}",
            "details": str(e)
        }

def handle_user_response(session_id: str, response: str) -> Dict:
    """Handle user response and determine next action - Modified for bulk processing"""
    try:
        # Get session from MongoDB
        db_manager = DatabaseManager()
        session = db_manager.get_session(session_id)
        
        if not session:
            return {"status": "error", "message": "Session not found"}
        
        # Since we're now handling all questions at once in the UI,
        # this function is mainly for compatibility
        # The actual processing happens in the UI form submission
        
        return {
            "status": "pending",
            "message": "Please complete all labeling questions in the form above."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Response handling error: {str(e)}"
        }

# ================== MoM GENERATION ==================
def generate_mom(session_data: dict) -> Dict:
    """
    Generate Minutes of Meeting from session data using the agent workflow
    
    Args:
        session_data: Dictionary containing session information
        
    Returns:
        Dict: Result containing success/error status and generated MoM
    """
    try:
        from agents import initialize_state
        
        # Initialize state for the agent workflow using the function from agents.py
        agent_state = initialize_state()
        
        # Prepare meeting notes from session data
        processed_text = session_data.get('processed_text', '')
        analysis = session_data.get('analysis_result', '')
        
        # Combine processed text and analysis for comprehensive input
        meeting_notes = f"""
        Meeting Transcript:
        {processed_text}
        
        Initial Analysis:
        {analysis}
        """
        
        # Set the notes in the agent state
        agent_state["notes"] = meeting_notes
        agent_state["mode"] = "meeting_notes"
        agent_state["current_phase"] = "qa_phase"
        
        # Return success with the agent state for processing in the UI
        return {
            "status": "success",
            "message": "Meeting data loaded for processing",
            "agent_state": agent_state,
            "current_phase": "qa_phase"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error initializing meeting data processing: {str(e)}",
            "details": str(e)
        }

def process_agent_workflow(agent_state, current_phase):
    """Process the agent workflow by calling the appropriate function from agents.py"""
    from agents import (
        qa_agent_node, intelligent_agent_node, 
        reflection_agent_node, classification_tool_node, 
        final_output_node
    )
    
    # Simply use the existing agent nodes from agents.py
    if current_phase == "qa_phase":
        updated_state = qa_agent_node(agent_state)
        return updated_state, "intelligent_agent_phase"
        
    elif current_phase == "intelligent_agent_phase":
        updated_state = intelligent_agent_node(agent_state)
        return updated_state, "reflection_agent_phase"
        
    elif current_phase == "reflection_agent_phase":
        updated_state = reflection_agent_node(agent_state)
        return updated_state, "classification_tool_phase"
        
    elif current_phase == "classification_tool_phase":
        updated_state = classification_tool_node(agent_state)
        
        # Determine next phase based on iteration count as per agents.py logic
        if updated_state.get("iteration_count", 0) >= updated_state.get("max_iterations", 2):
            return updated_state, "question_answer_manager_phase"
        else:
            return updated_state, "intelligent_agent_phase"
        
    elif current_phase == "final_output":
        updated_state = final_output_node(agent_state)
        return updated_state, "complete"
        
    else:
        raise ValueError(f"Unknown phase: {current_phase}")