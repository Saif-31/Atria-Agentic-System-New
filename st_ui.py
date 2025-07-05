import streamlit as st
import os
import tempfile
from datetime import datetime

# Handle imports with error checking for deployment
try:
    from main import process_user_input, handle_user_response, DatabaseManager, generate_mom, process_agent_workflow
    from model import get_openai_model
    from agents import process_user_input as agent_process_input
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please check that all required packages are installed correctly.")
    st.stop()


# Initialize session state variables
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'questions_complete' not in st.session_state:
    st.session_state.questions_complete = False
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
# Add new state variable for greeting
if 'greeting_shown' not in st.session_state:
    st.session_state.greeting_shown = False
# Add new state variables for interaction flow
if 'user_choice_made' not in st.session_state:
    st.session_state.user_choice_made = False
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I assist you with MoM generation? Do you want to process with some old call data or you want to process new one?"}
    ]
# Add new state variable for old data retrieval
if 'show_old_data' not in st.session_state:
    st.session_state.show_old_data = False
if 'old_data_selected' not in st.session_state:
    st.session_state.old_data_selected = None
# Add new state variables for MoM generation
if 'mom_generation_prompt' not in st.session_state:
    st.session_state.mom_generation_prompt = False
if 'mom_generation_approved' not in st.session_state:
    st.session_state.mom_generation_approved = False
if 'mom_generation_state' not in st.session_state:
    st.session_state.mom_generation_state = None
if 'mom_generation_complete' not in st.session_state:
    st.session_state.mom_generation_complete = False
if 'current_mom_question' not in st.session_state:
    st.session_state.current_mom_question = None
# Add new state variables for the agent workflow
if 'agent_workflow_phase' not in st.session_state:
    st.session_state.agent_workflow_phase = ""
if 'refined_questions' not in st.session_state:
    st.session_state.refined_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'questions_presented' not in st.session_state:
    st.session_state.questions_presented = False
if 'human_in_loop_completed' not in st.session_state:
    st.session_state.human_in_loop_completed = False
# Add duplicate detection state variables
if 'duplicate_detected' not in st.session_state:
    st.session_state.duplicate_detected = False
if 'duplicate_session' not in st.session_state:
    st.session_state.duplicate_session = None
if 'duplicate_similarity' not in st.session_state:
    st.session_state.duplicate_similarity = 0
if 'duplicate_input_type' not in st.session_state:
    st.session_state.duplicate_input_type = ""
if 'duplicate_input_data' not in st.session_state:
    st.session_state.duplicate_input_data = None
if 'duplicate_temp_result' not in st.session_state:
    st.session_state.duplicate_temp_result = None

def process_input():
    """Handle input processing and update session state"""
    try:
        if st.session_state.input_type == "text":
            if st.session_state.text_input:
                # Check for duplicate data before processing
                try:
                    db_manager = DatabaseManager()
                    duplicate_check = db_manager.check_duplicate_data(st.session_state.text_input)
                    
                    if duplicate_check.get("is_duplicate"):
                        existing_session = duplicate_check.get("existing_session")
                        similarity = duplicate_check.get("similarity", 0)
                        
                        # Set duplicate state to show the warning and buttons
                        st.session_state.duplicate_detected = True
                        st.session_state.duplicate_session = existing_session
                        st.session_state.duplicate_similarity = similarity
                        st.session_state.duplicate_input_type = "text"
                        st.session_state.duplicate_input_data = st.session_state.text_input
                        return  # Exit early to show duplicate warning
                except Exception as db_error:
                    st.error(f"Database connection error: {str(db_error)}")
                    st.error("Please check that MongoDB URI is correctly configured in Streamlit secrets.")
                    return
                
                # Proceed with normal processing if no duplicates
                result = process_user_input("text", st.session_state.text_input)
                st.session_state.result = result
                st.session_state.processing_complete = True
                st.session_state.current_session = result.get('session_id')
                
        else:  # audio input
            if st.session_state.audio_file is not None:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(st.session_state.audio_file.getbuffer())
                    st.session_state.temp_file_path = tmp_file.name
                    
                    # First transcribe to check for duplicates
                    temp_result = process_user_input("audio", tmp_file.name)
                    
                    if temp_result.get("status") == "success":
                        # Get the transcribed text for duplicate check
                        db_manager = DatabaseManager()
                        temp_session = db_manager.get_session(temp_result.get('session_id'))
                        transcribed_text = temp_session.get('processed_text', '') if temp_session else ''
                        
                        if transcribed_text:
                            duplicate_check = db_manager.check_duplicate_data(transcribed_text)
                            
                            if duplicate_check.get("is_duplicate"):
                                existing_session = duplicate_check.get("existing_session")
                                similarity = duplicate_check.get("similarity", 0)
                                
                                # Set duplicate state to show the warning and buttons
                                st.session_state.duplicate_detected = True
                                st.session_state.duplicate_session = existing_session
                                st.session_state.duplicate_similarity = similarity
                                st.session_state.duplicate_input_type = "audio"
                                st.session_state.duplicate_temp_result = temp_result
                                return  # Exit early to show duplicate warning
                    
                    # Proceed with normal processing if no duplicates
                    st.session_state.result = temp_result
                    st.session_state.processing_complete = True
                    st.session_state.current_session = temp_result.get('session_id')
                    
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")

def new_chat():
    """Reset all session state variables"""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

def handle_initial_response(response):
    """Process user's initial response to greeting"""
    st.session_state.messages.append({"role": "user", "content": response})
    
    # Check for keywords indicating choice
    response_lower = response.lower()
    new_data_keywords = ["new", "process new", "new one", "latest"]
    old_data_keywords = ["old", "previous", "continue with old", "stored", "existing"]
    
    if any(keyword in response_lower for keyword in new_data_keywords):
        # User wants to process new data
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Great! Please select the input type and provide your call data below."
        })
        st.session_state.user_choice_made = True
        st.session_state.show_old_data = False
    elif any(keyword in response_lower for keyword in old_data_keywords):
        # User wants to process old data
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I'll retrieve your previously stored call data. Please select one to continue."
        })
        st.session_state.user_choice_made = True
        st.session_state.show_old_data = True
    else:
        # Default case (unclear choice)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I'll help you process data. Would you like to use existing stored data or process new data?"
        })
        # Don't mark choice as made, wait for clearer response
        st.session_state.user_choice_made = False

# Sidebar
with st.sidebar:
    st.title("Chat Options")
    if st.button("New Chat", use_container_width=True, type="primary"):
        new_chat()

# Main UI
st.title("Atria Agentic System")
st.markdown("---")

# Chat message display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initial response input
if not st.session_state.user_choice_made:
    user_response = st.chat_input("Your response...")
    if user_response:
        handle_initial_response(user_response)
        st.rerun()

# Show input options based on user choice
if st.session_state.user_choice_made:
    if st.session_state.show_old_data:
        try:
            # Retrieve old call data with better error handling
            try:
                db_manager = DatabaseManager()
                old_sessions = db_manager.get_all_sessions(limit=20)
            except ValueError as ve:
                st.error(f"Database Configuration Error: {str(ve)}")
                st.error("Please ensure MONGODB_URI is properly configured in your Streamlit app secrets.")
                st.info("To fix this:")
                st.code("1. Go to your Streamlit app settings\n2. Click on 'Secrets'\n3. Add: MONGODB_URI = \"your-mongodb-connection-string\"")
                st.session_state.show_old_data = False
                st.rerun()  # Changed from return to st.rerun()
            except Exception as db_error:
                st.error(f"Database Connection Error: {str(db_error)}")
                st.error("Failed to connect to MongoDB. Please check your connection string.")
                st.session_state.show_old_data = False
                st.rerun()  # Changed from return to st.rerun()
            
            if not old_sessions:
                st.info("No previous call data found in the database.")
                # Show new data option as fallback
                st.session_state.show_old_data = False
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "No previous data found. Let's process new data instead."
                })
            else:
                # Show selection for old sessions
                st.subheader("Select Previous Call Data")
                
                # Create a formatted list of sessions
                session_options = {}
                for session in old_sessions:
                    # Format timestamp for display
                    timestamp = session.get('timestamp', '').split('T')[0]
                    session_type = session.get('input_type', 'unknown')
                    session_id = session.get('session_id', '')
                    
                    # Enhanced display with labels
                    company = session.get('company_name', 'Unknown Company')
                    attendees = session.get('attendees', 'Unknown Attendees')
                    duration = session.get('meeting_duration', 'Unknown Duration')
                    
                    # Create a more informative label
                    label = f"{timestamp} - {company} | {attendees[:30]}{'...' if len(attendees) > 30 else ''} | {duration}"
                    
                    session_options[label] = session_id
                
                # Display as radio buttons
                selected_label = st.radio("Select a session:", list(session_options.keys()))
                
                if st.button("Load Selected Session", type="primary"):
                    selected_id = session_options[selected_label]
                    session_data = db_manager.get_session(selected_id)
                    
                    if session_data:
                        # Display the selected session data
                        st.session_state.old_data_selected = session_data
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Here's the call data from {session_data.get('timestamp', '').split('T')[0]}:"
                        })
                        
                        # Store minimal result data without triggering questions
                        st.session_state.result = {
                            "status": "success",
                            "session_id": selected_id,
                            "analysis": session_data.get('analysis_result', ''),
                        }
                        # Mark as processing complete but don't set questions_complete to False
                        st.session_state.processing_complete = True
                        st.session_state.current_session = selected_id
                        st.session_state.questions_complete = True  # Mark questions as complete to avoid showing them
                        st.session_state.mom_generation_prompt = True  # Enable MoM generation option
                        st.rerun()

                # Option to process new data instead
                if st.button("Process New Data Instead", type="secondary"):
                    st.session_state.show_old_data = False
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Let's process new data instead. Please select the input type below."
                    })
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error retrieving previous data: {str(e)}")
            st.session_state.show_old_data = False
    elif not st.session_state.processing_complete:
        # Input type selection
        st.session_state.input_type = st.radio(
            "Select Input Type:",
            ["text", "audio"],
            key="input_type_radio"
        )

        # Input section
        if st.session_state.input_type == "text":
            st.text_area(
                "Enter your text:",
                key="text_input",
                height=150,
                placeholder="Type or paste your text here..."
            )
        else:
            st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3'],
                key="audio_file",
                help="Supported formats: WAV, MP3"
            )

        # Process button
        if st.button("Process Input", type="primary", use_container_width=True):
            process_input()

# Handle duplicate detection warning
if st.session_state.duplicate_detected:
    st.markdown("---")
    st.warning(f"⚠️ Similar call data already exists in the database (Similarity: {st.session_state.duplicate_similarity:.1%})")
    
    with st.expander("View Existing Data", expanded=False):
        existing_session = st.session_state.duplicate_session
        st.write(f"**Date:** {existing_session.get('timestamp', '').split('T')[0]}")
        st.write(f"**Company:** {existing_session.get('company_name', 'Unknown')}")
        st.write(f"**Attendees:** {existing_session.get('attendees', 'Unknown')}")
        st.write(f"**Duration:** {existing_session.get('meeting_duration', 'Unknown')}")
    
    # Ask user if they want to proceed
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Anyway", type="secondary", key="process_anyway_btn"):
            # Process the new data anyway
            if st.session_state.duplicate_input_type == "text":
                result = process_user_input("text", st.session_state.duplicate_input_data)
            else:
                result = st.session_state.duplicate_temp_result
            
            st.session_state.result = result
            st.session_state.processing_complete = True
            st.session_state.current_session = result.get('session_id')
            
            # Clear duplicate state
            st.session_state.duplicate_detected = False
            st.session_state.duplicate_session = None
            st.session_state.duplicate_similarity = 0
            st.session_state.duplicate_input_type = ""
            st.session_state.duplicate_input_data = None
            st.session_state.duplicate_temp_result = None
            
            st.rerun()
    
    with col2:
        if st.button("Use Existing Data", type="primary", key="use_existing_btn"):
            # Load the existing session
            existing_session = st.session_state.duplicate_session
            st.session_state.old_data_selected = existing_session
            st.session_state.result = {
                "status": "success",
                "session_id": existing_session.get('session_id'),
                "analysis": existing_session.get('analysis_result', ''),
            }
            st.session_state.processing_complete = True
            st.session_state.current_session = existing_session.get('session_id')
            st.session_state.questions_complete = True
            
            # Set up to show MoM generation prompt
            st.session_state.mom_generation_prompt = True
            st.session_state.mom_generation_approved = False
            
            # Clear duplicate state
            st.session_state.duplicate_detected = False
            st.session_state.duplicate_session = None
            st.session_state.duplicate_similarity = 0
            st.session_state.duplicate_input_type = ""
            st.session_state.duplicate_input_data = None
            st.session_state.duplicate_temp_result = None
            
            st.success("✅ Loaded existing call data successfully!")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I've loaded the existing call data. Would you like to process this meeting data to generate minutes?"
            })
            
            st.rerun()

# Results section
if st.session_state.processing_complete and st.session_state.result:
    st.markdown("---")
    
    if st.session_state.result.get("status") == "error":
        st.error(st.session_state.result.get("message"))
        if "details" in st.session_state.result:
            st.error("\n".join(st.session_state.result["details"]))
    else:
        # Display analysis
        st.subheader("Analysis")
        st.write(st.session_state.result.get("analysis"))
        
        # MoM Generation Prompt for old data
        if st.session_state.mom_generation_prompt and not st.session_state.mom_generation_approved:
            st.markdown("---")
            st.subheader("Meeting Data Processing")
            
            # Display the call data first
            with st.expander("View Meeting Transcript", expanded=True):
                st.write(st.session_state.old_data_selected.get('processed_text', 'No transcript available'))
            
            st.info("Would you like to process this meeting data to generate minutes?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process Meeting Data", type="primary"):
                    # start MoM generation workflow using agents.py
                    if st.session_state.old_data_selected:
                        try:
                            # Initialize agent state from agents.py
                            from agents import initialize_state, create_integrated_graph
                            
                            # Initialize the agent state
                            agent_state = initialize_state()
                            
                            # Prepare meeting notes from session data
                            processed_text = st.session_state.old_data_selected.get('processed_text', '')
                            analysis = st.session_state.old_data_selected.get('analysis_result', '')
                            
                            # Combine processed text and analysis for comprehensive input
                            meeting_notes = f"""
                            Meeting Transcript:
                            {processed_text}
                            
                            Initial Analysis:
                            {analysis}
                            """
                            
                            # Set the notes and configuration in the agent state
                            agent_state["notes"] = meeting_notes
                            agent_state["mode"] = "meeting_notes"
                            agent_state["current_phase"] = "qa_phase"
                            agent_state["status"] = "Processing meeting notes"
                            agent_state["notes_processed"] = False
                            
                            # Update UI state
                            st.session_state.mom_generation_state = agent_state
                            st.session_state.mom_generation_approved = True
                            st.session_state.agent_workflow_phase = "processing"
                            
                            # Add initial message
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "I'll analyze your meeting data and generate refined questions. This process involves multiple steps..."
                            })
                            
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error running agent workflow: {str(e)}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error starting MoM generation: {str(e)}"
                            })
                    st.rerun()
            
            with col2:
                if st.button("Cancel", type="secondary"):
                    st.session_state.mom_generation_prompt = False
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "No, I don't want to generate MoM"
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Alright, let me know if you need anything else."
                    })
                    st.rerun()
        
        # MoM Processing Flow - Step by step with progress
        elif st.session_state.mom_generation_approved and st.session_state.agent_workflow_phase == "processing":
            st.markdown("---")
            st.subheader("Meeting Data Analysis")
            
            # Step-by-step processing
            if not st.session_state.mom_generation_state.get("notes_processed", False):
                
                # Phase 1: Process with qa_agent_node
                with st.spinner("Step 1/4: Analyzing meeting notes..."):
                    from agents import qa_agent_node
                    updated_state = qa_agent_node(st.session_state.mom_generation_state)
                    st.session_state.mom_generation_state = updated_state
                    st.session_state.mom_generation_state["notes_processed"] = True
                    st.success("✅ Notes Analysis Complete!")
                
                # Phase 2: Process with intelligent agent
                st.info("Step 2/4: Generating intelligent questions...")
                with st.spinner("Generating questions from notes..."):
                    from agents import intelligent_agent_node
                    # Set conversation history for intelligent agent
                    st.session_state.mom_generation_state["intelligent_conversation_history"] = [
                        {"role": "assistant", "content": st.session_state.mom_generation_state.get("meeting_details", {}).get("extracted_from_notes", "")}
                    ]
                    updated_state = intelligent_agent_node(st.session_state.mom_generation_state)
                    st.session_state.mom_generation_state = updated_state
                    st.success("✅ Question Generation Complete!")
                
                # Phase 3: Process with reflection agent
                st.info("Step 3/4: Refining generated questions...")
                with st.spinner("Improving question quality..."):
                    from agents import reflection_agent_node
                    updated_state = reflection_agent_node(st.session_state.mom_generation_state)
                    st.session_state.mom_generation_state = updated_state
                    st.success("✅ Question Refinement Complete!")
                
                # Phase 4: Process with classification
                st.info("Step 4/4: Classifying questions...")
                with st.spinner("Identifying unanswered questions..."):
                    from agents import classification_tool_node
                    updated_state = classification_tool_node(st.session_state.mom_generation_state)
                    st.session_state.mom_generation_state = updated_state
                    
                    # Get unanswered questions
                    unanswered = updated_state.get("classified_unanswered_questions", [])
                    st.session_state.refined_questions = unanswered
                    
                    if unanswered:
                        st.success(f"✅ Analysis Complete! Found {len(unanswered)} questions that need answers.")
                        st.session_state.agent_workflow_phase = "questions_ready"
                        st.session_state.questions_presented = True
                        
                        # Add message about questions
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Analysis complete! I've identified {len(unanswered)} questions that need your input to generate comprehensive meeting minutes."
                        })
                    else:
                        st.success("✅ Analysis Complete! All information is available - ready to generate MoM.")
                        st.session_state.agent_workflow_phase = "human_in_loop"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Analysis complete! All necessary information has been extracted. Ready to generate the final Minutes of Meeting."
                        })
                
                st.rerun()

        # Show Questions with Form Interface - Present All Questions Together
        elif st.session_state.mom_generation_approved and st.session_state.agent_workflow_phase == "questions_ready":
            st.markdown("---")
            st.subheader("Meeting Questions")
            st.write("Please answer the following questions about the meeting:")
            
            unanswered = st.session_state.refined_questions
            
            # Display all questions in a single form
            with st.form("all_questions_form"):
                st.info(f"Found {len(unanswered)} questions that need your input:")
                
                questions_answers = {}
                
                # Display all questions with text areas
                for i, question in enumerate(unanswered):
                    st.markdown(f"**Q{i+1}:** {question}")
                    answer = st.text_area(
                        f"Answer {i+1}:", 
                        key=f"question_answer_{i}",
                        height=100,
                        placeholder="Enter your answer here..."
                    )
                    questions_answers[question] = answer
                    st.markdown("---")
                
                # Submit all answers at once
                col1, col2 = st.columns([3, 1])
                with col1:
                    submitted = st.form_submit_button("Submit All Answers", type="primary")
                with col2:
                    skip_all = st.form_submit_button("Skip All Questions", type="secondary")
                
                if submitted:
                    # Check if all questions have answers
                    answered_questions = {q: a for q, a in questions_answers.items() if a.strip()}
                    unanswered_count = len(unanswered) - len(answered_questions)
                    
                    if len(answered_questions) == 0:
                        st.error("Please answer at least one question to proceed.")
                    else:
                        # Store the answers
                        st.session_state.qa_answers = answered_questions
                        
                        if unanswered_count > 0:
                            st.warning(f"You answered {len(answered_questions)} out of {len(unanswered)} questions. {unanswered_count} questions were skipped.")
                        else:
                            st.success("All questions answered!")
                        
                        # Proceed to summary
                        st.session_state.agent_workflow_phase = "qa_complete"
                        st.rerun()
                
                elif skip_all:
                    # Skip all questions
                    st.session_state.qa_answers = {}
                    st.session_state.agent_workflow_phase = "qa_complete"
                    st.warning("All questions skipped. Proceeding with available information.")
                    st.rerun()

        # Q&A Complete - Show Summary and Generate MoM
        elif st.session_state.mom_generation_approved and st.session_state.agent_workflow_phase == "qa_complete":
            st.markdown("---")
            st.subheader("Questions & Answers Summary")
            
            # Prepare complete Q&A list
            all_qa_pairs = []
            
            # Add previously answered questions (from meeting notes)
            from agents import find_answer_in_history
            for q in st.session_state.mom_generation_state.get("classified_answered_questions", []):
                answer = find_answer_in_history(q, st.session_state.mom_generation_state["conversation_history"])
                all_qa_pairs.append({"question": q, "answer": answer or "Found in meeting notes"})
            
            # Add newly answered questions
            for q, a in st.session_state.qa_answers.items():
                all_qa_pairs.append({"question": q, "answer": a})
            
            # Display complete Q&A summary
            st.success("✅ All questions have been processed!")
            
            with st.expander("View Complete Q&A Summary", expanded=True):
                for i, qa in enumerate(all_qa_pairs, 1):
                    st.markdown(f"**Q{i}:** {qa['question']}")
                    st.markdown(f"**A:** {qa['answer']}")
                    st.markdown("---")
            
            # Update agent state with all answers
            st.session_state.mom_generation_state["qa_manager_answered_questions"] = all_qa_pairs
            st.session_state.mom_generation_state["qa_manager_complete"] = True
            st.session_state.agent_workflow_phase = "human_in_loop"
            
            # Auto-proceed to human in loop
            st.rerun()

        # Human in the loop confirmation
        elif st.session_state.mom_generation_approved and st.session_state.agent_workflow_phase == "human_in_loop":
            st.markdown("---")
            st.subheader("Generate Minutes of Meeting")
            st.info("All questions have been answered. Ready to generate the final Minutes of Meeting?")
            
            if st.button("Generate Final MoM", type="primary"):
                try:
                    # Use final_mom_agent_node directly from agents.py
                    from agents import final_mom_agent_node
                    
                    # Generate MoM using the Final MoM Agent
                    with st.spinner("Generating comprehensive Meeting Minutes..."):
                        final_state = final_mom_agent_node(st.session_state.mom_generation_state)
                    
                    # Check if MoM generation was successful
                    if final_state.get("error_message"):
                        st.error(f"Error generating MoM: {final_state['error_message']}")
                    elif final_state.get("final_mom_content"):
                        # Extract the generated MoM content
                        mom_content = final_state["final_mom_content"]
                        
                        # Store the MoM in the database under the same session ID
                        try:
                            db_manager = DatabaseManager()
                            session_id = st.session_state.current_session
                            
                            # Get the existing session data
                            existing_session = db_manager.get_session(session_id)
                            if existing_session:
                                # Update the session with the generated MoM
                                existing_session['final_mom_content'] = mom_content
                                existing_session['mom_generation_complete'] = True
                                existing_session['mom_generated_timestamp'] = datetime.now().isoformat()
                                
                                # Store the updated session
                                db_manager.store_session_data(existing_session)
                                st.success("✅ MoM saved to database successfully!")
                            else:
                                st.warning("⚠️ Could not find original session to update")
                            
                            db_manager.close_connection()
                            
                        except Exception as db_error:
                            st.error(f"Error saving MoM to database: {str(db_error)}")
                        
                        # Mark completion in UI state and store MoM content persistently
                        st.session_state.mom_generation_complete = True
                        st.session_state.human_in_loop_completed = True
                        st.session_state.final_mom_content = mom_content
                        st.session_state.agent_workflow_phase = "complete"  # Change phase to complete
                        
                        # Add MoM to chat messages for history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "✅ Your Minutes of Meeting has been generated and saved successfully!"
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**Meeting Minutes Generated:**\n\n{mom_content}"
                        })
                        
                        st.rerun()  # Rerun to update the UI state
                        
                    else:
                        st.error("No MoM content was generated. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating final MoM: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Error generating MoM: {str(e)}"
                    })
        
        # Display Final MoM persistently after generation
        elif st.session_state.mom_generation_complete and st.session_state.human_in_loop_completed:
            st.markdown("---")
            st.success("🎉 Minutes of Meeting generated successfully!")
            
            # Always display the MoM if it exists
            if hasattr(st.session_state, 'final_mom_content') and st.session_state.final_mom_content:
                st.markdown("---")
                st.subheader("📄 Meeting Minutes")
                
                # Display the generated MoM in an expandable container
                with st.expander("View Complete Meeting Minutes", expanded=True):
                    st.markdown(st.session_state.final_mom_content)
                
                # Add download functionality with unique keys to prevent conflicts
                st.markdown("### Download Options")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Use unique key and handle download without hiding content
                    st.download_button(
                        label="📥 Download as Text File",
                        data=st.session_state.final_mom_content,
                        file_name=f"meeting_minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_txt_persistent",
                        help="Download the Meeting Minutes as a text file"
                    )
                
                with col2:
                    st.download_button(
                        label="📄 Download as Markdown", 
                        data=st.session_state.final_mom_content,
                        file_name=f"meeting_minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key="download_md_persistent",
                        help="Download the Meeting Minutes as a markdown file"
                    )
            else:
                st.error("No MoM content found. Please try generating again.")

# Footer with chat input for Q&A flow
if st.session_state.processing_complete and st.session_state.result and not st.session_state.questions_complete:
    # Present all 3 labeling questions together in a form
    st.markdown("---")
    st.subheader("Data Labeling Questions")
    st.write("Please answer these questions to properly categorize and store the call data:")
    
    # Display all 3 questions in a single form
    with st.form("labeling_questions_form"):
        st.info("Please provide the following information for data organization:")
        
        # Question 1: Company Name
        st.markdown("**1. What is the company name discussed in this call/meeting?**")
        company_answer = st.text_input("Company Name:", placeholder="Enter the company name...")
        
        st.markdown("---")
        
        # Question 2: Attendees
        st.markdown("**2. Who were the attendees/participants in this meeting?**")
        attendees_answer = st.text_area("Attendees/Participants:", placeholder="List the attendees/participants...", height=100)
        
        st.markdown("---")
        
        # Question 3: Duration
        st.markdown("**3. What was the duration of this meeting (approximate time in minutes)?**")
        duration_answer = st.text_input("Duration (minutes):", placeholder="Enter duration in minutes...")
        
        st.markdown("---")
        
        # Submit all answers at once
        submitted = st.form_submit_button("Submit All Information", type="primary")
        
        if submitted:
            # Check if all questions have answers
            if not company_answer.strip() or not attendees_answer.strip() or not duration_answer.strip():
                st.error("Please answer all three questions to proceed.")
            else:
                # Process all the collected answers
                try:
                    # Store answers in the expected format
                    user_responses = [
                        {
                            'question': "What is the company name discussed in this call/meeting?",
                            'response': company_answer,
                            'timestamp': datetime.now().isoformat()
                        },
                        {
                            'question': "Who were the attendees/participants in this meeting?",
                            'response': attendees_answer,
                            'timestamp': datetime.now().isoformat()
                        },
                        {
                            'question': "What was the duration of this meeting (approximate time in minutes)?",
                            'response': duration_answer,
                            'timestamp': datetime.now().isoformat()
                        }
                    ]
                    
                    # Update the session in database
                    if st.session_state.current_session:
                        # Get the current session data
                        db_manager = DatabaseManager()
                        session_data = db_manager.get_session(st.session_state.current_session)
                        
                        if session_data:
                            # Update with user responses
                            session_data['user_responses'] = user_responses
                            session_data['questions_answered'] = True
                            
                            # Extract labels for enhanced storage
                            session_data.update({
                                "labels": {
                                    "company_name": company_answer,
                                    "attendees": attendees_answer,
                                    "duration": duration_answer
                                },
                                "company_name": company_answer,
                                "attendees": attendees_answer,
                                "meeting_duration": duration_answer,
                                "status": "completed"
                            })
                            
                            # Store updated data (this will now update instead of insert)
                            document_id = db_manager.store_session_data(session_data)
                            
                            # Mark as complete
                            st.session_state.questions_complete = True
                            
                            # Show success message
                            st.success("✅ All information collected and stored successfully!")
                            
                            # Display summary
                            with st.expander("View Stored Information", expanded=True):
                                st.markdown(f"**Company:** {company_answer}")
                                st.markdown(f"**Attendees:** {attendees_answer}")
                                st.markdown(f"**Duration:** {duration_answer} minutes")
                            
                            # Auto-trigger MoM generation workflow after successful storage
                            st.session_state.old_data_selected = session_data
                            st.session_state.mom_generation_prompt = True
                            
                            # Add message to chat asking about MoM generation
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Great! Your call data has been stored successfully. Would you like to process this meeting data to generate minutes?"
                            })
                            
                            st.rerun()
                        else:
                            st.error("Session data not found.")
                    else:
                        st.error("No active session found.")
                        
                except Exception as e:
                    st.error(f"Error storing information: {str(e)}")

# Cleanup on session end
def cleanup_temp_file():
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        os.remove(st.session_state.temp_file_path)

def handle_session_end():
    cleanup_temp_file()

# Register cleanup handler
st.session_state['_cleanup_handler'] = handle_session_end

def format_mom_content(agent_state):
    """Format the MoM content from the agent state - Updated to use Final MoM Agent output"""
    # Check if we have the final MoM content from the Final MoM Agent
    if agent_state.get("final_mom_content"):
        return agent_state["final_mom_content"]
    
    # Fallback to basic formatting if Final MoM Agent content is not available
    meeting_details = agent_state.get("meeting_details", {})
    qa_pairs = agent_state.get("answered_questions", [])
    
    # Format as Markdown
    mom = "# Minutes of Meeting\n\n"
    
    # Add meeting details
    mom += "## Meeting Details\n\n"
    for key, value in meeting_details.items():
        mom += f"- **{key}**: {value}\n"
    
    # Add discussion points
    mom += "\n## Discussion Points\n\n"
    for qa in qa_pairs:
        mom += f"### Q: {qa.get('question')}\n"
        mom += f"{qa.get('answer')}\n\n"
    
    # Add generated questions and answers
    mom += "\n## Additional Topics\n\n"
    for qa in agent_state.get("qa_manager_answered_questions", []):
        mom += f"### {qa.get('question')}\n"
        mom += f"{qa.get('answer')}\n\n"
    
    return mom





