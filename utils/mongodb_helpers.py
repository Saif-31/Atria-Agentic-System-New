"""
MongoDB helper functions for data sanitization and document preparation
"""
from typing import Any, Dict, List, Union
from datetime import datetime
import json

def sanitize_for_mongodb(value: Any) -> Any:
    """
    Sanitize a value to be MongoDB-compatible
    
    Args:
        value: Any value that needs to be stored in MongoDB
        
    Returns:
        MongoDB-compatible value
    """
    if value is None:
        return None
    
    # Handle datetime objects
    if isinstance(value, datetime):
        return value.isoformat()
    
    # Handle dictionaries recursively
    if isinstance(value, dict):
        return {k: sanitize_for_mongodb(v) for k, v in value.items()}
    
    # Handle lists recursively
    if isinstance(value, list):
        return [sanitize_for_mongodb(item) for item in value]
    
    # Handle file objects or other non-serializable objects
    if hasattr(value, 'read') or not _is_json_serializable(value):
        return str(value)
    
    return value

def _is_json_serializable(obj: Any) -> bool:
    """Check if an object is JSON serializable"""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False

def prepare_document_for_mongodb(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a document for MongoDB storage by sanitizing all fields
    
    Args:
        document: Dictionary representing the document to store
        
    Returns:
        Sanitized document ready for MongoDB storage
    """
    if not isinstance(document, dict):
        raise ValueError("Document must be a dictionary")
    
    # Create a copy to avoid modifying the original
    sanitized_doc = {}
    
    for key, value in document.items():
        # Ensure key is a string and doesn't start with '$' or contain '.'
        sanitized_key = str(key).replace('.', '_').replace('$', '_')
        sanitized_doc[sanitized_key] = sanitize_for_mongodb(value)
    
    return sanitized_doc

def handle_mongodb_errors(func):
    """Decorator to handle common MongoDB errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"MongoDB operation failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    return wrapper
