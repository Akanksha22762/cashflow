from typing import Optional, Tuple


def resolve_session_ids(session, state_manager=None, db_manager=None) -> Tuple[Optional[int], Optional[int]]:
    """
    Ensure mysql_session_id/mysql_file_id are available either from Flask session
    the persistent state manager, or by looking up the latest session in MySQL.
    Returns (session_id, file_id) and updates the Flask session dict when new
    values are discovered.
    """
    
    session_id = session.get('mysql_session_id') if session else None
    file_id = session.get('mysql_file_id') if session else None

    if state_manager:
        if not session_id and getattr(state_manager, 'current_session_id', None):
            session_id = state_manager.current_session_id
            if session is not None:
                session['mysql_session_id'] = session_id
        if not file_id and getattr(state_manager, 'current_file_id', None):
            file_id = state_manager.current_file_id
            if session is not None:
                session['mysql_file_id'] = file_id

    if (db_manager is not None) and (session_id is None or file_id is None):
        try:
            latest_state = db_manager.get_latest_session_state()
            if latest_state:
                if session_id is None and latest_state.get('session_id'):
                    session_id = latest_state['session_id']
                    if session is not None:
                        session['mysql_session_id'] = session_id
                if file_id is None and latest_state.get('file_id'):
                    file_id = latest_state['file_id']
                    if session is not None:
                        session['mysql_file_id'] = file_id
        except Exception as e:
            print(f"⚠️ Could not resolve session IDs from database: {e}")

    return session_id, file_id

