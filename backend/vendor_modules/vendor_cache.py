from typing import Optional, Dict, Any


def get_cached_vendor_data(db_manager, file_id: int) -> Optional[Dict[str, Any]]:
    """
    Check vendor_entities for the given file_id and return structured summaries
    plus transaction links if available.
    """
    if not db_manager or not file_id:
        return None

    try:
        summaries = db_manager.fetch_vendor_entities_by_file(file_id)
        if not summaries:
            return None

        transactions_by_vendor = db_manager.fetch_vendor_transactions_by_file(file_id)
        return {
            "summaries": summaries,
            "transactions": transactions_by_vendor,
        }
    except Exception as e:
        # Handle database connection errors gracefully
        print(f"⚠️ Error fetching cached vendor data: {e}")
        return None

