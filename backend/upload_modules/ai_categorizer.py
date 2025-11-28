"""
AI Categorizer Module
=====================
Handles AI/ML categorization of transactions.
"""

import pandas as pd
from typing import Tuple


def categorize_transactions(df: pd.DataFrame, use_cache: bool = False) -> Tuple[pd.DataFrame, int]:
    """
    Categorize transactions using AI/ML.
    
    Args:
        df: DataFrame with transactions
        use_cache: Whether to skip AI processing (using cached data)
        
    Returns:
        Tuple of (categorized_dataframe, categorized_count)
    """
    if use_cache:
        print(f"⏭️ SKIPPING AI/ML PROCESSING: Using cached categorized data from database")
        # Ensure Category column exists (should already be there from cache)
        if 'Category' not in df.columns and 'ai_category' in df.columns:
            df['Category'] = df['ai_category']
    else:
        print(f"🤖 ML PROCESSING: Using 100% AI/ML approach...")
        print("🤖 Applying AI/ML categorization to all transactions...")
        print("🔄 Running fresh AI categorization for all transactions...")
        
        # Import and use universal categorization from dedicated module
        from .universal_categorization import universal_categorize_any_dataset
        df = universal_categorize_any_dataset(df)
    
    # Verify categorization was applied
    if 'Category' in df.columns:
        ai_categorized = sum(1 for cat in df['Category'] if cat and str(cat).strip() != '')
        print(f"✅ AI categorization applied: {ai_categorized}/{len(df)} transactions categorized with AI")
        return df, ai_categorized
    else:
        print("⚠️ Warning: Category column not found after categorization")
        return df, 0

