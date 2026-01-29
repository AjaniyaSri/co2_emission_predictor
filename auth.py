from supabase import create_client
import os

# ⚠️ Replace with your own Supabase details
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

def supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user(sb):
    try:
        user_resp = sb.auth.get_user()
        return getattr(user_resp, "user", None)
    except Exception:
        return None
