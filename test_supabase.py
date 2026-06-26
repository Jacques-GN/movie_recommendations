from dotenv import load_dotenv
from supabase import create_client
import os

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

res1  = supabase.table("movies").select("*").limit(5).execute()
res2 = supabase.table("ratings").select("*").limit(5).execute()
res3 = supabase.table("tags").select("*").limit(5).execute()

print("STATUS:", getattr(res1, "status_code", None))
print("STATUS:", getattr(res2, "status_code", None))
print("STATUS:", getattr(res3, "status_code", None))
print("DATA:", res1.data)
print("DATA:", res3.data)
print("DATA:", res2.data)
print("ERROR:", getattr(res1, "error", None))
print("ERROR:", getattr(res2, "error", None))
print("ERROR:", getattr(res3    , "error", None))
print("RES:", res1)
print("RES:", res2)
print("RES:", res3)