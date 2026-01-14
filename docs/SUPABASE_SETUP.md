# 🔌 Supabase Integration Guide

## 📋 Prerequisites

1. **Supabase Account**: Create free account at [https://supabase.com](https://supabase.com)
2. **Database Setup**: Your DE team (Aman) should provide:
   - Supabase URL
   - Service Role Key (or anon key)
   - Database table schema

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Supabase Client

```bash
pip install supabase
```

Or add to your requirements.txt:
```
supabase>=2.0.0
```

### Step 2: Configure Environment Variables

Create a `.env` file in your project root:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Optional: Table name if different from default
SUPABASE_TABLE=stock_data
```

**Get these from Supabase Dashboard:**
- Project URL: Settings → API → Project URL
- Service Key: Settings → API → Project API keys → service_role

### Step 3: Update API with Real Supabase Queries

I've created a ready-to-use template: `signals/supabase_integration.py`

## 📊 Expected Database Schema

Your Supabase table should have columns like:

```sql
CREATE TABLE stock_data (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    rsi DECIMAL(5, 2),
    signal TEXT,
    prediction DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔧 Implementation

### Option A: Use the Template (Recommended)

Run this to integrate:
```bash
python signals/supabase_integration.py
```

### Option B: Manual Integration

1. Import Supabase in `signals/api.py`
2. Initialize client with your credentials
3. Replace placeholder returns with real queries

## 🧪 Testing Connection

```python
from supabase import create_client

url = "YOUR_SUPABASE_URL"
key = "YOUR_SUPABASE_KEY"
supabase = create_client(url, key)

# Test query
result = supabase.table("stock_data").select("*").limit(5).execute()
print(f"✅ Connected! Found {len(result.data)} records")
```

## 🔒 Security Best Practices

1. **Never commit** `.env` file to git
2. Add to `.gitignore`:
   ```
   .env
   *.env
   ```
3. Use service_role key for server-side API
4. Use anon key for client-side applications

## 📞 Need Help?

Contact your DE team (Aman) for:
- Database credentials
- Table schema confirmation
- Column name mappings
- Query optimization

## 🎯 Next Steps

After setup:
1. Test connection: `python test_new_api_endpoints.py`
2. Run dashboard: `streamlit run 0_Overview.py`
3. Verify real data appears instead of placeholder messages
