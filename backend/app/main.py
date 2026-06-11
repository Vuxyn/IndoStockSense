import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

from app.core.scraper import scrape_stock_news
from app.core.inference import predict_sentiment, load_model
from app.bot.telegram_bot import build_bot_app, broadcast_news

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "your_supabase_url_here" else None

scheduler = AsyncIOScheduler()
bot_app = None

async def cron_scrape_and_broadcast():
    print("[Cron] Scraping news...")
    news = scrape_stock_news()
    
    analyzed_news = []
    for item in news:
        pred = predict_sentiment(item["title"])
        item["sentiment"] = pred["sentiment"]
        item["confidence"] = pred["confidence"]
        analyzed_news.append(item)
        
        if supabase:
            try:
                supabase.table("news_sentiment").insert(item).execute()
            except Exception as e:
                print(f"[Supabase] Insert error: {e}")
                
    chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
    if bot_app and chat_id and chat_id != "your_chat_id_here":
        print("[Cron] Broadcasting to Telegram...")
        await broadcast_news(bot_app, chat_id, analyzed_news)

async def cron_ping_supabase():
    print("[Cron] Pinging Supabase to prevent pause...")
    if supabase:
        try:
            supabase.table("news_sentiment").select("id").limit(1).execute()
            print("[Cron] Supabase Ping Success.")
        except Exception as e:
            print(f"[Cron] Supabase Ping Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up FastAPI application...")
    load_model()
    
    global bot_app
    bot_app = build_bot_app()
    if bot_app:
        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling()
        print("Telegram Bot started.")
        
    scheduler.add_job(cron_scrape_and_broadcast, 'cron', hour=8, minute=0)
    scheduler.add_job(cron_ping_supabase, 'interval', hours=24)
    scheduler.start()
    print("Scheduler started.")
    
    yield
    
    print("Shutting down FastAPI application...")
    if bot_app:
        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()
    scheduler.shutdown()

app = FastAPI(title="IndoStockSense API", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "message": "IndoStockSense Backend Running"}

@app.get("/api/news")
def get_recent_news():
    if supabase:
        try:
            res = supabase.table("news_sentiment").select("*").order("created_at", desc=True).limit(20).execute()
            return {"data": res.data}
        except Exception as e:
            return {"error": str(e)}
            
    news = scrape_stock_news()
    for item in news:
        pred = predict_sentiment(item["title"])
        item.update(pred)
    return {"data": news}

@app.post("/api/predict")
def predict_endpoint(payload: dict):
    text = payload.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    result = predict_sentiment(text)
    return {"data": result}
