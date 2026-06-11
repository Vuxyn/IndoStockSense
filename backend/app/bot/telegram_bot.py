import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from app.core.inference import predict_sentiment

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def build_bot_app():
    if not BOT_TOKEN or BOT_TOKEN == "your_telegram_bot_token_here":
        print("WARNING: TELEGRAM_BOT_TOKEN not set. Bot will not run.")
        return None
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    return app

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Halo! Saya IndoStockSense Bot. Kirimkan berita saham apa saja, dan saya akan memprediksi sentimennya!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Kirim teks berita pasar modal, contoh:\n'Saham BBCA kembali cetak rekor All Time High hari ini.'")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    msg = await update.message.reply_text("⏳ Menganalisa sentimen...")
    
    result = predict_sentiment(text)
    
    signal_emoji = "🟢" if result["sentiment"] == "positif" else "🔴" if result["sentiment"] == "negatif" else "⚪"
    
    response = (
        f"{signal_emoji} **Prediksi Sentimen**\n\n"
        f"Teks: {text}\n"
        f"Sentimen: **{result['sentiment'].upper()}**\n"
        f"Confidence: {result['confidence']:.2%}"
    )
    
    await msg.edit_text(response, parse_mode="Markdown")

async def broadcast_news(bot_app, chat_id, news_data):
    if not bot_app: return
    
    message = "📈 **IndoStockSense Daily Briefing** 📉\n\n"
    for item in news_data:
        sentiment_emoji = "🟢" if item["sentiment"] == "positif" else "🔴" if item["sentiment"] == "negatif" else "⚪"
        message += f"{sentiment_emoji} [{item['sentiment'].upper()}] {item['title']}\n\n"
        
    try:
        await bot_app.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        print(f"Broadcast sent to {chat_id}")
    except Exception as e:
        print(f"Failed to broadcast to {chat_id}: {e}")
