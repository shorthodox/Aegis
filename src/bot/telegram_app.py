import logging
import random
from typing import Optional, Tuple, Dict, Any
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ---------------------- MOCK DEPENDENCIES ----------------------

class DBHandler:
    """Mock database handler – stores user data in memory."""
    def __init__(self):
        self.users: Dict[int, dict] = {}

    def add_user(self, user_id: int) -> None:
        if user_id not in self.users:
            self.users[user_id] = {
                "tier": "pro",          # Defaulting to pro for your testing
                "trial_end": None,
                "sub_end": None,
                "mode": "moderate",     
                "balance": 1000.0
            }

    def get_user(self, user_id: int) -> Optional[Tuple]:
        """Returns: (id, tier, trial_end, sub_end, mode, balance)"""
        user = self.users.get(user_id)
        if not user:
            return None
        return (user_id, user["tier"], user["trial_end"], user["sub_end"], user["mode"], user["balance"])

    def update_mode(self, user_id: int, mode: str) -> None:
        if user_id in self.users:
            self.users[user_id]["mode"] = mode

    def update_balance(self, user_id: int, balance: float) -> None:
        if user_id in self.users:
            self.users[user_id]["balance"] = balance

    def is_subscription_active(self, user_id: int) -> bool:
        """Mock check for active subscription or trial.

        In this mock every created user is considered active. Replace with
        real datetime checks against `trial_end`/`sub_end` in production.
        """
        return user_id in self.users


class Predictor:
    """Mock ML predictor – returns random confidence for testing."""
    def get_probability(self, symbol: str, timeframe: str) -> float:
        return random.uniform(0.5, 0.95)


class CapitalManager:
    """Calculates investment amount based on risk mode."""
    def __init__(self, mode: str):
        self.mode = mode

    def calculate_suggestion(self, balance: float, entry_price: float, sl_pct: float) -> Dict[str, Any]:
        risk_mult = {"safe": 0.02, "moderate": 0.05, "hrhp": 0.10}.get(self.mode, 0.05)
        amount = balance * risk_mult
        percentage = risk_mult * 100
        target_pct = risk_mult * 2 
        target = amount * (1 + target_pct)
        return {
            "amount": amount,
            "percentage": percentage,
            "target": target,
            "target_pct": target_pct * 100
        }


# ------------------------------ TELEGRAM BOT ---------------------------------

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class SignalBot:
    def __init__(self, token: str):
        self.app = ApplicationBuilder().token(token).build()
        self.db_handler = DBHandler()
        self.predictor = Predictor()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # TYPE GUARD: Ensure user and message exist
        if not update.effective_user or not update.message:
            return
            
        user_id = update.effective_user.id
        self.db_handler.add_user(user_id)

        welcome_text = (
            "🚀 **AI Crypto Signal Bot v1.0**\n\n"
            "Your 3-day PRO trial is active. I use XGBoost ML to predict "
            "BTC, ETH, BNB, and SOL movements.\n\n"
            "**Commands:**\n"
            "📈 /chart5min - 5m AI Signal (Pro)\n"
            "📊 /chart1h - 1h AI Signal (Basic/Pro)\n"
            "⚙️ /mode <safe|moderate|hrhp> - Set Risk\n"
            "🐋 /whale - Smart Money Tracker (Pro)\n"
            "💰 /balance <amount> - Set your trading capital"
        )
        await update.message.reply_text(welcome_text, parse_mode="Markdown")

    async def get_chart_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /chart5min – 5-minute AI signal (PRO only)."""
        try:
            if not update.effective_user or not update.message:
                return

            user_id = update.effective_user.id

            # Subscription check
            if not self.db_handler.is_subscription_active(user_id):
                await update.message.reply_text("❌ Your access has expired. Please use /upgrade to continue.")
                return

            user_data = self.db_handler.get_user(user_id)
            if not user_data:
                await update.message.reply_text("Please run /start first.")
                return

            _, tier, _, _, mode, balance = user_data
            balance = balance if balance else 1000.0

            if isinstance(tier, str) and tier.lower() == 'basic':
                await update.message.reply_text("❌ 5m signals are for PRO users. Try /chart1h or upgrade!")
                return

            prob = self.predictor.get_probability(symbol="BTC/USDT", timeframe="5m")
            cap_manager = CapitalManager(mode)
            suggestion = cap_manager.calculate_suggestion(balance, entry_price=65000, sl_pct=0.02)

            signal_type = "🟢 STRONG BUY" if prob > 0.85 else "🟡 WEAK BUY" if prob > 0.70 else "⚪️ NEUTRAL / WAIT"

            response = (
                f"🎯 **Signal: BTC/USDT (5m)**\n"
                f"Type: {signal_type}\n"
                f"Confidence: `{prob*100:.1f}%`\n"
                f"Risk Mode: `{mode.upper()}`\n\n"
                f"💡 **Capital Suggestion:**\n"
                f"Invest: `${suggestion['amount']:.2f}` ({suggestion['percentage']:.1f}% of capital)\n"
                f"Target Profit: `${suggestion['target']:.2f}` ({suggestion['target_pct']:.1f}%)"
            )
            await update.message.reply_text(response, parse_mode="Markdown")

        except Exception:
            logging.exception("Error in get_chart_signal")
            if update.message:
                await update.message.reply_text("❌ An error occurred. Please try again later.")

    async def get_chart1h_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            if not update.effective_user or not update.message:
                return

            user_id = update.effective_user.id
            user_data = self.db_handler.get_user(user_id)
            if not user_data:
                await update.message.reply_text("Please run /start first.")
                return

            _, _, _, _, mode, balance = user_data
            balance = balance if balance else 1000.0
            
            prob = self.predictor.get_probability(symbol="BTC/USDT", timeframe="1h")
            cap_manager = CapitalManager(mode)
            suggestion = cap_manager.calculate_suggestion(balance, entry_price=65000, sl_pct=0.02)

            signal_type = "🟢 STRONG BUY" if prob > 0.85 else "🟡 WEAK BUY" if prob > 0.70 else "⚪️ NEUTRAL / WAIT"
            response = (
                f"🎯 **Signal: BTC/USDT (1h)**\n"
                f"Type: {signal_type}\n"
                f"Confidence: `{prob*100:.1f}%`\n"
                f"Risk Mode: `{mode.upper()}`\n\n"
                f"💡 **Capital Suggestion:**\n"
                f"Invest: `${suggestion['amount']:.2f}` ({suggestion['percentage']:.1f}% of capital)\n"
                f"Target Profit: `${suggestion['target']:.2f}` ({suggestion['target_pct']:.1f}%)"
            )
            await update.message.reply_text(response, parse_mode="Markdown")

        except Exception as e:
            logging.error(f"Error in get_chart1h_signal: {e}")
            if update.message:
                await update.message.reply_text("❌ An error occurred. Please try again later.")

    async def change_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            if not update.effective_user or not update.message:
                return
            
            user_id = update.effective_user.id
            if not context.args:
                await update.message.reply_text("Usage: /mode <safe|moderate|hrhp>")
                return
            
            new_mode = context.args[0].lower()
            if new_mode in ['safe', 'moderate', 'hrhp']:
                self.db_handler.update_mode(user_id, new_mode)
                await update.message.reply_text(f"✅ Risk mode updated to: **{new_mode.upper()}**", parse_mode="Markdown")
            else:
                await update.message.reply_text("Invalid mode! Use safe, moderate, or hrhp.")
        except Exception as e:
            logging.error(f"Error in change_mode: {e}")

    async def get_whale_data(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            if not update.effective_user or not update.message:
                return

            user_id = update.effective_user.id
            user_data = self.db_handler.get_user(user_id)
            
            if not user_data or user_data[1].lower() != 'pro':
                await update.message.reply_text("🐋 Whale Tracking is a **PRO** feature.")
                return

            whale_msg = (
                "🐋 **Whale Alert (Last 60m)**\n"
                "• $45M USDT Inflow to Binance (Bullish)\n"
                "• 1,200 BTC moved to Cold Wallet (Supply Crunch)\n"
                "**Sentiment:** Overwhelmingly Bullish 🚀"
            )
            await update.message.reply_text(whale_msg, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"Error in get_whale_data: {e}")

    async def set_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            if not update.effective_user or not update.message:
                return

            user_id = update.effective_user.id
            if not context.args or len(context.args) != 1:
                await update.message.reply_text("Usage: /balance <amount>\nExample: /balance 5000")
                return
            
            amount = float(context.args[0])
            self.db_handler.update_balance(user_id, amount)
            await update.message.reply_text(f"✅ Trading capital set to: **${amount:.2f}**", parse_mode="Markdown")
        except ValueError:
            if update.message:
                await update.message.reply_text("Invalid amount. Please enter a number.")
        except Exception as e:
            logging.exception("Error in set_balance")
            if update.message:
                await update.message.reply_text("❌ Failed to update balance. Please try again later.")

    def run(self) -> None:
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("chart5min", self.get_chart_signal))
        self.app.add_handler(CommandHandler("chart1h", self.get_chart1h_signal))
        self.app.add_handler(CommandHandler("mode", self.change_mode))
        self.app.add_handler(CommandHandler("whale", self.get_whale_data))
        self.app.add_handler(CommandHandler("balance", self.set_balance))
        
        print("Bot is live... Listening for signals.")
        self.app.run_polling()



if __name__ == "__main__":
    # Note: Ensure you use Python 3.12 or 3.13 and install python-telegram-bot
    TOKEN = "YOUR_BOT_TOKEN"   
    bot = SignalBot(TOKEN)
    bot.run()