class CapitalManager:
    def __init__(self, mode):
        self.modes = {
            'safe': {'risk': 0.01, 'target': '1-2%'},
            'moderate': {'risk': 0.02, 'target': '2-3%'},
            'hrhp': {'risk': 0.05, 'target': '4-5%'}
        }
        self.current_settings = self.modes.get(mode, self.modes['safe'])

    def calculate_suggestion(self, capital, entry_price, stop_loss_price):
        risk_amount = capital * self.current_settings['risk']
        sl_pct = abs(entry_price - stop_loss_price) / entry_price
        
        # Suggested amount to put into the trade
        suggested_qty = risk_amount / sl_pct
        
        return {
            "amount": round(suggested_qty, 2),
            "percentage": f"{(suggested_qty/capital)*100:.1f}%",
            "target": self.current_settings['target']
        }