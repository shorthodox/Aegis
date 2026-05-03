from config.settings import RISK_MODES

class CapitalManager:
    def __init__(self, mode):
        self.mode = mode.lower()
        self.settings = RISK_MODES.get(self.mode, RISK_MODES['safe'])

    def calculate_suggestion(self, total_balance, entry_price, sl_pct=0.02):
        """
        Calculate capital allocation and target based on risk mode.
        
        Args:
            total_balance (float): User's total trading capital
            entry_price (float): Current price of the asset
            sl_pct (float): Stop loss percentage (default 2%)
        
        Returns:
            dict: {'amount': investment_amount, 'percentage': pct_of_capital, 'target': target_profit}
        """
        max_allocation_pct = self.settings['max_allocation']
        
        # Calculate investment amount
        invest_amount = total_balance * max_allocation_pct
        
        # Calculate target profit based on risk mode
        target_profit_pct = self.settings['target_profit']
        # Assuming target_profit is like "1-2%" or "2-3%"
        # Take the average for simplicity
        if isinstance(target_profit_pct, str):
            parts = target_profit_pct.replace('%', '').split('-')
            avg_target = (float(parts[0]) + float(parts[1])) / 2 / 100
        else:
            avg_target = target_profit_pct / 100
        
        target_profit = invest_amount * avg_target
        
        return {
            'amount': round(invest_amount, 2),
            'percentage': round(max_allocation_pct * 100, 1),
            'target': round(target_profit, 2),
            'target_pct': round(avg_target * 100, 1)
        }