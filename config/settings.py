import yaml
import os

# Load settings from YAML
settings_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')
try:
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    if not isinstance(settings, dict):
        settings = {}
except Exception as e:
    print(f"Error loading settings: {e}")
    settings = {}

# Extract configurations
SUBSCRIPTIONS = settings.get('subscriptions', {})
RISK_MODES = settings.get('risk_modes', {
    'safe': {'target_profit': '1-2%', 'max_allocation': 0.05},
    'moderate': {'target_profit': '2-3%', 'max_allocation': 0.15},
    'hrhp': {'target_profit': '4-5%', 'max_allocation': 0.30}
})