module.exports = {
  apps: [
    {
      name: 'aegis',
      script: 'D:\\Content\\Animesh\\bots\\ai_signal_bot\\.venv\\Scripts\\python.exe',
      args: 'main.py',
      cwd: 'D:\\Content\\Animesh\\bots\\ai_signal_bot',
      interpreter: 'none',
      windowsHide: true,
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PORT: '8000',
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      out_file: 'logs/pm2_out.log',
      error_file: 'logs/pm2_err.log',
      merge_logs: true
    }
  ]
};
