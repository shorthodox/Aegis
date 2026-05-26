module.exports = {
  apps: [
    {
      name: 'aegis',
      script: 'python',
      args: 'main.py',
      cwd: 'D:\\Content\\Animesh\\bots\\ai_signal_bot',
      interpreter: 'none',
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
