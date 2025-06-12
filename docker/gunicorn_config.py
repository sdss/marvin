import os

socket_dir = os.getenv("MARVIN_SOCKET_DIR", '/tmp/marvin')
bind = [f"unix:{socket_dir}/marvin.sock", "0.0.0.0:8000"]
workers = 1
worker_class = "gthread"
threads = 8
daemon = False
errorlog = os.path.join(os.getenv("MARVIN_LOGS_DIR", '/tmp/marvin/logs'), 'marvin_app_error.log')
accesslog = os.path.join(os.getenv("MARVIN_LOGS_DIR", '/tmp/marvin/logs'), 'marvin_app_access.log')
