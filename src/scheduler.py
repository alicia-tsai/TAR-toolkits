from datetime import datetime
from threading import Timer
from src.earthquake_data import get_earthquake_data

today = datetime.today()
tomorrow = today.replace(day=today.day+1, hour=1, minute=0, second=0, microsecond=0)
delta_t = tomorrow - today

secs = delta_t.seconds + 1

t = Timer(secs, get_earthquake_data)
t.start()
