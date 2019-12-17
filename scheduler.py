from datetime import datetime
from threading import Timer
from earthquake_twitter import query_earthquake

today = datetime.today()
tomorrow = today.replace(day=today.day+1, hour=1, minute=0, second=0, microsecond=0)
delta_t = tomorrow - today

secs = delta_t.seconds + 1

t = Timer(secs, query_earthquake)
t.start()
