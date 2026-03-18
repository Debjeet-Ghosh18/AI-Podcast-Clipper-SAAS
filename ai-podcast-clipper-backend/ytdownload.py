from pytubefix import YouTube
from pytubefix.cli import on_progress

url1 = "https://youtu.be/SOG0GmKts_I?si=otCbmYbLK7kuONUg"
url2 = "https://youtu.be/-vMgbJ6WqN4?si=WkQv1ivTJ6pn1kvP"

yt = YouTube(url1, on_progress_callback=on_progress)
print(yt.title)

ys = yt.streams.get_highest_resolution()
ys.download()
