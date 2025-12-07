from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = 'runs/detect/predict/vehicles1280x720.avi'

# Compressed video path
compressed_path = "result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f"""
<video width="640" height="480" controls>
  <source src="{data_url}" type="video/mp4">
</video>
""")