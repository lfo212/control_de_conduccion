FROM orgoro/dlib-opencv-python

# Set the display environment variable
ENV DISPLAY=:0

RUN apt update -y && \
    apt install -y \
    cmake \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ffmpeg

COPY requirements.txt ./
RUN pip3.8 install --no-cache-dir -r requirements.txt
RUN mkdir eventos
COPY . .

# Make port 8765 and 8766 available to the world outside this container
EXPOSE 8765
EXPOSE 8766

CMD [ "python3.8", "./src/main.py" ]
