// VideoPlayer.jsx
import React, { useEffect, useRef, useState } from 'react';

const VideoPlayer = ({ wsUrl }) => {
  const videoRef = useRef(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const img = new Image();
      img.src = 'data:image/jpeg;base64,' + event.data;
      img.onload = () => {
        if (videoRef.current) {
          videoRef.current.getContext('2d').drawImage(img, 0, 0);
        }
      };
    };

    ws.onclose = () => {
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [wsUrl]);

  return (
    <div>
      <h2>Video Output {connected ? '(Connected)' : '(Disconnected)'}</h2>
      <canvas ref={videoRef} width="640" height="480"></canvas>
    </div>
  );
};

export default VideoPlayer;
