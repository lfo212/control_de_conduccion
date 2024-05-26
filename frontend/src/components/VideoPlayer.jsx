// VideoPlayer.jsx
import React, { useEffect, useRef, useState } from 'react';

const VideoPlayer = ({ wsUrl, name, offline_image }) => {
  const videoRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const reconnectIntervalRef = useRef(null);

  const connectWebSocket = () => {
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnected(true);
      if (reconnectIntervalRef.current) {
        clearInterval(reconnectIntervalRef.current);
        reconnectIntervalRef.current = null;
      }
    };

    ws.onmessage = (event) => {
      const img = new Image();
      img.src = 'data:image/jpeg;base64,' + event.data;
      img.onload = () => {
        if (videoRef.current) {
          videoRef.current.getContext('2d').drawImage(img, 0, 0, videoRef.current.width, videoRef.current.height);
        }
      };
    };

    ws.onclose = () => {
      setConnected(false);
      if (!reconnectIntervalRef.current) {
        reconnectIntervalRef.current = setInterval(() => {
          connectWebSocket();
        }, 5000);
      }
    };

    ws.onerror = () => {
      ws.close(); // Close WebSocket on error to trigger onclose
    };
  };

  useEffect(() => {
    connectWebSocket();

    // Clean up function
    return () => {
      if (reconnectIntervalRef.current) {
        clearInterval(reconnectIntervalRef.current);
      }
    };
  }, [wsUrl]);

  return (
    
    <div>
      <h2>{name} {connected ? '(Encendida)' : '(Apagada)'}</h2>
      {connected ? (
        <canvas ref={videoRef} width="480" height="360"></canvas>
      ) : (
        <img src={offline_image} className="transparent-image" alt="Transparent" width="480" height="360" />
      )}
    </div>
  );
};

export default VideoPlayer;
