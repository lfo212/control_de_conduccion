import React, { useEffect, useRef, useState } from 'react';

const VideoPlayer = ({ wsUrl, name, offline_image }) => {
  const videoRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 480, height: 360 });

  useEffect(() => {
    let ws;
    const connectWebSocket = () => {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connection opened');
        setConnected(true);
      };

      ws.onmessage = (event) => {
        const img = new Image();
        img.src = 'data:image/jpeg;base64,' + event.data;
        img.onload = () => {
          const aspectRatio = img.width / img.height;
          const newWidth = 480; // Desired width
          const newHeight = newWidth / aspectRatio;
          setCanvasDimensions({ width: newWidth, height: newHeight });

          if (videoRef.current) {
            videoRef.current.width = newWidth;
            videoRef.current.height = newHeight;
            videoRef.current.getContext('2d').drawImage(img, 0, 0, newWidth, newHeight);
          }
        };
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        setConnected(false);
        // Reconnect after 5 seconds if the connection is closed
        setTimeout(connectWebSocket, 5000);
      };
    };

    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
        console.log('WebSocket connection closed on cleanup');
      }
    };
  }, [wsUrl]);

  return (
    <div>
      <h2>{name} {connected ? '(Encendida)' : '(Apagada)'}</h2>
      {connected ? (
        <canvas ref={videoRef} width={canvasDimensions.width} height={canvasDimensions.height}></canvas>
      ) : (
        <img src={offline_image} className="transparent-image" alt="Transparent" width="480" height="360" />
      )}
    </div>
  );
};

export default VideoPlayer;
