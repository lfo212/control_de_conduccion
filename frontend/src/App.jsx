import React from 'react';
import './App.css';
import CommandLauncher from './components/CommandLauncher';
import ConfigForm from './components/ConfigForm';
import VideoPlayer from './components/VideoPlayer';
import MediaList from './components/MediaList';
import RegisterDriver from './components/RegisterDriver'
import SelectDriver from './components/SelectDriver';
import DeleteDriver from './components/DeleteDriver';

const App = () => {
  const wsUrl1 = "ws://localhost:8765"; // WebSocket URL for the first stream
  const wsUrl2 = "ws://localhost:8766"; // WebSocket URL for the second stream (if needed)
  const videoPlayer1 = "Camara Frontal";
  const videoPlayer2 = "Camara Lateral";
  const offline_image1 = "/front.png";
  const offline_image2 = "/side.png";

  return (
    <div className="container mt-5">
      <h1>Control de Manejo</h1>
      <div className="row">
        <div className="col-md-6">
          <SelectDriver/>
          <ConfigForm />
          <CommandLauncher />
          <RegisterDriver />
          <DeleteDriver />
        </div>
        <div className="col-md-6">
          <VideoPlayer wsUrl={wsUrl1} name={videoPlayer1} offline_image={offline_image1} />
          <VideoPlayer wsUrl={wsUrl2} name={videoPlayer2} offline_image={offline_image2}/>
          <MediaList />
        </div>
      </div>
      
    </div>
  );
};

export default App;
