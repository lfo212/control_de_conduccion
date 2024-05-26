import React from 'react';
import './App.css';
import UploadImage from './components/UploadImage';
import UserList from './components/UserList';
import SettingsForm from './components/SettingsForm';
import CommandLauncher from './components/CommandLauncher';
import ConfigForm from './components/ConfigForm';
import VideoPlayer from './components/VideoPlayer';

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
          <ConfigForm />
          <CommandLauncher />
          <UploadImage />
          <SettingsForm />
          <UserList />
        </div>
        <div className="col-md-6">
          <VideoPlayer wsUrl={wsUrl1} name={videoPlayer1} offline_image={offline_image1} />
          <VideoPlayer wsUrl={wsUrl2} name={videoPlayer2} offline_image={offline_image2}/>
        </div>
      </div>
      
    </div>
  );
};

export default App;
