import React from 'react';
import UploadImage from './components/UploadImage';
import UserList from './components/UserList';
import SettingsForm from './components/SettingsForm';
import CommandLauncher from './components/CommandLauncher';
import ConfigForm from './components/ConfigForm';
import VideoPlayer from './components/VideoPlayer';

const App = () => {
  const wsUrl1 = "ws://localhost:8765"; // WebSocket URL for the first stream
  const wsUrl2 = "ws://localhost:8766"; // WebSocket URL for the second stream (if needed)

  return (
    <div className="container mt-5">
      <h1>Control de manejo</h1>
      <ConfigForm />
      <CommandLauncher />
      <VideoPlayer wsUrl={wsUrl1} />
      <VideoPlayer wsUrl={wsUrl2} />
      <UploadImage />
      <SettingsForm />
      <UserList />
    </div>
  );
};

export default App;
