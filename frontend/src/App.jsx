import React from 'react';
import UploadImage from './components/UploadImage';
import UserList from './components/UserList';
import SettingsForm from './components/SettingsForm';
import CommandLauncher from './components/CommandLauncher';
import ConfigForm from './components/ConfigForm';

const App = () => {
  return (
    <div className="container mt-5">
      <h1>Control de manejo</h1>
      <ConfigForm />
      <CommandLauncher />
      <UploadImage />
      <SettingsForm />
      <UserList />
    </div>
  );
};

export default App;
