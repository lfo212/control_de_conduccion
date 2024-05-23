import React, { useState } from 'react';
import axios from 'axios';

const SettingsForm = () => {
  const [setting, setSetting] = useState('');

  const handleChange = (event) => {
    setSetting(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      await axios.post('/settings', { setting });
      alert('Setting saved successfully');
    } catch (error) {
      console.error('Error saving setting:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="mb-3">
        <label htmlFor="setting" className="form-label"><h5>Setting</h5></label>
        <input
          type="text"
          className="form-control"
          id="setting"
          value={setting}
          onChange={handleChange}
        />
      </div>
      <button type="submit" className="btn btn-primary">Save</button>
    </form>
  );
};

export default SettingsForm;
