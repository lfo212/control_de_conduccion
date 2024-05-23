import React, { useState, useEffect } from 'react';
import axios from 'axios';

const editableConfigKeys = [
  'front_video_input',
  'side_video_input',
  'confidence_threshold',
  'head_grades_threshold',
];

const configKeyLabels = {
  front_video_input: 'Camara frontal',
  side_video_input: 'Camara lateral',
  confidence_threshold: 'Umbral de confiaza',
  head_grades_threshold: 'Limite de movimiento cabeza',
};

const ConfigSettings = () => {
    const [config, setConfig] = useState({});
    const [newConfig, setNewConfig] = useState({});
    const [videoFiles, setVideoFiles] = useState([]);
    const [cameraDevices, setCameraDevices] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
  
    useEffect(() => {
      const fetchConfig = async () => {
        try {
          const response = await axios.get('/get_config');
          setConfig(response.data.config);
          setVideoFiles(response.data.video_files);
          setCameraDevices(response.data.camera_devices);
          setIsLoading(false);
        } catch (error) {
          console.error('Error fetching config:', error);
        }
      };
      fetchConfig();
    }, []);
  
    const handleChange = (e) => {
      const { name, value, type, checked } = e.target;
      const newValue = type === 'checkbox' ? checked : value;
      setNewConfig((prevConfig) => ({
        ...prevConfig,
        [name]: newValue,
      }));
    };
  
    const handleSave = async () => {
      try {
        await axios.post('/save_config', newConfig);
        alert('Configuracion guardada exitosamente.');
      } catch (error) {
        console.error('Error guardando configuracion:', error);
      }
    };
  
    if (isLoading) {
      return <div>Cargando...</div>;
    }
  
    return (
      <div>
        <h2>Configuracion</h2>
        <div>
          {Object.entries(config).map(([key, value]) => (
            editableConfigKeys.includes(key) && (
              <div key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <label style={{ marginRight: '10px', minWidth: '200px' }}>
                <p>{configKeyLabels[key] || key}:</p>
                </label>
                {key === 'front_video_input' || key === 'side_video_input' ? (
                  <select
                    name={key}
                    value={newConfig[key] !== undefined ? newConfig[key] : value}
                    onChange={handleChange}
                    style={{ flex: '1' }}
                  >
                    <option value="">Seleccione una opcion</option>
                    {cameraDevices.map((device) => (
                      <option key={device.index} value={device.index}>
                        {device.name}
                      </option>
                    ))}
                    {videoFiles.map((file) => (
                      <option key={file} value={file}>
                        {file}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    style={{ marginRight: '10px', marginLeft: '10px', flex: '1' }}
                    type={typeof value === 'boolean' ? 'checkbox' : 'text'}
                    name={key}
                    value={newConfig[key] !== undefined ? newConfig[key] : value}
                    checked={typeof value === 'boolean' ? newConfig[key] !== undefined ? newConfig[key] : value : undefined}
                    onChange={handleChange}
                  />
                )}
              </div>
            )
          ))}
        </div>
        <div>
          <button onClick={handleSave}>Save</button>
        </div>
      </div>
    );
  };
  
  export default ConfigSettings;

