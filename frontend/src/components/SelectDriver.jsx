import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SelectDriver = () => {
  const [drivers, setDrivers] = useState([]);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [driverPhoto, setDriverPhoto] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDrivers = async () => {
      try {
        const response = await axios.get('https://localhost:8000/drivers_list/', {
          withCredentials: true,  // Include cookies (including authToken)
        });
        setDrivers(response.data);
      } catch (error) {
        setError('Error obteniendo conductores');
      }
    };

    fetchDrivers();
  }, []);

  const handleSelectDriver = async (driver) => {
    setSelectedDriver(driver);

    try {
      // Fetch the driver's photo from the FastAPI endpoint
      const photoResponse = await axios.get(`https://localhost:8000/driver_photo/${driver.id}`, {
        responseType: 'blob',
        withCredentials: true,
      });
      const photoUrl = URL.createObjectURL(photoResponse.data);
      setDriverPhoto(photoUrl);

      // Convert the Blob to a File object
      const photoFile = new File([photoResponse.data], `${driver.driver_name}.jpg`, { type: 'image/jpeg' });

      // Call the store_driver_info endpoint on the Flask backend
      const formData = new FormData();
      formData.append('driver_name', driver.driver_name);
      formData.append('photo', photoFile);

      await axios.post('/store_driver_info', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setMessage('Conductor seleccionado correctamente.');
      setError('');
    } catch (error) {
      setMessage('');
      setError('Fallo la seleccion del conductor.');
    }
  };

  return (
    <div className="container mt-5">
      {selectedDriver && (
        <div style={{ position: 'fixed', top: '20px', right: '20px', background: '#000', padding: '10px', boxShadow: '0 0 10px rgba(0,0,0,0.1)', zIndex: 1000 }}>
          <h4>{selectedDriver.driver_name.replace(/_/g, " ")}</h4>
          <p>ID: {selectedDriver.driver_id}</p>
          {driverPhoto && <img src={driverPhoto} alt={selectedDriver.driver_name} style={{ width: '100px', height: '100px', objectFit: 'cover' }} />}
        </div>
      )}
      <h2 className="mb-4">Seleccionar conductor</h2>
      {drivers.length > 0 ? (
        <ul className="list-group" style={{maxHeight: '200px', overflowY: 'auto'}}>
          {drivers.map((driver) => (
            <li 
              key={driver.id} 
              className={`list-group-item ${selectedDriver && selectedDriver.id === driver.id ? 'active' : ''}`} 
              onClick={() => handleSelectDriver(driver)}
              style={{ cursor: 'pointer' }}
            >
              {driver.driver_name.replace(/_/g, " ")}
            </li>
          ))}
        </ul>
      ) : (
        <p>No hay conductore registrados</p>
      )}
      {error && <div className="alert alert-danger mt-3">{error}</div>}
      {message && <div className="alert alert-success mt-3">{message}</div>}
    </div>
  );
};

export default SelectDriver;