import React, { useState } from 'react';
import axios from 'axios';
import Toast from './Toast';

const RegisterDriver = () => {
  const [driverName, setDriverName] = useState('');
  const [driverId, setDriverId] = useState('');
  const [photo, setPhoto] = useState(null);
  const [toast, setToast] = useState({ show: false, message: '' });

  const handleNameChange = (e) => {
    setDriverName(e.target.value);
  };

  const handleIdChange = (e) => {
    setDriverId(e.target.value);
  };

  const handlePhotoChange = (e) => {
    setPhoto(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Replace spaces with underscores in driverName
    const formattedDriverName = driverName.replace(/ /g, '_');

    // Check if driverId contains only numbers
    if (!/^\d+$/.test(driverId)) {
      setToast({ show: true, message: 'El ID debe contener solo numeros.' });
      return;
    }

    const formData = new FormData();
    formData.append('driver_name', formattedDriverName);
    formData.append('driver_id', driverId);
    formData.append('photo', photo);

    try {
      const response = await axios.post('http://localhost:8000/register_driver/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setToast({show: true, message:`Conductor registrado exitosamente. ID: ${response.data.driver_id}`});
      setDriverName('');
      setDriverId('');
      setPhoto(null);
    } catch (error) {
      console.error('Error registrando el conductor:', error);
      setToast({show: true, message: 'Fallo el registro del conductor.'});
    }
    
  };

  const closeToast = () => {
    setToast({ show: false, message: '' });
  };

  return (
    <div className="container mt-5">
      <h2 className="mb-4">Registrar conductor</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group d-flex align-items-center">
          <label htmlFor="driverName" className='mr-3'><p>Nombre:</p></label>
          <input
            style={{ marginRight: '10px', marginLeft: '10px', flex: '1', maxWidth: '200px' }}
            type="text"
            className="form-control"
            id="driverName"
            value={driverName}
            onChange={handleNameChange}
            required
          />
        </div>
        <div className="form-group d-flex align-items-center mt-3">
          <label htmlFor="driverId" className='mr-3'><p>ID:</p></label>
          <input
            style={{ marginRight: '10px', marginLeft: '55px', flex: '1', maxWidth: '200px' }}
            type="text"
            className="form-control"
            id="driverId"
            value={driverId}
            onChange={handleIdChange}
            required
          />
        </div>
        <div className="form-group d-flex align-items-center mt-3">
          <label htmlFor="photo" className='mr-3'><p>Foto:</p></label>
          <input
            style={{ marginRight: '10px', marginLeft: '10px', flex: '1', maxWidth: '200px' }}
            type="file"
            className="form-control-file "
            id="photo"
            onChange={handlePhotoChange}
            required
          />
        </div>
        <button type="submit" className="btn btn-primary mt-3">Registrar</button>
      </form>
      <Toast message={toast.message} show={toast.show} onClose={closeToast} />
    </div>
  );
};

export default RegisterDriver;
