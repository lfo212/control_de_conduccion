import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DeleteDriver = () => {
  const [drivers, setDrivers] = useState([]);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDrivers = async () => {
      try {
        const response = await axios.get('http://localhost:8000/drivers_list/');
        setDrivers(response.data);
      } catch (error) {
        setError('Error obteniendo los conductores.');
      }
    };

    fetchDrivers();
  }, []);

  const handleDeleteDriver = async (driverId) => {
    try {
      await axios.delete(`http://localhost:8000/delete_driver/${driverId}`);
      setMessage('Conductor eliminado correctamente');
      setError('');
      setDrivers(drivers.filter(driver => driver.id !== driverId));
    } catch (error) {
      setMessage('');
      setError('Error al eliminar conductor');
    }
  };

  const handleDeleteAllDrivers = async () => {
    try {
      await axios.delete('http://localhost:8000/delete_all_drivers');
      setMessage('Todos los conductores eliminados correctamente.');
      setError('');
      setDrivers([]);
    } catch (error) {
      setMessage('');
      setError('Error al eliminar conductores');
    }
  };

  return (
    <div className="container mt-5">
      <h2 className="mb-4">Borrar conductor</h2>
      {drivers.length > 0 ? (
        <ul className="list-group">
          {drivers.map((driver) => (
            <li 
              key={driver.id} 
              className="list-group-item" 
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              {driver.driver_name.replace(/_/g, ' ')}
              <button 
                className="btn btn-danger"
                onClick={() => handleDeleteDriver(driver.id)}
              >
                Borrar
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p>No hay conductores disponibles.</p>
      )}
      {drivers.length > 0 && (
        <button 
          className="btn btn-danger mt-3"
          onClick={handleDeleteAllDrivers}
        >
          Eliminar Todos
        </button>
      )}
      {error && <div className="alert alert-danger mt-3">{error}</div>}
      {message && <div className="alert alert-success mt-3">{message}</div>}
    </div>
  );
};

export default DeleteDriver;
