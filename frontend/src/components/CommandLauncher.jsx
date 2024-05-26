import React, { useState } from 'react';
import axios from 'axios';
import Toast from './Toast';

const ControlButton = () => {
    const [buttonText, setButtonText] = useState('Start');
    const [isStarted, setIsStarted] = useState(false);
    const [toastMessage, setToastMessage] = useState('');
    const [showToast, setShowToast] = useState(false);
  
    const handleClick = async () => {
      setToastMessage('Ejecutando comando...');
      setShowToast(true);
  
      try {
        const response = await axios.post('/toggle_command');
        const { output, started } = response.data;
        setIsStarted(started);
        setButtonText(started ? 'Stop' : 'Start');
        setToastMessage('Ejecucion exitosa!');
        console.log(output); // Output from the server command execution
      } catch (error) {
        setToastMessage('Error toggling command');
        console.error('Error toggling command:', error);
      } finally {
        // Automatically close the toast after 2 seconds
        setTimeout(() => {
          setShowToast(false);
        }, 2000);
      }
    };
  
    return (
      <div>
        <button onClick={handleClick} className="btn btn-primary d-flex justify-content-end mt-3">{buttonText}</button>
        <p>Status: {isStarted ? 'Iniciado' : 'Detenido'}</p>
        <Toast message={toastMessage} show={showToast} />
      </div>
    );
  };

export default ControlButton;