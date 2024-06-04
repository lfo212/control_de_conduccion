import React from 'react';

const Toast = ({ message, show, onClose }) => {
  return (
    <div
      className={`toast ${show ? 'show' : ''}`}
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        minWidth: '200px',
        zIndex: 1000,
      }}
    >
      <div className="toast-header">
        <strong className="mr-auto">Notificacion</strong>
        <button
          type="button"
          className="ml-2 mb-1 close"
          data-dismiss="toast"
          aria-label="Close"
          onClick={onClose}
        >
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div className="toast-body">{message}</div>
    </div>
  );
};

export default Toast;
