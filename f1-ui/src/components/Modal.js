import React from 'react';
import './Modal.css';

const Modal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2 className="text-2xl font-semibold mb-4 text-white">{title}</h2>
        <button onClick={onClose} className="close-button">❌</button>
        {children}
      </div>
    </div>
  );
};

export default Modal;