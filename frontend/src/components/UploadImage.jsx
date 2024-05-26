import React, { useState } from 'react';
import axios from 'axios';

const UploadImage = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      await axios.put('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert('Image uploaded successfully');
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="mb-3">
      <label htmlFor="formFile" className="form-label"><h5>Upload Image</h5></label>
      <input className="form-control" type="file" id="formFile" onChange={handleFileChange} />
      <button className="btn btn-primary mt-2" onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default UploadImage;
