// MediaList.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ListGroup, ListGroupItem, Spinner } from 'reactstrap';

const MediaList = () => {
  const [mediaFiles, setMediaFiles] = useState([]);
  const [selectedMedia, setSelectedMedia] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMediaFiles = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/eventos');
        setMediaFiles(response.data);
      } catch (error) {
        console.error('Error fetching media files:', error);
      } finally {
        setLoading(false);
      }
    };

    // Fetch media files immediately when the component mounts
    fetchMediaFiles();

    // Set up an interval to fetch media files every 5 seconds
    const intervalId = setInterval(fetchMediaFiles, 5000);

    // Clean up the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []);

  const handleMediaClick = (filename) => {
    setSelectedMedia(filename);
  };

  return (
    <div>
      <h2>Eventos registrados</h2>
      {loading ? (
        <div className="d-flex justify-content-center my-3">
          <Spinner color="primary" />
        </div>
      ) : (
        <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
          <ListGroup>
            {mediaFiles.map((file, index) => (
              <ListGroupItem
                key={index}
                action
                onClick={() => handleMediaClick(file)}
                active={selectedMedia === file}
              >
                {file}
              </ListGroupItem>
            ))}
          </ListGroup>
        </div>
      )}
      {selectedMedia && (
        <div className="media-preview mt-4">
          {selectedMedia.endsWith('.mp4') || selectedMedia.endsWith('.mov') ? (
            <video key={selectedMedia} width="480" height="360" controls>
              <source src={`/eventos/${selectedMedia}`} type="video/mp4" />
              Formato no soportado.
            </video>
          ) : (
            <img key={selectedMedia} src={`/eventos/${selectedMedia}`} alt={selectedMedia} className="img-fluid" />
          )}
        </div>
      )}
    </div>
  );
};

export default MediaList;
