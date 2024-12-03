import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import Login from './components/LogIn.jsx'; // Login component
import 'bootstrap/dist/css/bootstrap.min.css';
import './index.css';

const Root = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check authentication status with the backend
    const checkAuth = async () => {
      try {
        const response = await fetch('http://localhost:8000/validate-token', {
          method: 'GET',
          credentials: 'include', // Include cookies in the request
        });
        if (response.ok) {
          setIsAuthenticated(true);
        }
      } catch (error) {
        console.error('Authentication check failed:', error);
      } finally {
        setLoading(false);
      }
    };
    checkAuth();
  }, []);

  if (loading) {
    return <div>Loading...</div>; // Show a loading screen while checking authentication
  }

  return isAuthenticated ? (
    <App />
  ) : (
    <Login onLoginSuccess={() => setIsAuthenticated(true)} />
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
