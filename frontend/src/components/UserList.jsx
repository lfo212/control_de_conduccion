import React, { useEffect, useState } from 'react';
import axios from 'axios';

const UserList = () => {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await axios.get('/users');
        setUsers(response.data);
      } catch (error) {
        console.error('Error fetching users:', error);
      }
    };

    fetchUsers();
  }, []);

  return (
    <div>
      <h5>User List</h5>
      <ul className="list-group">
        {users.map((user) => (
          <li key={user.id} className="list-group-item">{user.name}</li>
        ))}
      </ul>
    </div>
  );
};

export default UserList;
