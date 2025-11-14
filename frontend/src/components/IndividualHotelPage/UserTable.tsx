import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchUsersByHotel, updateUserData, deleteUserData } from '../../redux/services/api';
import '../../stylesheet/ui/component-ui-user-table.css';

interface UserTableProps {
    hotelId: string;
}

interface EditFormData {
    name: string;
    email: string;
    password: string;
}

const UserTable: React.FC<UserTableProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { users, loading, error } = useSelector((state: RootState) => state.users);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [selectedUser, setSelectedUser] = useState<any | null>(null);
    const [formData, setFormData] = useState<EditFormData>({
        name: '',
        email: '',
        password: ''
    });

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchUsersByHotel(hotelId) as any);
        }
    }, [hotelId, dispatch]);

    const handleEdit = (user: any) => {
        setSelectedUser(user);
        setFormData({
            name: user.name,
            email: user.email,
            password: ''
        });
        setIsDialogOpen(true);
    };

    const handleDelete = (userId: string) => {
        if (window.confirm('Are you sure you want to delete this user?')) {
            dispatch(deleteUserData(userId) as any).catch((err: any) => {
                console.error('Delete error:', err);
            });
        }
    };

    const handleFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSave = () => {
        if (selectedUser && selectedUser.id) {
            const updatePayload = {
                name: formData.name,
                email: formData.email,
                ...(formData.password && { password: formData.password })
            };
            dispatch(updateUserData(selectedUser.id, updatePayload) as any)
                .then(() => {
                    setIsDialogOpen(false);
                    // Refresh users list
                    dispatch(fetchUsersByHotel(hotelId) as any);
                })
                .catch((err: any) => {
                    console.error('Update error:', err);
                });
        }
    };

    const handleCancel = () => {
        setIsDialogOpen(false);
    };

    if (loading) {
        return <div className="component-ui-user-table-loading">Loading users...</div>;
    }

    if (error) {
        return <div className="component-ui-user-table-loading">Error: {error}</div>;
    }

    return (
        <div className="component-ui-user-table-container">
            <h2 className="component-ui-user-table-title">Users</h2>
            <div className="component-ui-user-table-wrapper">
                <table className="component-ui-user-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Role</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {users.map(user => (
                            <tr key={user.id}>
                                <td>{user.name}</td>
                                <td>{user.email}</td>
                                <td>
                                    <span className={`component-ui-user-table-role component-ui-user-table-role-${user.role}`}>
                                        {user.role}
                                    </span>
                                </td>
                                <td>
                                    <div className="component-ui-user-table-actions">
                                        <button
                                            className="component-ui-user-table-btn component-ui-user-table-btn-edit"
                                            onClick={() => handleEdit(user)}
                                        >
                                            Edit
                                        </button>
                                        <button
                                            className="component-ui-user-table-btn component-ui-user-table-btn-delete"
                                            onClick={() => handleDelete(user.id)}
                                        >
                                            Delete
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {isDialogOpen && (
                <div className="component-ui-user-table-dialog-overlay">
                    <div className="component-ui-user-table-dialog">
                        <h3 className="component-ui-user-table-dialog-title">Edit User</h3>
                        <div className="component-ui-user-table-dialog-body">
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="name" className="component-ui-user-table-form-label">Name</label>
                                <input
                                    type="text"
                                    id="name"
                                    name="name"
                                    value={formData.name}
                                    onChange={handleFormChange}
                                    className="component-ui-user-table-form-input"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="email" className="component-ui-user-table-form-label">Email</label>
                                <input
                                    type="email"
                                    id="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleFormChange}
                                    className="component-ui-user-table-form-input"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="password" className="component-ui-user-table-form-label">Password</label>
                                <input
                                    type="password"
                                    id="password"
                                    name="password"
                                    value={formData.password}
                                    onChange={handleFormChange}
                                    placeholder="Leave blank to keep current password"
                                    className="component-ui-user-table-form-input"
                                />
                            </div>
                        </div>
                        <div className="component-ui-user-table-dialog-footer">
                            <button
                                className="component-ui-user-table-dialog-btn component-ui-user-table-dialog-btn-cancel"
                                onClick={handleCancel}
                            >
                                Cancel
                            </button>
                            <button
                                className="component-ui-user-table-dialog-btn component-ui-user-table-dialog-btn-save"
                                onClick={handleSave}
                            >
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserTable;
