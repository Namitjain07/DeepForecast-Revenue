import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchUsersByHotel, updateUserData, deleteUserData } from '../../redux/services/usersApi';
import { addNewUser } from '../../redux/services/authApi';
import '../../stylesheet/ui/component-ui-user-table.css';

interface UserTableProps {
    hotelId: string;
}

interface EditFormData {
    name: string;
    email: string;
    password: string;
}

interface AddUserFormData {
    name: string;
    email: string;
    password: string;
    role: 'owner' | 'manager';
}

const UserTable: React.FC<UserTableProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { users, loading, error } = useSelector((state: RootState) => state.users);
    const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
    const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
    const [selectedUser, setSelectedUser] = useState<any | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [submitSuccess, setSubmitSuccess] = useState<string | null>(null);
    const [editFormData, setEditFormData] = useState<EditFormData>({
        name: '',
        email: '',
        password: ''
    });
    const [addFormData, setAddFormData] = useState<AddUserFormData>({
        name: '',
        email: '',
        password: '',
        role: 'manager'
    });

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchUsersByHotel(hotelId) as any);
        }
    }, [hotelId, dispatch]);

    const handleEdit = (user: any) => {
        setSelectedUser(user);
        setEditFormData({
            name: user.name,
            email: user.email,
            password: ''
        });
        setIsEditDialogOpen(true);
        setSubmitError(null);
    };

    const handleDelete = (userId: string) => {
        if (window.confirm('Are you sure you want to delete this user?')) {
            dispatch(deleteUserData(userId) as any).catch((err: any) => {
                console.error('Delete error:', err);
            });
        }
    };

    const handleEditFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setEditFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleAddFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setAddFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSave = async () => {
        if (selectedUser && selectedUser.id) {
            try {
                setIsSubmitting(true);
                setSubmitError(null);
                const updatePayload = {
                    name: editFormData.name,
                    email: editFormData.email,
                    ...(editFormData.password && { password: editFormData.password })
                };
                await dispatch(updateUserData(selectedUser.id, updatePayload) as any);
                setSubmitSuccess('User updated successfully');
                setTimeout(() => {
                    setIsEditDialogOpen(false);
                    setSubmitSuccess(null);
                    // Refresh users list
                    dispatch(fetchUsersByHotel(hotelId) as any);
                }, 1500);
            } catch (err: any) {
                setSubmitError(err.response?.data?.message || 'Failed to update user');
                console.error('Update error:', err);
            } finally {
                setIsSubmitting(false);
            }
        }
    };

    const handleAddUser = async () => {
        try {
            setIsSubmitting(true);
            setSubmitError(null);

            // Validate required fields
            if (!addFormData.name || !addFormData.email || !addFormData.password || !addFormData.role) {
                setSubmitError('All fields are required');
                setIsSubmitting(false);
                return;
            }

            const userData = {
                name: addFormData.name,
                email: addFormData.email,
                password: addFormData.password,
                hotelId: hotelId,
                role: addFormData.role
            };

            await dispatch(addNewUser(userData) as any);
            setSubmitSuccess('User added successfully');
            setTimeout(() => {
                setIsAddDialogOpen(false);
                setSubmitSuccess(null);
                setAddFormData({
                    name: '',
                    email: '',
                    password: '',
                    role: 'manager'
                });
                // Refresh users list
                dispatch(fetchUsersByHotel(hotelId) as any);
            }, 1500);
        } catch (err: any) {
            setSubmitError(err.response?.data?.message || 'Failed to add user');
            console.error('Add user error:', err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleEditCancel = () => {
        setIsEditDialogOpen(false);
        setSubmitError(null);
        setSubmitSuccess(null);
    };

    const handleAddCancel = () => {
        setIsAddDialogOpen(false);
        setSubmitError(null);
        setSubmitSuccess(null);
        setAddFormData({
            name: '',
            email: '',
            password: '',
            role: 'manager'
        });
    };

    if (loading) {
        return <div className="component-ui-user-table-loading">Loading users...</div>;
    }

    if (error) {
        return <div className="component-ui-user-table-loading">Error: {error}</div>;
    }

    // @ts-ignore
    return (
        <div className="component-ui-user-table-container">
            <div className="component-ui-user-table-header">
                <h2 className="component-ui-user-table-title">Users</h2>
                <button
                    className="component-ui-user-table-add-btn"
                    onClick={() => setIsAddDialogOpen(true)}
                >
                    + Add User
                </button>
            </div>
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
                        {users && users.length > 0 ? (
                            users.map(user => (
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
                            ))
                        ) : (
                            <tr>
                                <td colSpan={4} style={{ textAlign: 'center', padding: '20px', color: '#6b7280' }}>
                                    No users found
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* Edit User Dialog */}
            {isEditDialogOpen && (
                <div className="component-ui-user-table-dialog-overlay">
                    <div className="component-ui-user-table-dialog">
                        <h3 className="component-ui-user-table-dialog-title">Edit User</h3>
                        {submitError && (
                            <div className="component-ui-user-table-dialog-error">
                                ⚠️ {submitError}
                            </div>
                        )}
                        {submitSuccess && (
                            <div className="component-ui-user-table-dialog-success">
                                ✓ {submitSuccess}
                            </div>
                        )}
                        <div className="component-ui-user-table-dialog-body">
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="name" className="component-ui-user-table-form-label">Name</label>
                                <input
                                    type="text"
                                    id="name"
                                    name="name"
                                    value={editFormData.name}
                                    onChange={handleEditFormChange}
                                    className="component-ui-user-table-form-input"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="email" className="component-ui-user-table-form-label">Email</label>
                                <input
                                    type="email"
                                    id="email"
                                    name="email"
                                    value={editFormData.email}
                                    onChange={handleEditFormChange}
                                    className="component-ui-user-table-form-input"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="password" className="component-ui-user-table-form-label">Password</label>
                                <input
                                    type="password"
                                    id="password"
                                    name="password"
                                    value={editFormData.password}
                                    onChange={handleEditFormChange}
                                    placeholder="Leave blank to keep current password"
                                    className="component-ui-user-table-form-input"
                                />
                            </div>
                        </div>
                        <div className="component-ui-user-table-dialog-footer">
                            <button
                                className="component-ui-user-table-dialog-btn component-ui-user-table-dialog-btn-cancel"
                                onClick={handleEditCancel}
                                disabled={isSubmitting}
                            >
                                Cancel
                            </button>
                            <button
                                className="component-ui-user-table-dialog-btn component-ui-user-table-dialog-btn-save"
                                onClick={handleSave}
                                disabled={isSubmitting}
                            >
                                {isSubmitting ? 'Saving...' : 'Save'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Add User Dialog */}
            {isAddDialogOpen && (
                <div className="component-ui-user-table-dialog-overlay">
                    <div className="component-ui-user-table-dialog">
                        <h3 className="component-ui-user-table-dialog-title">Add New User</h3>
                        {submitError && (
                            <div className="component-ui-user-table-dialog-error">
                                ⚠️ {submitError}
                            </div>
                        )}
                        {submitSuccess && (
                            <div className="component-ui-user-table-dialog-success">
                                ✓ {submitSuccess}
                            </div>
                        )}
                        <div className="component-ui-user-table-dialog-body">
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="add-name" className="component-ui-user-table-form-label">Name *</label>
                                <input
                                    type="text"
                                    id="add-name"
                                    name="name"
                                    value={addFormData.name}
                                    onChange={handleAddFormChange}
                                    className="component-ui-user-table-form-input"
                                    placeholder="Enter user name"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="add-email" className="component-ui-user-table-form-label">Email *</label>
                                <input
                                    type="email"
                                    id="add-email"
                                    name="email"
                                    value={addFormData.email}
                                    onChange={handleAddFormChange}
                                    className="component-ui-user-table-form-input"
                                    placeholder="Enter user email"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="add-password" className="component-ui-user-table-form-label">Password *</label>
                                <input
                                    type="password"
                                    id="add-password"
                                    name="password"
                                    value={addFormData.password}
                                    onChange={handleAddFormChange}
                                    className="component-ui-user-table-form-input"
                                    placeholder="Enter password"
                                />
                            </div>
                            <div className="component-ui-user-table-form-group">
                                <label htmlFor="add-role" className="component-ui-user-table-form-label">Role *</label>
                                <select
                                    id="add-role"
                                    name="role"
                                    value={addFormData.role}
                                    onChange={handleAddFormChange}
                                    className="component-ui-user-table-form-input"
                                >
                                    <option value="manager">Manager</option>
                                    <option value="owner">Owner</option>
                                </select>
                            </div>
                        </div>
                        <div className="component-ui-user-table-dialog-footer">
                            <button
                                className="component-ui-user-table-dialog-btn component-ui-user-table-dialog-btn-cancel"
                                onClick={handleAddCancel}
                                disabled={isSubmitting}
                            >
                                Cancel
                            </button>
                            <button
                                className="component-ui-user-table-dialog-btn component-ui-user-table-dialog-btn-save"
                                onClick={handleAddUser}
                                disabled={isSubmitting}
                            >
                                {isSubmitting ? 'Adding...' : 'Add User'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserTable;
