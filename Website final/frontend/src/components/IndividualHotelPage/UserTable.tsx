import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchUsersByHotel, updateUserData, deleteUserData } from '../../redux/services/usersApi';
import { addNewUser } from '../../redux/services/authApi';

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
        return <div className="flex justify-center items-center h-32 text-indigo-600 font-medium animate-pulse">Loading users...</div>;
    }

    if (error) {
        return <div className="text-red-500 p-4 bg-red-50 rounded-lg">Error: {error}</div>;
    }

    // @ts-ignore
    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex justify-between items-center mb-6">
                <div className="flex items-center">
                    <div className="bg-indigo-100 p-2 rounded-lg mr-3">
                        <span className="text-2xl">👥</span>
                    </div>
                    <h2 className="text-xl font-bold text-gray-900">Users</h2>
                </div>
                <button
                    className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200"
                    onClick={() => setIsAddDialogOpen(true)}
                >
                    + Add User
                </button>
            </div>
            
            <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                    <thead className="bg-gray-50">
                        <tr>
                            <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">Name</th>
                            <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Email</th>
                            <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Role</th>
                            <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6">
                                <span className="sr-only">Actions</span>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 bg-white">
                        {users && users.length > 0 ? (
                            users.map(user => (
                                <tr key={user.id} className="hover:bg-gray-50 transition-colors duration-150">
                                    <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">{user.name}</td>
                                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{user.email}</td>
                                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${
                                            user.role === 'owner' ? 'bg-purple-100 text-purple-800' : 'bg-green-100 text-green-800'
                                        }`}>
                                            {user.role}
                                        </span>
                                    </td>
                                    <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                                        <button
                                            className="text-indigo-600 hover:text-indigo-900 mr-4 transition-colors duration-200"
                                            onClick={() => handleEdit(user)}
                                        >
                                            Edit
                                        </button>
                                        <button
                                            className="text-red-600 hover:text-red-900 transition-colors duration-200"
                                            onClick={() => handleDelete(user.id)}
                                        >
                                            Delete
                                        </button>
                                    </td>
                                </tr>
                            ))
                        ) : (
                            <tr>
                                <td colSpan={4} className="text-center py-8 text-gray-500">
                                    No users found
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* Edit User Dialog */}
            {isEditDialogOpen && (
                <div className="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
                    <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity backdrop-blur-sm" aria-hidden="true" onClick={handleEditCancel}></div>
                        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                        <div className="relative inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                                <div className="sm:flex sm:items-start">
                                    <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                                        <h3 className="text-lg leading-6 font-medium text-gray-900" id="modal-title">Edit User</h3>
                                        {submitError && (
                                            <div className="mt-2 p-2 bg-red-50 text-red-700 text-sm rounded-md">
                                                ⚠️ {submitError}
                                            </div>
                                        )}
                                        {submitSuccess && (
                                            <div className="mt-2 p-2 bg-green-50 text-green-700 text-sm rounded-md">
                                                ✓ {submitSuccess}
                                            </div>
                                        )}
                                        <div className="mt-4 space-y-4">
                                            <div>
                                                <label htmlFor="name" className="block text-sm font-medium text-gray-700">Name</label>
                                                <input
                                                    type="text"
                                                    id="name"
                                                    name="name"
                                                    value={editFormData.name}
                                                    onChange={handleEditFormChange}
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                />
                                            </div>
                                            <div>
                                                <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
                                                <input
                                                    type="email"
                                                    id="email"
                                                    name="email"
                                                    value={editFormData.email}
                                                    onChange={handleEditFormChange}
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                />
                                            </div>
                                            <div>
                                                <label htmlFor="password" className="block text-sm font-medium text-gray-700">Password</label>
                                                <input
                                                    type="password"
                                                    id="password"
                                                    name="password"
                                                    value={editFormData.password}
                                                    onChange={handleEditFormChange}
                                                    placeholder="Leave blank to keep current password"
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                                <button
                                    type="button"
                                    className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-indigo-600 text-base font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:ml-3 sm:w-auto sm:text-sm disabled:opacity-50"
                                    onClick={handleSave}
                                    disabled={isSubmitting}
                                >
                                    {isSubmitting ? 'Saving...' : 'Save'}
                                </button>
                                <button
                                    type="button"
                                    className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
                                    onClick={handleEditCancel}
                                    disabled={isSubmitting}
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Add User Dialog */}
            {isAddDialogOpen && (
                <div className="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
                    <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity backdrop-blur-sm" aria-hidden="true" onClick={handleAddCancel}></div>
                        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                        <div className="relative inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                                <div className="sm:flex sm:items-start">
                                    <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                                        <h3 className="text-lg leading-6 font-medium text-gray-900" id="modal-title">Add New User</h3>
                                        {submitError && (
                                            <div className="mt-2 p-2 bg-red-50 text-red-700 text-sm rounded-md">
                                                ⚠️ {submitError}
                                            </div>
                                        )}
                                        {submitSuccess && (
                                            <div className="mt-2 p-2 bg-green-50 text-green-700 text-sm rounded-md">
                                                ✓ {submitSuccess}
                                            </div>
                                        )}
                                        <div className="mt-4 space-y-4">
                                            <div>
                                                <label htmlFor="add-name" className="block text-sm font-medium text-gray-700">Name *</label>
                                                <input
                                                    type="text"
                                                    id="add-name"
                                                    name="name"
                                                    value={addFormData.name}
                                                    onChange={handleAddFormChange}
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    placeholder="Enter user name"
                                                />
                                            </div>
                                            <div>
                                                <label htmlFor="add-email" className="block text-sm font-medium text-gray-700">Email *</label>
                                                <input
                                                    type="email"
                                                    id="add-email"
                                                    name="email"
                                                    value={addFormData.email}
                                                    onChange={handleAddFormChange}
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    placeholder="Enter user email"
                                                />
                                            </div>
                                            <div>
                                                <label htmlFor="add-password" className="block text-sm font-medium text-gray-700">Password *</label>
                                                <input
                                                    type="password"
                                                    id="add-password"
                                                    name="password"
                                                    value={addFormData.password}
                                                    onChange={handleAddFormChange}
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    placeholder="Enter password"
                                                />
                                            </div>
                                            <div>
                                                <label htmlFor="add-role" className="block text-sm font-medium text-gray-700">Role *</label>
                                                <select
                                                    id="add-role"
                                                    name="role"
                                                    value={addFormData.role}
                                                    onChange={handleAddFormChange}
                                                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                >
                                                    <option value="manager">Manager</option>
                                                    <option value="owner">Owner</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                                <button
                                    type="button"
                                    className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-indigo-600 text-base font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:ml-3 sm:w-auto sm:text-sm disabled:opacity-50"
                                    onClick={handleAddUser}
                                    disabled={isSubmitting}
                                >
                                    {isSubmitting ? 'Adding...' : 'Add User'}
                                </button>
                                <button
                                    type="button"
                                    className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
                                    onClick={handleAddCancel}
                                    disabled={isSubmitting}
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserTable;
