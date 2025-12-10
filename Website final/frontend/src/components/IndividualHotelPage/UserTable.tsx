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
                    <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                        <div className="fixed inset-0 bg-gray-900 bg-opacity-75 backdrop-blur-sm transition-opacity" aria-hidden="true" onClick={handleAddCancel}></div>
                        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                        
                        <div className="relative inline-block align-bottom bg-white rounded-2xl text-left overflow-hidden shadow-2xl transform transition-all sm:my-8 sm:align-middle sm:max-w-md sm:w-full">
                            {/* Header with gradient */}
                            <div className="px-6 py-6 bg-gradient-to-r from-indigo-500 to-indigo-600">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-3">
                                        <div className="p-2 bg-white bg-opacity-20 rounded-lg backdrop-blur-sm">
                                            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                                            </svg>
                                        </div>
                                        <h3 className="text-xl font-bold text-white" id="modal-title">
                                            Add New User
                                        </h3>
                                    </div>
                                    <button 
                                        onClick={handleAddCancel}
                                        className="p-1 rounded-lg hover:bg-white hover:bg-opacity-20 transition-colors"
                                    >
                                        <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </button>
                                </div>
                            </div>

                            <div className="bg-white px-6 py-6">
                                {submitError && (
                                    <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-400 rounded-r-lg flex items-start space-x-3 animate-pulse">
                                        <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <p className="text-sm text-red-700 font-medium">{submitError}</p>
                                    </div>
                                )}
                                
                                {submitSuccess && (
                                    <div className="mb-4 p-4 bg-green-50 border-l-4 border-green-400 rounded-r-lg flex items-start space-x-3">
                                        <svg className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <p className="text-sm text-green-700 font-medium">{submitSuccess}</p>
                                    </div>
                                )}
                                
                                <div className="space-y-5">
                                    <div className="group">
                                        <label htmlFor="add-name" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                            <svg className="w-4 h-4 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                            </svg>
                                            Full Name <span className="text-red-500 ml-1">*</span>
                                        </label>
                                        <input
                                            type="text"
                                            id="add-name"
                                            name="name"
                                            value={addFormData.name}
                                            onChange={handleAddFormChange}
                                            placeholder="Enter full name"
                                            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300"
                                        />
                                    </div>
                                    <div className="group">
                                        <label htmlFor="add-email" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                            <svg className="w-4 h-4 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                            </svg>
                                            Email Address <span className="text-red-500 ml-1">*</span>
                                        </label>
                                        <input
                                            type="email"
                                            id="add-email"
                                            name="email"
                                            value={addFormData.email}
                                            onChange={handleAddFormChange}
                                            placeholder="user@example.com"
                                            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300"
                                        />
                                    </div>
                                    <div className="group">
                                        <label htmlFor="add-password" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                            <svg className="w-4 h-4 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                            </svg>
                                            Password <span className="text-red-500 ml-1">*</span>
                                        </label>
                                        <input
                                            type="password"
                                            id="add-password"
                                            name="password"
                                            value={addFormData.password}
                                            onChange={handleAddFormChange}
                                            placeholder="Enter secure password"
                                            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300"
                                        />
                                    </div>
                                    <div className="group">
                                        <label htmlFor="add-role" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                            <svg className="w-4 h-4 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                            </svg>
                                            User Role <span className="text-red-500 ml-1">*</span>
                                        </label>
                                        <select
                                            id="add-role"
                                            name="role"
                                            value={addFormData.role}
                                            onChange={handleAddFormChange}
                                            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300 appearance-none bg-white"
                                        >
                                            <option value="manager">Manager</option>
                                            <option value="owner">Owner</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <div className="bg-gray-50 px-6 py-4 flex justify-end space-x-3">
                                <button
                                    type="button"
                                    className="px-6 py-2.5 border border-gray-300 rounded-xl text-gray-700 font-medium hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-all duration-200"
                                    onClick={handleAddCancel}
                                    disabled={isSubmitting}
                                >
                                    Cancel
                                </button>
                                <button
                                    type="button"
                                    className="px-6 py-2.5 rounded-xl text-white font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 focus:ring-indigo-500 shadow-lg shadow-indigo-500/50"
                                    onClick={handleAddUser}
                                    disabled={isSubmitting}
                                >
                                    {isSubmitting ? (
                                        <>
                                            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            <span>Creating...</span>
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                            </svg>
                                            <span>Create User</span>
                                        </>
                                    )}
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
