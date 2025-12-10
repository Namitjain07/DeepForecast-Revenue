import { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../redux/store';
import AdminNavbar from '../components/dashboard/AdminNavbar';
import { addNewAdmin, addNewUser } from '../redux/services/authApi';
import { fetchAllHotels } from '../redux/services/hotelApi';
import '../stylesheet/pages/page-user-manager.css';

interface AddUserFormData {
    name: string;
    email: string;
    password: string;
    role: 'owner' | 'manager';
    hotelId: string;
}

interface AddAdminFormData {
    name: string;
    email: string;
    password: string;
    imageUrl?: string;
}

type AddFormType = 'user' | 'admin';

const AdminUserManager = () => {
    const dispatch = useDispatch<AppDispatch>();
    const { hotels, loading: hotelsLoading } = useSelector((state: RootState) => state.hotels);
    const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
    const [addFormType, setAddFormType] = useState<AddFormType>('user');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [submitSuccess, setSubmitSuccess] = useState<string | null>(null);
    
    const [addUserFormData, setAddUserFormData] = useState<AddUserFormData>({
        name: '',
        email: '',
        password: '',
        role: 'manager',
        hotelId: ''
    });
    
    const [addAdminFormData, setAddAdminFormData] = useState<AddAdminFormData>({
        name: '',
        email: '',
        password: '',
        imageUrl: ''
    });

    useEffect(() => {
        // Fetch all hotels for the dropdown
        dispatch(fetchAllHotels());
    }, [dispatch]);

    const handleAddUserFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setAddUserFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleAddAdminFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setAddAdminFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleOpenAddDialog = (type: AddFormType) => {
        setAddFormType(type);
        setIsAddDialogOpen(true);
        setSubmitError(null);
        setSubmitSuccess(null);
    };

    const handleAddCancel = () => {
        setIsAddDialogOpen(false);
        setAddUserFormData({
            name: '',
            email: '',
            password: '',
            role: 'manager',
            hotelId: ''
        });
        setAddAdminFormData({
            name: '',
            email: '',
            password: '',
            imageUrl: ''
        });
        setSubmitError(null);
        setSubmitSuccess(null);
    };

    const handleAddUser = async () => {
        try {
            setIsSubmitting(true);
            setSubmitError(null);

            // Validate required fields
            if (!addUserFormData.name || !addUserFormData.email || !addUserFormData.password || !addUserFormData.hotelId) {
                setSubmitError('All fields are required');
                setIsSubmitting(false);
                return;
            }

            await dispatch(addNewUser(addUserFormData));
            setSubmitSuccess('User added successfully');
            setTimeout(() => {
                setIsAddDialogOpen(false);
                handleAddCancel();
                setSubmitSuccess(null);
            }, 1500);
        } catch (err: unknown) {
            const error = err as { response?: { data?: { message?: string } } };
            setSubmitError(error.response?.data?.message || 'Failed to add user');
            console.error('Add user error:', err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleAddAdmin = async () => {
        try {
            setIsSubmitting(true);
            setSubmitError(null);

            // Validate required fields
            if (!addAdminFormData.name || !addAdminFormData.email || !addAdminFormData.password) {
                setSubmitError('Name, email, and password are required');
                setIsSubmitting(false);
                return;
            }

            await dispatch(addNewAdmin(addAdminFormData));
            setSubmitSuccess('Admin added successfully');
            setTimeout(() => {
                setIsAddDialogOpen(false);
                handleAddCancel();
                setSubmitSuccess(null);
            }, 1500);
        } catch (err: unknown) {
            const error = err as { response?: { data?: { message?: string } } };
            setSubmitError(error.response?.data?.message || 'Failed to add admin');
            console.error('Add admin error:', err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleSubmit = () => {
        if (addFormType === 'user') {
            handleAddUser();
        } else {
            handleAddAdmin();
        }
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <AdminNavbar role="admin" />
            
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-2xl font-bold text-gray-900">User Management</h1>
                    <p className="mt-1 text-sm text-gray-500">Add and manage system users and administrators</p>
                </div>

                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="px-6 py-5 border-b border-gray-100 bg-gray-50/50">
                        <h3 className="text-lg font-medium leading-6 text-gray-900">Quick Actions</h3>
                    </div>
                    
                    <div className="p-6">
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                            <div className="border-2 border-indigo-200 rounded-lg p-6 hover:border-indigo-400 transition-colors">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex-shrink-0">
                                        <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">
                                            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                            </svg>
                                        </div>
                                    </div>
                                </div>
                                <h3 className="text-lg font-medium text-gray-900 mb-2">Add New User</h3>
                                <p className="text-sm text-gray-500 mb-4">Create hotel owner or manager accounts</p>
                                <button
                                    onClick={() => handleOpenAddDialog('user')}
                                    className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200"
                                >
                                    Add User
                                </button>
                            </div>

                            <div className="border-2 border-purple-200 rounded-lg p-6 hover:border-purple-400 transition-colors">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex-shrink-0">
                                        <div className="flex items-center justify-center h-12 w-12 rounded-md bg-purple-500 text-white">
                                            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                            </svg>
                                        </div>
                                    </div>
                                </div>
                                <h3 className="text-lg font-medium text-gray-900 mb-2">Add New Admin</h3>
                                <p className="text-sm text-gray-500 mb-4">Create system administrator accounts</p>
                                <button
                                    onClick={() => handleOpenAddDialog('admin')}
                                    className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition-colors duration-200"
                                >
                                    Add Admin
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Add Dialog */}
            {isAddDialogOpen && (
                <div className="fixed z-50 inset-0 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
                    <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" aria-hidden="true" onClick={handleAddCancel}></div>
                        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                        
                        <div className="relative inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                                <div className="sm:flex sm:items-start">
                                    <div className="mt-3 text-center sm:mt-0 sm:text-left w-full">
                                        <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4" id="modal-title">
                                            {addFormType === 'user' ? 'Add New User' : 'Add New Admin'}
                                        </h3>
                                        
                                        {submitError && (
                                            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
                                                <p className="text-sm text-red-600">{submitError}</p>
                                            </div>
                                        )}
                                        
                                        {submitSuccess && (
                                            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md">
                                                <p className="text-sm text-green-600">{submitSuccess}</p>
                                            </div>
                                        )}
                                        
                                        {addFormType === 'user' ? (
                                            <div className="space-y-4">
                                                <div>
                                                    <label htmlFor="add-name" className="block text-sm font-medium text-gray-700">Name *</label>
                                                    <input
                                                        type="text"
                                                        id="add-name"
                                                        name="name"
                                                        value={addUserFormData.name}
                                                        onChange={handleAddUserFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label htmlFor="add-email" className="block text-sm font-medium text-gray-700">Email *</label>
                                                    <input
                                                        type="email"
                                                        id="add-email"
                                                        name="email"
                                                        value={addUserFormData.email}
                                                        onChange={handleAddUserFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label htmlFor="add-password" className="block text-sm font-medium text-gray-700">Password *</label>
                                                    <input
                                                        type="password"
                                                        id="add-password"
                                                        name="password"
                                                        value={addUserFormData.password}
                                                        onChange={handleAddUserFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label htmlFor="add-hotel" className="block text-sm font-medium text-gray-700">Hotel *</label>
                                                    <select
                                                        id="add-hotel"
                                                        name="hotelId"
                                                        value={addUserFormData.hotelId}
                                                        onChange={handleAddUserFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    >
                                                        <option value="">Select a hotel</option>
                                                        {hotelsLoading ? (
                                                            <option value="">Loading hotels...</option>
                                                        ) : (
                                                            hotels.map((hotel) => (
                                                                <option key={hotel.id} value={hotel.id}>
                                                                    {hotel.name} - {hotel.city}
                                                                </option>
                                                            ))
                                                        )}
                                                    </select>
                                                </div>
                                                <div>
                                                    <label htmlFor="add-role" className="block text-sm font-medium text-gray-700">Role *</label>
                                                    <select
                                                        id="add-role"
                                                        name="role"
                                                        value={addUserFormData.role}
                                                        onChange={handleAddUserFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                                                    >
                                                        <option value="manager">Manager</option>
                                                        <option value="owner">Owner</option>
                                                    </select>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="space-y-4">
                                                <div>
                                                    <label htmlFor="add-admin-name" className="block text-sm font-medium text-gray-700">Name *</label>
                                                    <input
                                                        type="text"
                                                        id="add-admin-name"
                                                        name="name"
                                                        value={addAdminFormData.name}
                                                        onChange={handleAddAdminFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-purple-500 focus:border-purple-500 sm:text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label htmlFor="add-admin-email" className="block text-sm font-medium text-gray-700">Email *</label>
                                                    <input
                                                        type="email"
                                                        id="add-admin-email"
                                                        name="email"
                                                        value={addAdminFormData.email}
                                                        onChange={handleAddAdminFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-purple-500 focus:border-purple-500 sm:text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label htmlFor="add-admin-password" className="block text-sm font-medium text-gray-700">Password *</label>
                                                    <input
                                                        type="password"
                                                        id="add-admin-password"
                                                        name="password"
                                                        value={addAdminFormData.password}
                                                        onChange={handleAddAdminFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-purple-500 focus:border-purple-500 sm:text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label htmlFor="add-admin-imageUrl" className="block text-sm font-medium text-gray-700">Image URL (Optional)</label>
                                                    <input
                                                        type="text"
                                                        id="add-admin-imageUrl"
                                                        name="imageUrl"
                                                        value={addAdminFormData.imageUrl}
                                                        onChange={handleAddAdminFormChange}
                                                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-purple-500 focus:border-purple-500 sm:text-sm"
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                                <button
                                    type="button"
                                    className={`w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 text-base font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 sm:ml-3 sm:w-auto sm:text-sm disabled:opacity-50 ${
                                        addFormType === 'user' 
                                            ? 'bg-indigo-600 hover:bg-indigo-700 focus:ring-indigo-500' 
                                            : 'bg-purple-600 hover:bg-purple-700 focus:ring-purple-500'
                                    }`}
                                    onClick={handleSubmit}
                                    disabled={isSubmitting}
                                >
                                    {isSubmitting ? 'Adding...' : `Add ${addFormType === 'user' ? 'User' : 'Admin'}`}
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

export default AdminUserManager;
