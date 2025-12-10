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
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
            <AdminNavbar role="admin" />
            
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Header Section */}
                <div className="mb-10">
                    <div className="flex items-center space-x-3 mb-3">
                        <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg">
                            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                            </svg>
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-indigo-900 bg-clip-text text-transparent">User Management</h1>
                            <p className="mt-1 text-sm text-gray-600">Create and manage system users and administrators with ease</p>
                        </div>
                    </div>
                </div>

                {/* Action Cards */}
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-2 max-w-4xl mx-auto">
                    {/* Add User Card */}
                    <div className="group relative bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 overflow-hidden border border-indigo-100 hover:border-indigo-300 transform hover:-translate-y-1">
                        <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-br from-indigo-400 to-indigo-600 opacity-10 rounded-full -mr-20 -mt-20 group-hover:scale-150 transition-transform duration-500"></div>
                        <div className="relative p-8">
                            <div className="flex items-start justify-between mb-6">
                                <div className="p-3 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl shadow-lg group-hover:shadow-xl transition-shadow duration-300">
                                    <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                                    </svg>
                                </div>
                                <span className="px-3 py-1 bg-indigo-100 text-indigo-700 text-xs font-semibold rounded-full">User</span>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-3">Add New User</h3>
                            <p className="text-gray-600 mb-6 leading-relaxed">Create hotel owner or manager accounts to manage properties and operations</p>
                            <button
                                onClick={() => handleOpenAddDialog('user')}
                                className="w-full group/btn bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white font-semibold py-3 px-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center space-x-2"
                            >
                                <svg className="w-5 h-5 group-hover/btn:rotate-90 transition-transform duration-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                </svg>
                                <span>Create User Account</span>
                            </button>
                        </div>
                    </div>

                    {/* Add Admin Card */}
                    <div className="group relative bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 overflow-hidden border border-purple-100 hover:border-purple-300 transform hover:-translate-y-1">
                        <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-br from-purple-400 to-purple-600 opacity-10 rounded-full -mr-20 -mt-20 group-hover:scale-150 transition-transform duration-500"></div>
                        <div className="relative p-8">
                            <div className="flex items-start justify-between mb-6">
                                <div className="p-3 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl shadow-lg group-hover:shadow-xl transition-shadow duration-300">
                                    <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                    </svg>
                                </div>
                                <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-semibold rounded-full">Admin</span>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-3">Add New Admin</h3>
                            <p className="text-gray-600 mb-6 leading-relaxed">Create system administrator accounts with full access and privileges</p>
                            <button
                                onClick={() => handleOpenAddDialog('admin')}
                                className="w-full group/btn bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-semibold py-3 px-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center space-x-2"
                            >
                                <svg className="w-5 h-5 group-hover/btn:rotate-90 transition-transform duration-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                </svg>
                                <span>Create Admin Account</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Info Banner */}
                <div className="mt-10 max-w-4xl mx-auto">
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6 flex items-start space-x-4">
                        <div className="flex-shrink-0">
                            <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <div>
                            <h4 className="text-sm font-semibold text-gray-900 mb-1">Quick Tips</h4>
                            <ul className="text-sm text-gray-600 space-y-1">
                                <li>• <strong>Users</strong> can be assigned as owners or managers for specific hotels</li>
                                <li>• <strong>Admins</strong> have full system access and can manage all hotels and users</li>
                                <li>• All accounts require a valid email address and secure password</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {/* Add Dialog */}
            {isAddDialogOpen && (
                <div className="fixed z-50 inset-0 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
                    <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                        <div className="fixed inset-0 bg-gray-900 bg-opacity-75 backdrop-blur-sm transition-opacity" aria-hidden="true" onClick={handleAddCancel}></div>
                        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
                        
                        <div className="relative inline-block align-bottom bg-white rounded-2xl text-left overflow-hidden shadow-2xl transform transition-all sm:my-8 sm:align-middle sm:max-w-md sm:w-full">
                            {/* Header with gradient */}
                            <div className={`px-6 py-6 ${addFormType === 'user' ? 'bg-gradient-to-r from-indigo-500 to-indigo-600' : 'bg-gradient-to-r from-purple-500 to-purple-600'}`}>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-3">
                                        <div className="p-2 bg-white bg-opacity-20 rounded-lg backdrop-blur-sm">
                                            {addFormType === 'user' ? (
                                                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                                                </svg>
                                            ) : (
                                                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                                </svg>
                                            )}
                                        </div>
                                        <h3 className="text-xl font-bold text-white" id="modal-title">
                                            {addFormType === 'user' ? 'Add New User' : 'Add New Admin'}
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
                                
                                {addFormType === 'user' ? (
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
                                                value={addUserFormData.name}
                                                onChange={handleAddUserFormChange}
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
                                                value={addUserFormData.email}
                                                onChange={handleAddUserFormChange}
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
                                                value={addUserFormData.password}
                                                onChange={handleAddUserFormChange}
                                                placeholder="Enter secure password"
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300"
                                            />
                                        </div>
                                        <div className="group">
                                            <label htmlFor="add-hotel" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                                <svg className="w-4 h-4 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                                                </svg>
                                                Assigned Hotel <span className="text-red-500 ml-1">*</span>
                                            </label>
                                            <select
                                                id="add-hotel"
                                                name="hotelId"
                                                value={addUserFormData.hotelId}
                                                onChange={handleAddUserFormChange}
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300 appearance-none bg-white"
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
                                                value={addUserFormData.role}
                                                onChange={handleAddUserFormChange}
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200 group-hover:border-indigo-300 appearance-none bg-white"
                                            >
                                                <option value="manager">Manager</option>
                                                <option value="owner">Owner</option>
                                            </select>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="space-y-5">
                                        <div className="group">
                                            <label htmlFor="add-admin-name" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                                <svg className="w-4 h-4 mr-2 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                                </svg>
                                                Full Name <span className="text-red-500 ml-1">*</span>
                                            </label>
                                            <input
                                                type="text"
                                                id="add-admin-name"
                                                name="name"
                                                value={addAdminFormData.name}
                                                onChange={handleAddAdminFormChange}
                                                placeholder="Enter full name"
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200 group-hover:border-purple-300"
                                            />
                                        </div>
                                        <div className="group">
                                            <label htmlFor="add-admin-email" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                                <svg className="w-4 h-4 mr-2 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                                </svg>
                                                Email Address <span className="text-red-500 ml-1">*</span>
                                            </label>
                                            <input
                                                type="email"
                                                id="add-admin-email"
                                                name="email"
                                                value={addAdminFormData.email}
                                                onChange={handleAddAdminFormChange}
                                                placeholder="admin@example.com"
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200 group-hover:border-purple-300"
                                            />
                                        </div>
                                        <div className="group">
                                            <label htmlFor="add-admin-password" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                                <svg className="w-4 h-4 mr-2 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                                </svg>
                                                Password <span className="text-red-500 ml-1">*</span>
                                            </label>
                                            <input
                                                type="password"
                                                id="add-admin-password"
                                                name="password"
                                                value={addAdminFormData.password}
                                                onChange={handleAddAdminFormChange}
                                                placeholder="Enter secure password"
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200 group-hover:border-purple-300"
                                            />
                                        </div>
                                        <div className="group">
                                            <label htmlFor="add-admin-imageUrl" className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                                                <svg className="w-4 h-4 mr-2 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                                </svg>
                                                Profile Image URL
                                            </label>
                                            <input
                                                type="text"
                                                id="add-admin-imageUrl"
                                                name="imageUrl"
                                                value={addAdminFormData.imageUrl}
                                                onChange={handleAddAdminFormChange}
                                                placeholder="https://example.com/image.jpg (optional)"
                                                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200 group-hover:border-purple-300"
                                            />
                                        </div>
                                    </div>
                                )}
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
                                    className={`px-6 py-2.5 rounded-xl text-white font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 ${
                                        addFormType === 'user' 
                                            ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 focus:ring-indigo-500 shadow-lg shadow-indigo-500/50' 
                                            : 'bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 focus:ring-purple-500 shadow-lg shadow-purple-500/50'
                                    }`}
                                    onClick={handleSubmit}
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
                                            <span>Create {addFormType === 'user' ? 'User' : 'Admin'}</span>
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

export default AdminUserManager;
