// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\pages\UserHotelInfo.tsx
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../redux/store';
import UserNavbar from '../components/dashboard/UserNavbar';
import { fetchGeneralInfo, updateHotelInfo } from '../redux/services';
import '../stylesheet/pages/page-user-hotel-info.css';

function UserHotelInfo() {
    const { hotelId } = useParams<{ hotelId: string }>();
    const dispatch = useDispatch<AppDispatch>();
    const { user } = useSelector((state: RootState) => state.auth);
    const hotelInfo = useSelector((state: RootState) => state.hotels.generalInfo);
    const role = user?.role as 'owner' | 'manager' || 'manager';

    const [isEditing, setIsEditing] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        contactNumber: '',
        plotNo: '',
        streetName: '',
        city: '',
        state: '',
        pincode: ''
    });

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchGeneralInfo(hotelId) as any);
        }
    }, [hotelId, dispatch]);

    useEffect(() => {
        if (hotelInfo) {
            setFormData({
                name: hotelInfo.name || '',
                email: hotelInfo.email || '',
                contactNumber: hotelInfo.contactNumber || '',
                plotNo: hotelInfo.plotNo || '',
                streetName: hotelInfo.streetName || '',
                city: hotelInfo.city || '',
                state: hotelInfo.state || '',
                pincode: hotelInfo.pincode || ''
            });
        }
    }, [hotelInfo]);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleEdit = () => {
        setIsEditing(true);
        setError(null);
        setSuccess(false);
    };

    const handleCancel = () => {
        setIsEditing(false);
        setError(null);
        setSuccess(false);
        if (hotelInfo) {
            setFormData({
                name: hotelInfo.name || '',
                email: hotelInfo.email || '',
                contactNumber: hotelInfo.contactNumber || '',
                plotNo: hotelInfo.plotNo || '',
                streetName: hotelInfo.streetName || '',
                city: hotelInfo.city || '',
                state: hotelInfo.state || '',
                pincode: hotelInfo.pincode || ''
            });
        }
    };

    const handleSave = async () => {
        try {
            setIsSaving(true);
            setError(null);

            // Validate all fields
            if (!formData.name || !formData.email || !formData.contactNumber || !formData.plotNo ||
                !formData.streetName || !formData.city || !formData.state || !formData.pincode) {
                setError('All fields are required');
                setIsSaving(false);
                return;
            }

            // Call the updateHotelInfo API
            await dispatch(updateHotelInfo(hotelId || '', formData) as any);

            setSuccess(true);
            setIsEditing(false);

            // Clear success message after 3 seconds
            setTimeout(() => setSuccess(false), 3000);
        } catch (err: any) {
            setError(err.response?.data?.message || 'Failed to update hotel information');
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div>
            <UserNavbar role={role} hotelId={hotelId} />
            <div className="page-user-hotel-info">
                <div className="page-user-hotel-info-header">
                    <h1>Hotel Information</h1>
                    <p>View and manage your hotel property details.</p>
                </div>
                {hotelInfo ? (
                    <div className="page-user-hotel-info-card">
                        {error && (
                            <div className="page-user-hotel-info-error">
                                ⚠️ {error}
                            </div>
                        )}
                        {success && (
                            <div className="page-user-hotel-info-success">
                                ✓ Hotel information updated successfully
                            </div>
                        )}
                        <div className="page-user-hotel-info-actions">
                            <h3>Info</h3>
                            {!isEditing && (
                                <button className="page-user-hotel-info-edit-btn" onClick={handleEdit}>
                                    ✎ Edit
                                </button>
                            )}
                            {isEditing && (
                                <div className="page-user-hotel-info-action-buttons">
                                    <button
                                        className="page-user-hotel-info-save-btn"
                                        onClick={handleSave}
                                        disabled={isSaving}
                                    >
                                        {isSaving ? '⏳ Saving...' : '✓ Save'}
                                    </button>
                                    <button
                                        className="page-user-hotel-info-cancel-btn"
                                        onClick={handleCancel}
                                        disabled={isSaving}
                                    >
                                        ✕ Cancel
                                    </button>
                                </div>
                            )}
                        </div>
                        <div className="page-user-hotel-info-grid">
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">Hotel Name</label>
                                {isEditing ? (
                                    <input
                                        type="text"
                                        name="name"
                                        className="page-user-hotel-info-input"
                                        value={formData.name}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.name}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">Email</label>
                                {isEditing ? (
                                    <input
                                        type="email"
                                        name="email"
                                        className="page-user-hotel-info-input"
                                        value={formData.email}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.email}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">Contact Number</label>
                                {isEditing ? (
                                    <input
                                        type="tel"
                                        name="contactNumber"
                                        className="page-user-hotel-info-input"
                                        value={formData.contactNumber}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.contactNumber}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">Plot Number</label>
                                {isEditing ? (
                                    <input
                                        type="text"
                                        name="plotNo"
                                        className="page-user-hotel-info-input"
                                        value={formData.plotNo}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.plotNo}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">Street Name</label>
                                {isEditing ? (
                                    <input
                                        type="text"
                                        name="streetName"
                                        className="page-user-hotel-info-input"
                                        value={formData.streetName}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.streetName}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">City</label>
                                {isEditing ? (
                                    <input
                                        type="text"
                                        name="city"
                                        className="page-user-hotel-info-input"
                                        value={formData.city}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.city}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">State</label>
                                {isEditing ? (
                                    <input
                                        type="text"
                                        name="state"
                                        className="page-user-hotel-info-input"
                                        value={formData.state}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.state}</p>
                                )}
                            </div>
                            <div className="page-user-hotel-info-field">
                                <label className="page-user-hotel-info-label">Pincode</label>
                                {isEditing ? (
                                    <input
                                        type="text"
                                        name="pincode"
                                        className="page-user-hotel-info-input"
                                        value={formData.pincode}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <p className="page-user-hotel-info-value">{formData.pincode}</p>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="page-user-hotel-info-loading">
                        Loading hotel information...
                    </div>
                )}
            </div>
        </div>
    );
}

export default UserHotelInfo;

