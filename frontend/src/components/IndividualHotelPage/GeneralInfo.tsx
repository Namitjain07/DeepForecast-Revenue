import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchGeneralInfo } from '../../redux/services/api';

interface GeneralInfoProps {
    hotelId: string;
}

const GeneralInfo: React.FC<GeneralInfoProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { generalInfo, loading, error } = useSelector(
        (state: RootState) => state.hotels
    );

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchGeneralInfo(hotelId) as any);
        }
    }, [hotelId, dispatch]);

    if (loading) {
        return <div className="flex justify-center items-center h-32 text-indigo-600 font-medium animate-pulse">Loading hotel details...</div>;
    }

    if (error) {
        return <div className="text-red-500 p-4 bg-red-50 rounded-lg">Error: {error}</div>;
    }

    if (!generalInfo) {
        return <div className="text-gray-500 p-4 bg-gray-50 rounded-lg">No hotel information found</div>;
    }

    const InfoField = ({ label, value, icon }: { label: string, value: string, icon?: string }) => (
        <div className="bg-gray-50 rounded-lg p-4 hover:bg-indigo-50 transition-colors duration-200 group">
            <div className="flex items-center mb-1">
                {icon && <span className="mr-2 text-lg group-hover:scale-110 transition-transform duration-200">{icon}</span>}
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{label}</label>
            </div>
            <p className="text-gray-900 font-medium text-lg truncate" title={value}>{value}</p>
        </div>
    );

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center mb-6">
                <div className="bg-indigo-100 p-2 rounded-lg mr-3">
                    <span className="text-2xl">🏨</span>
                </div>
                <h2 className="text-xl font-bold text-gray-900">General Information</h2>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <InfoField label="Hotel Name" value={generalInfo.name} icon="🏷️" />
                <InfoField label="Email" value={generalInfo.email} icon="📧" />
                <InfoField label="Contact Number" value={generalInfo.contactNumber} icon="📱" />
                <InfoField label="Plot No" value={generalInfo.plotNo} icon="📍" />
                <InfoField label="Street Name" value={generalInfo.streetName} icon="🛣️" />
                <InfoField label="City" value={generalInfo.city} icon="🏙️" />
                <InfoField label="State" value={generalInfo.state} icon="🗺️" />
                <InfoField label="Pincode" value={generalInfo.pincode} icon="📮" />
            </div>
        </div>
    );
};

export default GeneralInfo;

