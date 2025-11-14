import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchGeneralInfo } from '../../redux/services/api';
import '../../stylesheet/ui/component-ui-general-info.css';

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
        return <div className="component-ui-general-info-loading">Loading...</div>;
    }

    if (error) {
        return <div className="component-ui-general-info-error">Error: {error}</div>;
    }

    if (!generalInfo) {
        return <div className="component-ui-general-info-error">No hotel information found</div>;
    }

    return (
        <div className="component-ui-general-info-container">
            <h2 className="component-ui-general-info-title">General Information</h2>
            <div className="component-ui-general-info-grid">
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Name</label>
                    <p className="component-ui-general-info-value">{generalInfo.name}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Email</label>
                    <p className="component-ui-general-info-value">{generalInfo.email}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Contact Number</label>
                    <p className="component-ui-general-info-value">{generalInfo.contactNumber}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Plot No</label>
                    <p className="component-ui-general-info-value">{generalInfo.plotNo}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Street Name</label>
                    <p className="component-ui-general-info-value">{generalInfo.streetName}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">City</label>
                    <p className="component-ui-general-info-value">{generalInfo.city}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">State</label>
                    <p className="component-ui-general-info-value">{generalInfo.state}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Pincode</label>
                    <p className="component-ui-general-info-value">{generalInfo.pincode}</p>
                </div>
            </div>
        </div>
    );
};

export default GeneralInfo;

