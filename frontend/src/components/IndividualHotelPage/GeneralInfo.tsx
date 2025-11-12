import React, { useEffect, useState } from 'react';
import '../../stylesheet/ui/component-ui-general-info.css';

interface GeneralInfoProps {
    hotelId: string;
}

interface HotelData {
    name: string;
    email: string;
    contactNumber: string;
    plotNo: string;
    streetName: string;
    city: string;
    state: string;
    pincode: string;
}

const GeneralInfo: React.FC<GeneralInfoProps> = ({ hotelId }) => {
    const [hotelData, setHotelData] = useState<HotelData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Fetch hotel data based on hotelId
        const fetchHotelData = async () => {
            try {
                setLoading(true);
                // Replace with your actual API call
                // const response = await fetch(`/api/hotels/${hotelId}`);
                // const data = await response.json();
                // setHotelData(data);

                // Temporary mock data
                setHotelData({
                    name: 'Hotel 1',
                    email: 'hotel1@example.com',
                    contactNumber: '9579365001',
                    plotNo: '1A',
                    streetName: 'Street 1',
                    city: 'City 1',
                    state: 'State 1',
                    pincode: '100001'
                });
            } catch (error) {
                console.error('Error fetching hotel data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchHotelData();
    }, [hotelId]);

    if (loading) {
        return <div className="component-ui-general-info-loading">Loading...</div>;
    }

    if (!hotelData) {
        return <div className="component-ui-general-info-error">Failed to load hotel information</div>;
    }

    return (
        <div className="component-ui-general-info-container">
            <h2 className="component-ui-general-info-title">General Information</h2>
            <div className="component-ui-general-info-grid">
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Name</label>
                    <p className="component-ui-general-info-value">{hotelData.name}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Email</label>
                    <p className="component-ui-general-info-value">{hotelData.email}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Contact Number</label>
                    <p className="component-ui-general-info-value">{hotelData.contactNumber}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Plot No</label>
                    <p className="component-ui-general-info-value">{hotelData.plotNo}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Street Name</label>
                    <p className="component-ui-general-info-value">{hotelData.streetName}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">City</label>
                    <p className="component-ui-general-info-value">{hotelData.city}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">State</label>
                    <p className="component-ui-general-info-value">{hotelData.state}</p>
                </div>
                <div className="component-ui-general-info-field">
                    <label className="component-ui-general-info-label">Pincode</label>
                    <p className="component-ui-general-info-value">{hotelData.pincode}</p>
                </div>
            </div>
        </div>
    );
};

export default GeneralInfo;

