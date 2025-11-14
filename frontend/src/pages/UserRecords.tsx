// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\pages\UserRecords.tsx
// import React from 'react';
import { useParams } from 'react-router-dom';
import UserNavbar from '../components/dashboard/UserNavbar';
import RecentRecords from '../components/IndividualHotelPage/RecentRecords';
import { useSelector } from 'react-redux';
import type { RootState } from '../redux/store';
import '../stylesheet/pages/page-user-records.css';

function UserRecords() {
    const { hotelId } = useParams<{ hotelId: string }>();
    const { user } = useSelector((state: RootState) => state.auth);
    const role = user?.role as 'owner' | 'manager' || 'manager';

    return (
        <div>
            <UserNavbar role={role} hotelId={hotelId} />
            <div className="page-user-records">
                <div className="page-user-records-header">
                    <h1>Records</h1>
                    <p>View all hotel records and operational data.</p>
                </div>
                <div className="page-user-records-container">
                    <h2 className="page-user-records-title">Recent Records</h2>
                    <div className="page-user-records-content">
                        <RecentRecords hotelId={hotelId} />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UserRecords;
