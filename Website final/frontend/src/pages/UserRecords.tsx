// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\pages\UserRecords.tsx
// import React from 'react';
import { useParams } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
// ...existing code...
import UserNavbar from '../components/dashboard/UserNavbar';
import RecentRecords from '../components/IndividualHotelPage/RecentRecords';
import CSVUploader from '../components/general/CSVUploader.tsx';
import DownloadRecordCSV from '../components/IndividualHotelPage/DownloadRecordCSV';
import { fetchRecentRecords } from '../redux/services';
import '../stylesheet/pages/page-user-records.css';
import type {AppDispatch, RootState} from "../redux/store.ts";

function UserRecords() {
    const { hotelId } = useParams<{ hotelId: string }>();
    const dispatch = useDispatch<AppDispatch>();
    const { user } = useSelector((state: RootState) => state.auth);
    const role = user?.role as 'owner' | 'manager' || 'manager';

    const handleUploadSuccess = () => {
        // Refresh records after successful upload
        if (hotelId) {
            dispatch(fetchRecentRecords(hotelId) as any);
        }
    };

    return (
        <div>
            <UserNavbar role={role} hotelId={hotelId} />
            <div className="page-user-records">
                <div className="page-user-records-header">
                    <h1>Records</h1>
                    <p>View all hotel records and operational data.</p>
                </div>


                {/* CSV Uploader Section */}
                {hotelId && (
                    <CSVUploader hotelId={hotelId} onSuccess={handleUploadSuccess}/>
                )}
                <div className="page-user-records-container">
                    <div className="page-user-records-content">
                        {hotelId && (
                            <DownloadRecordCSV hotelId={hotelId}/>
                        )}
                    </div>
                </div>
                {/* Download Records CSV Section */}


                {/* Recent Records Section */}
                <div className="page-user-records-container">
                    {/*<h2 className="page-user-records-title">Recent Records</h2>*/}
                    <div className="page-user-records-content">
                        {hotelId && <RecentRecords hotelId={hotelId}/>}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UserRecords;
