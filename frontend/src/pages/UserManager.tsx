// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\pages\UserManager.tsx
import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../redux/store';
import UserNavbar from '../components/dashboard/UserNavbar';
import { fetchUsersByHotel } from '../redux/services';
import '../stylesheet/pages/page-user-manager.css';
import UserTable from "../components/IndividualHotelPage/UserTable.tsx";

function UserManager() {
    const { hotelId } = useParams<{ hotelId: string }>();
    const dispatch = useDispatch<AppDispatch>();
    const { user } = useSelector((state: RootState) => state.auth);
    const role = user?.role as 'owner' | 'manager' || 'manager';

    useEffect(() => {
        if (hotelId && role === 'owner') {
            dispatch(fetchUsersByHotel(hotelId) as any);
        }
    }, [hotelId, role, dispatch]);

    if (role !== 'owner') {
        return (
            <div>
                <UserNavbar role={role} hotelId={hotelId} />
                <div className="page-user-manager">
                    <div className="page-user-manager-header">
                        <h1>Access Denied</h1>
                        <p>Only hotel owners can manage users.</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div>
            <UserNavbar role={role} hotelId={hotelId} />
            <div className="page-user-manager">

                {hotelId && <UserTable hotelId={hotelId} />}
            </div>
        </div>
    );
}

export default UserManager;
