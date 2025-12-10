// src/components/dashboard/UserNavbar.tsx
import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import type { RootState, AppDispatch } from "../../redux/store";
import { logout } from "../../redux/slices/authSlice";
import { addLastTrainRecord } from "../../redux/services/modelTrainAPI";
import axios from "axios";

interface UserNavbarProps {
    role: "owner" | "manager";
    hotelId?: string;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';
const getToken = () => localStorage.getItem('token');

const UserNavbar: React.FC<UserNavbarProps> = ({ role, hotelId }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const dispatch = useDispatch<AppDispatch>();
    const { user } = useSelector((state: RootState) => state.auth);
    const { modelTrain } = useSelector((state: RootState) => state);
    const [activeTab, setActiveTab] = useState("Dashboard");
    const [showProfileDialog, setShowProfileDialog] = useState(false);
    const [trainMessage, setTrainMessage] = useState<string | null>(null);
    const [isPolling, setIsPolling] = useState(false);
    const profileRef = useRef<HTMLDivElement>(null);
    const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const handleNavigation = (tab: string, path: string) => {
        setActiveTab(tab);
        navigate(path);
    };

    const handleLogout = () => {
        dispatch(logout());
        setShowProfileDialog(false);
        navigate("/");
    };

    const handleRetrainModel = async () => {
        // @ts-ignore
        if (!user?.id || !hotelId) {
            setTrainMessage("Error: User ID or Hotel ID is missing");
            setTimeout(() => setTrainMessage(null), 3000);
            return;
        }

        try {
            const response = await dispatch(
                // @ts-ignore
                addLastTrainRecord(user.id, hotelId) as any
            );
            if (response && response.train_id) {
                setTrainMessage("✓ Model retraining started! Checking status...");
                setIsPolling(true);
                
                // Start polling for training completion
                pollTrainingStatus(response.train_id);
            }
        } catch (error: any) {
            setTrainMessage(`✕ ${error.response?.data?.message || 'Failed to start retraining'}`);
            setTimeout(() => setTrainMessage(null), 3000);
        }
    };

    // Poll training status and refresh page when complete
    const pollTrainingStatus = async (trainId: string) => {
        let pollCount = 0;
        const maxPolls = 60; // Poll for up to 5 minutes (60 * 5 seconds)

        const checkStatus = async () => {
            try {
                const response = await axios.get(
                    `${API_URL}/train/${hotelId}`,
                    {
                        headers: {
                            'Authorization': `Bearer ${getToken()}`,
                            'Content-Type': 'application/json',
                        },
                    }
                );

                const lastTrain = response.data?.lastTrain;
                
                if (lastTrain && lastTrain.id === trainId) {
                    if (lastTrain.status === 'success') {
                        setTrainMessage("✓ Training completed! Refreshing forecasts...");
                        setIsPolling(false);
                        
                        // Clear interval
                        if (pollIntervalRef.current) {
                            clearInterval(pollIntervalRef.current);
                            pollIntervalRef.current = null;
                        }
                        
                        // Reload page after 2 seconds to fetch new forecasts
                        setTimeout(() => {
                            window.location.reload();
                        }, 2000);
                        return;
                    } else if (lastTrain.status === 'failure') {
                        setTrainMessage("✕ Training failed. Please try again.");
                        setIsPolling(false);
                        
                        // Clear interval
                        if (pollIntervalRef.current) {
                            clearInterval(pollIntervalRef.current);
                            pollIntervalRef.current = null;
                        }
                        
                        setTimeout(() => setTrainMessage(null), 5000);
                        return;
                    } else if (lastTrain.status === 'running') {
                        setTrainMessage("⏳ Training in progress... Please wait.");
                    }
                }

                pollCount++;
                
                // Stop polling after max attempts
                if (pollCount >= maxPolls) {
                    setTrainMessage("⚠️ Training is taking longer than expected. Check back later.");
                    setIsPolling(false);
                    
                    if (pollIntervalRef.current) {
                        clearInterval(pollIntervalRef.current);
                        pollIntervalRef.current = null;
                    }
                    
                    setTimeout(() => setTrainMessage(null), 5000);
                }
            } catch (error) {
                console.error('Error polling training status:', error);
            }
        };

        // Start polling every 5 seconds
        pollIntervalRef.current = setInterval(checkStatus, 5000);
        
        // Do initial check immediately
        checkStatus();
    };

    // Cleanup interval on unmount
    useEffect(() => {
        return () => {
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
            }
        };
    }, []);

    // Automatically update active tab when user navigates manually
    useEffect(() => {
        if (location.pathname.includes("dashboard")) setActiveTab("Dashboard");
        else if (location.pathname.includes("hotel-info")) setActiveTab("Hotel Info");
        else if (location.pathname.includes("forecast")) setActiveTab("Forecast");
        else if (location.pathname.includes("manager")) setActiveTab("Manager");
        else if (location.pathname.includes("records")) setActiveTab("Records");
    }, [location.pathname]);

    // Close profile dialog when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
                setShowProfileDialog(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [profileRef]);

    // Get first letter of name for avatar
    const getInitial = () => {
        return user?.name ? user.name.charAt(0).toUpperCase() : "U";
    };

    const navItems = [
        { name: "Dashboard", path: `/user-dashboard`, show: true },
        { name: "Hotel Info", path: `/hotel-info/${hotelId}`, show: true },
        { name: "Forecast", path: `/forecast/${hotelId}`, show: true },
        { name: "Manager", path: `/manager/${hotelId}`, show: role === "owner" },
        { name: "Records", path: `/records/${hotelId}`, show: true },
    ];

    return (
        <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50 transition-all duration-300">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    <div className="flex">
                        <div className="flex-shrink-0 flex items-center cursor-pointer group" onClick={() => navigate('/')}>
                            <span className="text-2xl mr-2 transform group-hover:scale-110 transition-transform duration-200">🏨</span>
                            <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 hidden md:block">Revenue Predictor</h2>
                        </div>
                        <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                            {navItems.filter(item => item.show).map((item) => (
                                <button
                                    key={item.name}
                                    onClick={() => handleNavigation(item.name, item.path)}
                                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-all duration-200 ${
                                        activeTab === item.name
                                            ? "border-indigo-500 text-indigo-600"
                                            : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 hover:-translate-y-0.5"
                                    }`}
                                >
                                    {item.name}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="flex items-center space-x-4">
                        <div className="relative">
                            <button
                                onClick={handleRetrainModel}
                                disabled={modelTrain.loading || isPolling}
                                className={`inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white ${
                                    (modelTrain.loading || isPolling) ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'
                                } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors`}
                                title={(modelTrain.loading || isPolling) ? "Model is retraining..." : "Click to retrain the model"}
                            >
                                {(modelTrain.loading || isPolling) ? "⏳ Retraining..." : "🔄 Retrain Model"}
                            </button>
                            
                            {trainMessage && (
                                <div className={`absolute top-full mt-2 right-0 w-64 p-2 rounded-md text-xs font-medium shadow-lg z-50 ${
                                    trainMessage.startsWith('✓') || trainMessage.startsWith('⏳') ? 'bg-green-50 text-green-700 border border-green-200' : 
                                    trainMessage.startsWith('⚠️') ? 'bg-yellow-50 text-yellow-700 border border-yellow-200' :
                                    'bg-red-50 text-red-700 border border-red-200'
                                }`}>
                                    {trainMessage}
                                </div>
                            )}
                        </div>

                        <div className="relative" ref={profileRef}>
                            <div>
                                <button
                                    onClick={() => setShowProfileDialog(!showProfileDialog)}
                                    className="bg-indigo-100 flex text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 items-center justify-center w-10 h-10 text-indigo-700 font-bold hover:bg-indigo-200 transition-colors"
                                >
                                    {getInitial()}
                                </button>
                            </div>

                            {showProfileDialog && (
                                <div className="origin-top-right absolute right-0 mt-2 w-64 rounded-md shadow-lg py-1 bg-white ring-1 ring-black ring-opacity-5 focus:outline-none transform transition-all duration-200 ease-out z-50">
                                    <div className="px-4 py-3 border-b border-gray-100">
                                        <div className="flex items-center">
                                            <div className="flex-shrink-0 h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold text-lg">
                                                {getInitial()}
                                            </div>
                                            <div className="ml-3">
                                                <p className="text-sm font-medium text-gray-900 truncate">{user?.name || "User"}</p>
                                                <p className="text-xs text-gray-500 truncate">{user?.role || role}</p>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div className="px-4 py-2">
                                        <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Account</p>
                                        <p className="text-sm text-gray-700 truncate mb-1">{user?.email || "N/A"}</p>
                                    </div>

                                    <div className="border-t border-gray-100">
                                        <button
                                            onClick={handleLogout}
                                            className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                                        >
                                            Sign out
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default UserNavbar;

