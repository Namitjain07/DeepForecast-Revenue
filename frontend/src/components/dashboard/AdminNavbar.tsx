// src/components/AdminNavbar.tsx
import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import type { RootState } from "../../redux/store";
import { logout } from "../../redux/slices/authSlice";

interface NavbarProps {
    role: "admin" | "owner" | "manager";
}

const AdminNavbar: React.FC<NavbarProps> = ({ role }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const dispatch = useDispatch();
    const { user } = useSelector((state: RootState) => state.auth);
    const [activeTab, setActiveTab] = useState("Dashboard");
    const [showProfileDialog, setShowProfileDialog] = useState(false);
    const profileRef = useRef<HTMLDivElement>(null);

    const handleNavigation = (tab: string, path: string) => {
        setActiveTab(tab);
        navigate(path);
    };

    const handleLogout = () => {
        dispatch(logout());
        setShowProfileDialog(false);
        navigate("/");
    };

    // Automatically update active tab when user navigates manually
    useEffect(() => {
        if (location.pathname.includes("dashboard")) setActiveTab("Dashboard");
        else if (location.pathname.includes("all-hotels")) setActiveTab("All Hotels");
        else if (location.pathname.includes("add-hotel")) setActiveTab("Add Hotel");
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
        return user?.name ? user.name.charAt(0).toUpperCase() : "A";
    };

    const navItems = [
        { name: "Dashboard", path: `/${role}-dashboard`, show: true },
        { name: "All Hotels", path: "/all-hotels", show: role === "admin" },
        { name: "Add Hotel", path: "/add-hotel", show: role === "admin" },
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

                    <div className="flex items-center">
                        {role === "owner" && (
                            <button className="mr-4 px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors">
                                Retrain Model
                            </button>
                        )}
                        
                        <div className="ml-3 relative" ref={profileRef}>
                            <div>
                                <button
                                    onClick={() => setShowProfileDialog(!showProfileDialog)}
                                    className="bg-indigo-100 flex text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 items-center justify-center w-10 h-10 text-indigo-700 font-bold hover:bg-indigo-200 transition-colors"
                                >
                                    {getInitial()}
                                </button>
                            </div>

                            {showProfileDialog && (
                                <div className="origin-top-right absolute right-0 mt-2 w-64 rounded-md shadow-lg py-1 bg-white ring-1 ring-black ring-opacity-5 focus:outline-none transform transition-all duration-200 ease-out">
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

export default AdminNavbar;
