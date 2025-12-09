// src/pages/Home.tsx
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Input from "../components/ui/Input";
import Button from "../components/ui/Button";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { loginUser } from "../redux/services/api";

function Home() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");

    const dispatch = useAppDispatch();
    const navigate = useNavigate();
    const { user, isAuthenticated, loading } = useAppSelector((state) => state.auth);

    useEffect(() => {
        if (isAuthenticated && user) {
            switch (user.role) {
                case "admin":
                    navigate("/admin-dashboard");
                    break;
                case "owner":
                    navigate("/user-dashboard");
                    break;
                case "manager":
                    navigate("/user-dashboard");
                    break;
                default:
                    setError("Invalid role");
            }
        }
    }, [isAuthenticated, user, navigate]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");

        try {
            await dispatch(loginUser(email, password));
        } catch (err: any) {
            setError(err.response?.data?.message || "Login failed");
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 p-4 relative overflow-hidden">
            {/* Decorative background elements */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0">
                <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
                <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
            </div>

            <div className="w-full max-w-md relative z-10">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-white/20 backdrop-blur-md shadow-xl mb-6 text-4xl border border-white/30 transform hover:scale-105 transition-transform duration-300">
                        🏨
                    </div>
                    <h1 className="text-4xl font-bold text-white tracking-tight drop-shadow-md">
                        Revenue Predictor
                    </h1>
                    <p className="mt-3 text-indigo-100 text-lg font-medium">
                        Sign in to access your dashboard
                    </p>
                </div>

                <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8 space-y-6 border border-white/50">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-2xl font-bold text-gray-800">Welcome Back</h2>
                        <div className="h-1 w-10 bg-indigo-500 rounded-full"></div>
                    </div>

                    <form className="space-y-5" onSubmit={handleSubmit}>
                        {error && (
                            <div className="p-4 rounded-xl bg-red-50 border border-red-100 text-red-600 text-sm font-medium flex items-center animate-pulse">
                                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                {error}
                            </div>
                        )}

                        <Input
                            label="Email Address"
                            placeholder="name@company.com"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                        <Input
                            label="Password"
                            placeholder="Enter your password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                        
                        <div className="pt-4">
                            <Button 
                                label={loading ? "Signing in..." : "Sign In"} 
                                type="submit" 
                                disabled={loading}
                                variant="primary"
                                className="w-full py-3 text-lg shadow-lg hover:shadow-indigo-500/30 transform hover:-translate-y-0.5 transition-all duration-200"
                            />
                        </div>
                    </form>

                    <div className="pt-6 border-t border-gray-100">
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4 text-center">
                            Demo Credentials
                        </p>
                        <div className="space-y-3 text-sm">
                            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 hover:bg-indigo-50 transition-colors border border-gray-100 cursor-pointer group">
                                <span className="font-semibold text-gray-700 group-hover:text-indigo-700">Admin</span>
                                <div className="text-right">
                                    <div className="font-mono text-gray-500 text-xs bg-white px-2 py-1 rounded border border-gray-200 mb-1">admin1@example.com</div>
                                    <div className="font-mono text-gray-400 text-[10px]">Pass: password123</div>
                                </div>
                            </div>
                            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 hover:bg-indigo-50 transition-colors border border-gray-100 cursor-pointer group">
                                <span className="font-semibold text-gray-700 group-hover:text-indigo-700">Owner</span>
                                <div className="text-right">
                                    <div className="font-mono text-gray-500 text-xs bg-white px-2 py-1 rounded border border-gray-200 mb-1">owner1@example.com</div>
                                    <div className="font-mono text-gray-400 text-[10px]">Pass: ownerpass123</div>
                                </div>
                            </div>
                            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 hover:bg-indigo-50 transition-colors border border-gray-100 cursor-pointer group">
                                <span className="font-semibold text-gray-700 group-hover:text-indigo-700">Manager</span>
                                <div className="text-right">
                                    <div className="font-mono text-gray-500 text-xs bg-white px-2 py-1 rounded border border-gray-200 mb-1">manager1@example.com</div>
                                    <div className="font-mono text-gray-400 text-[10px]">Pass: managerpass123</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <footer className="mt-8 text-center text-sm text-indigo-100 font-medium opacity-80">
                    © 2025 Hotel Revenue Predictor. All rights reserved.
                </footer>
            </div>
        </div>
    );
}

export default Home;
