// src/pages/Home.tsx
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../stylesheet/pages/page-home.css";
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
        <div className="page-home">
            <div className="page-home__container">
                <div className="page-home__logo">
                    <div className="page-home__icon">🏨</div>
                    <h1>Hotel Revenue Predictor</h1>
                </div>

                <div className="page-home__card">
                    <h2>Welcome Back</h2>

                    <form className="page-home__form" onSubmit={handleSubmit}>
                        {error && <div className="error-message" style={{ color: 'red', marginBottom: '10px' }}>{error}</div>}

                        <Input
                            label="Email"
                            placeholder="Enter your email"
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
                        <Button label={loading ? "Logging in..." : "Login"} type="submit" disabled={loading} />
                    </form>

                    <div className="page-home__demo">
                        <p><strong>Demo Credentials:</strong></p>
                        <p><b>Admin:</b> admin1@example.com / password123</p>
                        <p><b>Owner:</b> owner1@example.com / ownerpass123</p>
                        <p><b>Manager:</b> manager1@example.com / managerpass123</p>
                    </div>
                </div>

                <footer>© 2025 Hotel Revenue Predictor</footer>
            </div>
        </div>
    );
}

export default Home;
