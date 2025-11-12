import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AdminNavbar from "../components/dashboard/AdminNavbar";
import { useAppDispatch } from "../redux/hooks";
import { addHotel } from "../redux/services/api";
import "../stylesheet/pages/page-add-hotel.css";

const AddHotel: React.FC = () => {
    const role =
        (localStorage.getItem("userRole") as "admin" | "owner" | "manager") ||
        "admin";
    const navigate = useNavigate();
    const dispatch = useAppDispatch();

    const [step, setStep] = useState(1);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [formData, setFormData] = useState({
        name: "",
        email: "",
        contactNumber: "",
        plotNo: "",
        streetName: "",
        city: "",
        state: "",
        pincode: "",
        ownerName: "",
        ownerEmail: "",
        ownerPassword: "",
        managerName: "",
        managerEmail: "",
        managerPassword: "",
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            // Create preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleNext = () => {
        if (!formData.name || !formData.email || !formData.city || !formData.state) {
            setError("Please fill all required fields");
            return;
        }
        setError(null);
        setStep(2);
    };

    const handleBack = () => setStep(1);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);

        try {
            // Validate owner fields
            if (!formData.ownerName || !formData.ownerEmail || !formData.ownerPassword) {
                setError('Please fill all owner fields: Name, Email, and Password');
                return;
            }

            setLoading(true);

            // Prepare hotel data with owner information
            const hotelPayload = {
                name: formData.name,
                email: formData.email,
                contactNumber: formData.contactNumber,
                plotNo: formData.plotNo,
                streetName: formData.streetName,
                city: formData.city,
                state: formData.state,
                pincode: formData.pincode,
                imageUrl: imagePreview || undefined, // Send base64 image if available
                ownerName: formData.ownerName,
                ownerEmail: formData.ownerEmail,
                ownerPassword: formData.ownerPassword
            };

            // Call API to add hotel and owner simultaneously
            await dispatch(addHotel(hotelPayload) as any);

            // Show success message
            alert("Hotel and owner created successfully!");

            // Navigate back to hotels page
            navigate("/all-hotels");
        } catch (err: any) {
            const message = err.response?.data?.message || err.message || "Failed to add hotel";
            setError(message);
            console.error("Error adding hotel:", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-add-hotel">
            <AdminNavbar role={role} />

            <div className="page-add-hotel-container">
                {/* Header */}
                <div className="page-add-hotel-header">
                    <button
                        className="page-add-hotel-back-btn"
                        onClick={() => navigate(-1)}
                    >
                        ← Back to Hotels
                    </button>
                    <h1>Add New Hotel</h1>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="page-add-hotel-error-message">
                        {error}
                    </div>
                )}

                {/* Stepper */}
                <div className="page-add-hotel-steps">
                    <div
                        className={`page-add-hotel-step ${
                            step >= 1 ? "active" : ""
                        }`}
                    >
                        <div className="step-circle">1</div>
                        <p>Hotel Information</p>
                    </div>
                    <div className="step-line" />
                    <div
                        className={`page-add-hotel-step ${
                            step === 2 ? "active" : ""
                        }`}
                    >
                        <div className="step-circle">2</div>
                        <p>Add Staff</p>
                    </div>
                </div>

                {/* Step 1: Hotel Info */}
                {step === 1 && (
                    <div className="page-add-hotel-form-card">
                        <h2 className="page-add-hotel-section-title">🏨 Hotel Information</h2>
                        <form onSubmit={(e) => e.preventDefault()} className="page-add-hotel-form">
                            {/* Hotel Image Upload */}
                            <div className="page-add-hotel-image-section">
                                <label className="page-add-hotel-image-label">
                                    Hotel Image
                                </label>
                                <div className="page-add-hotel-image-upload">
                                    {imagePreview ? (
                                        <div className="page-add-hotel-image-preview">
                                            <img src={imagePreview} alt="Hotel preview" />
                                            <button
                                                type="button"
                                                className="page-add-hotel-image-remove-btn"
                                                onClick={() => {
                                                    setImagePreview(null);
                                                }}
                                            >
                                                ✕ Remove
                                            </button>
                                        </div>
                                    ) : (
                                        <label className="page-add-hotel-upload-box">
                                            <input
                                                type="file"
                                                accept="image/*"
                                                onChange={handleImageChange}
                                                style={{ display: "none" }}
                                            />
                                            <div className="upload-placeholder">
                                                <span>📷 Click to upload hotel image</span>
                                                <p>Supported formats: JPG, PNG, WebP</p>
                                            </div>
                                        </label>
                                    )}
                                </div>
                            </div>

                            <div className="page-add-hotel-row">
                                <input
                                    type="text"
                                    name="name"
                                    placeholder="Hotel Name *"
                                    value={formData.name}
                                    onChange={handleChange}
                                    required
                                />
                                <input
                                    type="email"
                                    name="email"
                                    placeholder="Hotel Email *"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required
                                />
                            </div>

                            <div className="page-add-hotel-row">
                                <input
                                    type="tel"
                                    name="contactNumber"
                                    placeholder="Contact Number *"
                                    value={formData.contactNumber}
                                    onChange={handleChange}
                                    required
                                />
                                <input
                                    type="text"
                                    name="plotNo"
                                    placeholder="Plot Number"
                                    value={formData.plotNo}
                                    onChange={handleChange}
                                />
                            </div>

                            <div className="page-add-hotel-row">
                                <input
                                    type="text"
                                    name="streetName"
                                    placeholder="Street Name"
                                    value={formData.streetName}
                                    onChange={handleChange}
                                />
                            </div>

                            <div className="page-add-hotel-row">
                                <input
                                    type="text"
                                    name="city"
                                    placeholder="City *"
                                    value={formData.city}
                                    onChange={handleChange}
                                    required
                                />
                                <input
                                    type="text"
                                    name="state"
                                    placeholder="State *"
                                    value={formData.state}
                                    onChange={handleChange}
                                    required
                                />
                                <input
                                    type="text"
                                    name="pincode"
                                    placeholder="Pincode *"
                                    value={formData.pincode}
                                    onChange={handleChange}
                                    required
                                />
                            </div>

                            <div className="page-add-hotel-actions">
                                <button
                                    type="button"
                                    className="page-add-hotel-next-btn"
                                    onClick={handleNext}
                                >
                                    Next Step →
                                </button>
                            </div>
                        </form>
                    </div>
                )}

                {/* Step 2: Add Staff */}
                {step === 2 && (
                    <form onSubmit={handleSubmit}>
                        <div className="page-add-hotel-form-card">
                            <h2 className="page-add-hotel-section-title">👤 Add Owner</h2>
                            <div className="page-add-hotel-row">
                                <input
                                    type="text"
                                    name="ownerName"
                                    placeholder="Owner Name *"
                                    value={formData.ownerName}
                                    onChange={handleChange}
                                    required
                                />
                                <input
                                    type="email"
                                    name="ownerEmail"
                                    placeholder="Owner Email *"
                                    value={formData.ownerEmail}
                                    onChange={handleChange}
                                    required
                                />
                            </div>
                            <div className="page-add-hotel-row">
                                <input
                                    type="password"
                                    name="ownerPassword"
                                    placeholder="Password *"
                                    value={formData.ownerPassword}
                                    onChange={handleChange}
                                    required
                                />
                            </div>
                        </div>

                        {/*<div className="page-add-hotel-form-card">*/}
                        {/*    <h2 className="page-add-hotel-section-title">*/}
                        {/*        👥 Add Manager (Optional)*/}
                        {/*    </h2>*/}
                        {/*    <div className="page-add-hotel-row">*/}
                        {/*        <input*/}
                        {/*            type="text"*/}
                        {/*            name="managerName"*/}
                        {/*            placeholder="Manager Name"*/}
                        {/*            value={formData.managerName}*/}
                        {/*            onChange={handleChange}*/}
                        {/*        />*/}
                        {/*        <input*/}
                        {/*            type="email"*/}
                        {/*            name="managerEmail"*/}
                        {/*            placeholder="Manager Email"*/}
                        {/*            value={formData.managerEmail}*/}
                        {/*            onChange={handleChange}*/}
                        {/*        />*/}
                        {/*    </div>*/}
                        {/*    <div className="page-add-hotel-row">*/}
                        {/*        <input*/}
                        {/*            type="password"*/}
                        {/*            name="managerPassword"*/}
                        {/*            placeholder="Password"*/}
                        {/*            value={formData.managerPassword}*/}
                        {/*            onChange={handleChange}*/}
                        {/*        />*/}
                        {/*    </div>*/}
                        {/*</div>*/}

                        <div className="page-add-hotel-actions between">
                            <button
                                type="button"
                                className="page-add-hotel-back-btn-outline"
                                onClick={handleBack}
                                disabled={loading}
                            >
                                ← Back
                            </button>
                            <button
                                type="submit"
                                className="page-add-hotel-create-btn"
                                disabled={loading}
                            >
                                {loading ? "Creating..." : "✓ Create Hotel"}
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
};

export default AddHotel;
