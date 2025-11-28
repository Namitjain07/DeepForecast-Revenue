import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AdminNavbar from "../components/dashboard/AdminNavbar";
import { useAppDispatch } from "../redux/hooks";
import { addHotel } from "../redux/services/api";

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
            const message = err.response?.data?.message  || "Failed to add hotel";
            setError(message);
            console.error("Error adding hotel:", err);
        } finally {
            setLoading(false);
        }
    };

    const InputField = ({ name, placeholder, type = "text", required = false, value }: any) => (
        <input
            type={type}
            name={name}
            placeholder={placeholder}
            value={value}
            onChange={handleChange}
            required={required}
            className="block w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors duration-200 bg-gray-50 focus:bg-white"
        />
    );

    return (
        <div className="min-h-screen bg-gray-50">
            <AdminNavbar role={role} />

            <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <button
                        className="text-gray-500 hover:text-indigo-600 font-medium transition-colors duration-200 flex items-center"
                        onClick={() => navigate(-1)}
                    >
                        <span className="mr-2">←</span> Back to Hotels
                    </button>
                    <h1 className="text-2xl font-bold text-gray-900">Add New Hotel</h1>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-r-lg shadow-sm animate-pulse">
                        <p className="font-medium">Error</p>
                        <p className="text-sm">{error}</p>
                    </div>
                )}

                {/* Stepper */}
                <div className="flex items-center justify-center mb-10">
                    <div className={`flex flex-col items-center ${step >= 1 ? "text-indigo-600" : "text-gray-400"}`}>
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg mb-2 transition-colors duration-300 ${step >= 1 ? "bg-indigo-600 text-white shadow-lg shadow-indigo-200" : "bg-gray-200 text-gray-500"}`}>
                            1
                        </div>
                        <span className="text-sm font-medium">Hotel Information</span>
                    </div>
                    <div className={`w-24 h-1 bg-gray-200 mx-4 rounded-full overflow-hidden`}>
                        <div className={`h-full bg-indigo-600 transition-all duration-500 ease-out ${step === 2 ? "w-full" : "w-0"}`}></div>
                    </div>
                    <div className={`flex flex-col items-center ${step === 2 ? "text-indigo-600" : "text-gray-400"}`}>
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg mb-2 transition-colors duration-300 ${step === 2 ? "bg-indigo-600 text-white shadow-lg shadow-indigo-200" : "bg-gray-200 text-gray-500"}`}>
                            2
                        </div>
                        <span className="text-sm font-medium">Add Staff</span>
                    </div>
                </div>

                {/* Step 1: Hotel Info */}
                {step === 1 && (
                    <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
                        <div className="px-8 py-6 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-gray-100">
                            <h2 className="text-xl font-bold text-gray-800 flex items-center">
                                <span className="mr-2">🏨</span> Hotel Information
                            </h2>
                        </div>
                        
                        <form onSubmit={(e) => e.preventDefault()} className="p-8 space-y-6">
                            {/* Hotel Image Upload */}
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-700">
                                    Hotel Image
                                </label>
                                <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-xl hover:border-indigo-500 transition-colors duration-200 bg-gray-50 hover:bg-indigo-50/30 cursor-pointer group">
                                    {imagePreview ? (
                                        <div className="relative w-full h-64">
                                            <img src={imagePreview} alt="Hotel preview" className="w-full h-full object-cover rounded-lg shadow-md" />
                                            <button
                                                type="button"
                                                className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors shadow-lg"
                                                onClick={() => setImagePreview(null)}
                                            >
                                                ✕
                                            </button>
                                        </div>
                                    ) : (
                                        <label className="space-y-1 text-center cursor-pointer w-full h-full flex flex-col items-center justify-center">
                                            <input
                                                type="file"
                                                accept="image/*"
                                                onChange={handleImageChange}
                                                className="sr-only"
                                            />
                                            <div className="text-5xl mb-3 group-hover:scale-110 transition-transform duration-200">📷</div>
                                            <div className="flex text-sm text-gray-600">
                                                <span className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                                                    Upload a file
                                                </span>
                                                <p className="pl-1">or drag and drop</p>
                                            </div>
                                            <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                                        </label>
                                    )}
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <InputField name="name" placeholder="Hotel Name *" value={formData.name} required />
                                <InputField name="email" placeholder="Hotel Email *" value={formData.email} required type="email" />
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <InputField name="contactNumber" placeholder="Contact Number *" value={formData.contactNumber} required type="tel" />
                                <InputField name="plotNo" placeholder="Plot Number" value={formData.plotNo} />
                            </div>

                            <div>
                                <InputField name="streetName" placeholder="Street Name" value={formData.streetName} />
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <InputField name="city" placeholder="City *" value={formData.city} required />
                                <InputField name="state" placeholder="State *" value={formData.state} required />
                                <InputField name="pincode" placeholder="Pincode *" value={formData.pincode} required />
                            </div>

                            <div className="pt-4 flex justify-end">
                                <button
                                    type="button"
                                    className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transform hover:-translate-y-0.5 transition-all duration-200"
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
                    <form onSubmit={handleSubmit} className="space-y-8">
                        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
                            <div className="px-8 py-6 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-gray-100">
                                <h2 className="text-xl font-bold text-gray-800 flex items-center">
                                    <span className="mr-2">👤</span> Add Owner
                                </h2>
                            </div>
                            <div className="p-8 space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <InputField name="ownerName" placeholder="Owner Name *" value={formData.ownerName} required />
                                    <InputField name="ownerEmail" placeholder="Owner Email *" value={formData.ownerEmail} required type="email" />
                                </div>
                                <div>
                                    <InputField name="ownerPassword" placeholder="Password *" value={formData.ownerPassword} required type="password" />
                                </div>
                            </div>
                        </div>

                        <div className="flex justify-between pt-4">
                            <button
                                type="button"
                                className="inline-flex items-center px-6 py-3 border border-gray-300 shadow-sm text-base font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200"
                                onClick={handleBack}
                                disabled={loading}
                            >
                                ← Back
                            </button>
                            <button
                                type="submit"
                                className={`inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-lg shadow-lg text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transform hover:-translate-y-0.5 transition-all duration-200 ${loading ? 'opacity-75 cursor-not-allowed' : ''}`}
                                disabled={loading}
                            >
                                {loading ? (
                                    <>
                                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Creating...
                                    </>
                                ) : (
                                    "✓ Create Hotel"
                                )}
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
};

export default AddHotel;
