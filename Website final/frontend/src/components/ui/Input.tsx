// src/components/Input.tsx
import React from "react";

interface InputProps {
    label: string;
    placeholder: string;
    type: string;
    value: string;
    onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
    className?: string;
}

const Input: React.FC<InputProps> = ({ label, placeholder, type, value, onChange, className = "" }) => {
    return (
        <div className={`flex flex-col space-y-1.5 ${className}`}>
            <label className="text-sm font-medium text-gray-700">
                {label}
            </label>
            <input
                className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm placeholder-gray-400 
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 
                         transition-colors duration-200 text-sm"
                type={type}
                placeholder={placeholder}
                value={value}
                onChange={onChange}
            />
        </div>
    );
};

export default Input;
