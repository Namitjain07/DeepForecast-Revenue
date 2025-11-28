// src/components/Button.tsx
import React from "react";

interface ButtonProps {
    label: string;
    onClick?: () => void;
    type?: "button" | "submit" | "reset";
    disabled?: boolean;
    variant?: "primary" | "secondary" | "outline";
    className?: string;
}

const Button: React.FC<ButtonProps> = ({
    label,
    onClick,
    type = "button",
    disabled = false,
    variant = "primary",
    className = ""
}) => {
    const baseStyles = "w-full px-4 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";
    
    const variants = {
        primary: "bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-500 shadow-sm hover:shadow-md",
        secondary: "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50 focus:ring-indigo-500 shadow-sm",
        outline: "bg-transparent text-indigo-600 border border-indigo-600 hover:bg-indigo-50 focus:ring-indigo-500"
    };

    return (
        <button
            className={`${baseStyles} ${variants[variant]} ${className}`}
            onClick={onClick}
            type={type}
            disabled={disabled}
        >
            {label}
        </button>
    );
};

export default Button;
