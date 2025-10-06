// src/components/Button.tsx
import React from "react";
import "../../stylesheet/ui/components-ui-button.css";

interface ButtonProps {
    label: string;
    onClick?: () => void;
    type?: "button" | "submit" | "reset";
    disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({
    label,
    onClick,
    type = "button",
    disabled = false
}) => {
    return (
        <button
            className="components-ui-button"
            onClick={onClick}
            type={type}
            disabled={disabled}
        >
            {label}
        </button>
    );
};

export default Button;
