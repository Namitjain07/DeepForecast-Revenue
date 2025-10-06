// src/components/Input.tsx
import React from "react";
import "../../stylesheet/ui/components-ui-input.css";

interface InputProps {
    label: string;
    placeholder: string;
    type: string;
    value: string;
    onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

const Input: React.FC<InputProps> = ({ label, placeholder, type, value, onChange }) => {
    return (
        <div className="components-ui-input">
            <label>{label}</label>
            <input
                type={type}
                placeholder={placeholder}
                value={value}
                onChange={onChange}
            />
        </div>
    );
};

export default Input;
