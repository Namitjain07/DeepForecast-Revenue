// src/components/StatCard.tsx
import React from "react";
import "../../stylesheet/ui/components-ui-statcard.css";

interface StatCardProps {
    title: string;
    value: string | number;
    description: string;
    icon: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, description, icon }) => {
    return (
        <div className="components-ui-statcard">
            <div className="statcard-header">
                <span className="statcard-title">{title}</span>
                <span className="statcard-icon">{icon}</span>
            </div>
            <div className="statcard-value">{value}</div>
            <div className="statcard-desc">{description}</div>
        </div>
    );
};

export default StatCard;
