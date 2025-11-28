// src/components/StatCard.tsx
import React from "react";

interface StatCardProps {
    title: string;
    value: string | number;
    description: string;
    icon: React.ReactNode;
    color?: "indigo" | "green" | "blue" | "purple" | "pink" | "orange";
}

const StatCard: React.FC<StatCardProps> = ({ title, value, description, icon, color = "indigo" }) => {
    const colorStyles = {
        indigo: { bg: "bg-indigo-50", text: "text-indigo-600", border: "border-indigo-100" },
        green: { bg: "bg-green-50", text: "text-green-600", border: "border-green-100" },
        blue: { bg: "bg-blue-50", text: "text-blue-600", border: "border-blue-100" },
        purple: { bg: "bg-purple-50", text: "text-purple-600", border: "border-purple-100" },
        pink: { bg: "bg-pink-50", text: "text-pink-600", border: "border-pink-100" },
        orange: { bg: "bg-orange-50", text: "text-orange-600", border: "border-orange-100" },
    };

    const currentStyle = colorStyles[color];

    return (
        <div className={`bg-white overflow-hidden rounded-xl shadow-sm border ${currentStyle.border} hover:shadow-lg hover:-translate-y-1 transition-all duration-300`}>
            <div className="p-5">
                <div className="flex items-center">
                    <div className="flex-shrink-0">
                        <div className={`flex items-center justify-center h-12 w-12 rounded-lg ${currentStyle.bg} ${currentStyle.text} text-2xl shadow-sm`}>
                            {icon}
                        </div>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                        <dl>
                            <dt className="text-sm font-medium text-gray-500 truncate uppercase tracking-wide">
                                {title}
                            </dt>
                            <dd>
                                <div className="text-2xl font-bold text-gray-900 mt-1">
                                    {value}
                                </div>
                            </dd>
                        </dl>
                    </div>
                </div>
            </div>
            <div className={`bg-gray-50 px-5 py-3 border-t ${currentStyle.border}`}>
                <div className="text-sm flex items-center">
                    <span className={`${currentStyle.text} font-medium mr-2`}>
                        Trend
                    </span>
                    <span className="text-gray-500">
                        {description}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default StatCard;
