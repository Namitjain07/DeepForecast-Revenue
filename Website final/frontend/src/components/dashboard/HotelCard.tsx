// src/components/HotelCard.tsx
import React from "react";

interface HotelCardProps {
    name: string;
    owner: string;
    city: string;
    contact: string;
    imageUrl?: string; // Optional image URL
    onClick: () => void;
}

const HotelCard: React.FC<HotelCardProps> = ({
    name,
    owner,
    city,
    contact,
    imageUrl,
    onClick,
}) => {
    return (
        <div 
            className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-xl hover:-translate-y-1 transition-all duration-300 cursor-pointer group"
            onClick={onClick}
        >
            <div className="relative h-48 w-full overflow-hidden">
                <img
                    src={imageUrl || '/default-hotel.jpg'}
                    alt={name}
                    className="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-500"
                    onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = '/default-hotel.jpg';
                    }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent opacity-60 group-hover:opacity-80 transition-opacity duration-300" />
                
                <div className="absolute bottom-0 left-0 p-4 w-full">
                    <h3 className="text-xl font-bold text-white mb-1 drop-shadow-md group-hover:text-indigo-200 transition-colors">
                        {name}
                    </h3>
                    <div className="flex items-center text-white/90 text-sm">
                        <span className="mr-1">📍</span> {city}
                    </div>
                </div>
            </div>
            
            <div className="p-5 bg-white relative z-10">
                <div className="space-y-3 text-sm text-gray-600">
                    <div className="flex items-center p-2 rounded-lg bg-gray-50 group-hover:bg-indigo-50 transition-colors duration-200">
                        <span className="font-medium min-w-[4rem] text-gray-500 group-hover:text-indigo-500">Owner</span>
                        <span className="text-gray-800 font-medium ml-2">{owner}</span>
                    </div>
                    <div className="flex items-center p-2 rounded-lg bg-gray-50 group-hover:bg-indigo-50 transition-colors duration-200">
                        <span className="font-medium min-w-[4rem] text-gray-500 group-hover:text-indigo-500">Contact</span>
                        <span className="text-gray-800 font-medium ml-2">{contact}</span>
                    </div>
                </div>
                
                <div className="mt-4 pt-4 border-t border-gray-100 flex justify-end">
                    <span className="text-indigo-600 text-sm font-semibold group-hover:translate-x-1 transition-transform duration-200 flex items-center">
                        View Details 
                        <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </span>
                </div>
            </div>
        </div>
    );
};

export default HotelCard;
