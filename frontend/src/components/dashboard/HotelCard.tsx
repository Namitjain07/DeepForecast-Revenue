// src/components/HotelCard.tsx
import React from "react";
import "../../stylesheet/ui/components-ui-hotelcard.css";

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
        <div className="components-ui-hotelcard" onClick={onClick}>
            <div className="hotel-image-container">
                <img
                    src={imageUrl || '/default-hotel.jpg'}
                    alt={name}
                    className="hotel-image"
                    onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = '/default-hotel.jpg';
                    }}
                />
            </div>
            <div className="hotel-details">
                <h3 className="hotel-name">{name}</h3>
                <p><b>Owner:</b> {owner}</p>
                <p><b>City:</b> {city}</p>
                <p><b>Contact:</b> {contact}</p>
            </div>
        </div>
    );
};

export default HotelCard;
