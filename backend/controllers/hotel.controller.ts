import { Request, Response } from 'express';
import Hotel from '../models/Hotel';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

export const addHotel = async (req: Request, res: Response) => {
    try {
        const {
            name,
            email,
            contactNumber,
            plotNo,
            streetName,
            city,
            state,
            pincode,
            imageUrl
        } = req.body;

        // Check for required fields
        const requiredFields = [
            'name',
            'email',
            'contactNumber',
            'plotNo',
            'streetName',
            'city',
            'state',
            'pincode'
        ];

        const missingFields = requiredFields.filter(field => !req.body[field]);
        if (missingFields.length > 0) {
            throw new ApiError(`Missing required fields: ${missingFields.join(', ')}`, 400);
        }

        // Get admin ID from authenticated user
        // @ts-ignore
        const adminId = req.user._id;

        // Check if hotel with same email exists
        const hotelExists = await Hotel.findOne({ email });
        if (hotelExists) {
            throw new ApiError('Hotel with this email already exists', 400);
        }

        // Create hotel
        const hotel = await Hotel.create({
            adminId,
            name,
            email,
            contactNumber,
            plotNo,
            streetName,
            city,
            state,
            pincode,
            imageUrl
        });

        res.status(201).json({
            message: 'Hotel added successfully',
            hotel: {
                id: hotel._id,
                name: hotel.name,
                email: hotel.email,
                city: hotel.city,
                state: hotel.state,
                imageUrl: hotel.imageUrl
            }
        });

    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};
