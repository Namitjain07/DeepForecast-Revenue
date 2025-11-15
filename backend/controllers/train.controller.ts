import { Request, Response } from 'express';
import LastTrain from '../models/Lasttrain';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

/**
 * Add the last train record
 */
export const addLastTrain = async (req: Request, res: Response) => {
    try {
        const { userId, hotelId } = req.body;

        if (!userId) {
            throw new ApiError('User ID is required', 400);
        }

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Create new last train record
        const lastTrain = new LastTrain({
            hotelId,
            userId,
            startDateTime: new Date(),
            endDateTime: new Date(),
            status: 'queued'
        });

        const savedLastTrain = await lastTrain.save();

        // TODO: Call another API here as needed

        res.status(201).json({
            message: 'Last train record created successfully',
            lastTrain: {
                id: savedLastTrain._id,
                hotelId: savedLastTrain.hotelId,
                userId: savedLastTrain.userId,
                startDateTime: savedLastTrain.startDateTime,
                endDateTime: savedLastTrain.endDateTime,
                status: savedLastTrain.status,
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get last train record by hotel ID
 */
export const getLastTrainByHotel = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const lastTrain = await LastTrain.findOne({ hotelId })
            .sort({ createdAt: -1 });

        if (!lastTrain) {
            throw new ApiError('No train record found for this hotel', 404);
        }

        res.status(200).json({
            message: 'Last train record retrieved successfully',
            lastTrain: {
                id: lastTrain._id,
                hotelId: lastTrain.hotelId,
                userId: lastTrain.userId,
                startDateTime: lastTrain.startDateTime,
                endDateTime: lastTrain.endDateTime,
                status: lastTrain.status,
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

