import { Request, Response } from 'express';
import axios from 'axios';
import LastTrain from '../models/Lasttrain';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

/**
 * Add the last train record and trigger Python backend training
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

        console.log(`🚀 Starting training request for hotel: ${hotelId}, user: ${userId}`);

        // Call Python FastAPI Backend_2 to start training
        try {
            const fastApiResponse = await axios.post(
                `${FASTAPI_URL}/api/v1/train/start`,
                {
                    userId,
                    hotelId
                },
                {
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    timeout: 10000 // 10 second timeout
                }
            );

            console.log('✅ FastAPI training job queued:', fastApiResponse.data);

            // Return the response from FastAPI
            res.status(201).json({
                message: fastApiResponse.data.message || 'Training job queued successfully',
                job_id: fastApiResponse.data.job_id,
                train_id: fastApiResponse.data.train_id,
                lastTrain: fastApiResponse.data.lastTrain
            });

        } catch (fastApiError: any) {
            console.error('❌ FastAPI call failed:', fastApiError.response?.data || fastApiError.message);
            
            // If FastAPI is unreachable, throw a meaningful error
            if (fastApiError.code === 'ECONNREFUSED') {
                throw new ApiError('Training service is unavailable. Please ensure Backend_2 (FastAPI) is running.', 503);
            }
            
            throw new ApiError(
                fastApiError.response?.data?.detail || 'Failed to start training job',
                fastApiError.response?.status || 500
            );
        }

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

