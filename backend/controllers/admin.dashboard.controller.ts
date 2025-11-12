import { Request, Response } from 'express';
import Hotel from '../models/Hotel';
import User from '../models/User';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

export const getDashboardStats = async (req: Request, res: Response) => {
    try {
        // Get total number of hotels
        const totalHotels = await Hotel.countDocuments();

        // Get total number of users
        const totalUsers = await User.countDocuments();

        // Get recently added hotels (last 7 days)
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        const recentHotels = await Hotel.countDocuments({
            createdAt: { $gte: sevenDaysAgo }
        });

        // Get recently added users (last 7 days)
        const recentUsers = await User.countDocuments({
            createdAt: { $gte: sevenDaysAgo }
        });

        res.status(200).json({
            message: 'Dashboard statistics retrieved successfully',
            stats: {
                totalHotels,
                totalUsers,
                recentHotels,
                recentUsers,
                lastUpdated: new Date()
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

