import { Request, Response } from 'express';
import User from '../models/User';
import * as bcrypt from 'bcryptjs';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

/**
 * Get all users for a specific hotel
 */
export const getUsersByHotel = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const users = await User.find({ hotelId }).select('-password');

        if (!users || users.length === 0) {
            throw new ApiError('No users found for this hotel', 404);
        }

        const formattedUsers = users.map(user => ({
            id: user._id,
            name: user.name,
            email: user.email,
            role: user.role
        }));

        res.status(200).json({
            message: 'Users retrieved successfully',
            count: formattedUsers.length,
            users: formattedUsers
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get user by ID
 */
export const getUserById = async (req: Request, res: Response) => {
    try {
        const { userId } = req.params;

        if (!userId) {
            throw new ApiError('User ID is required', 400);
        }

        const user = await User.findById(userId).select('-password');

        if (!user) {
            throw new ApiError('User not found', 404);
        }

        res.status(200).json({
            message: 'User retrieved successfully',
            user: {
                id: user._id,
                name: user.name,
                email: user.email,
                role: user.role,
                hotelId: user.hotelId
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
 * Update user info
 */
export const updateUser = async (req: Request, res: Response) => {
    try {
        const { userId } = req.params;
        // @ts-ignore
        const currentUser = req.user;
        const { name, email, password } = req.body;

        if (!userId) {
            throw new ApiError('User ID is required', 400);
        }

        const user = await User.findById(userId);
        if (!user) {
            throw new ApiError('User not found', 404);
        }

        // Check authorization: user can update their own info or an admin can update any user
        if (currentUser._id.toString() !== userId && currentUser.role !== 'admin') {
            throw new ApiError('You are not authorized to update this user', 403);
        }

        // Update allowed fields
        if (name) user.name = name;
        if (email) {
            // Check if email is already taken by another user
            const existingUser = await User.findOne({ email, _id: { $ne: userId } });
            if (existingUser) {
                throw new ApiError('Email already in use', 400);
            }
            user.email = email;
        }
        if (password) {
            const salt = await bcrypt.genSalt(10);
            user.password = await bcrypt.hash(password, salt);
        }

        await user.save();

        res.status(200).json({
            message: 'User updated successfully',
            user: {
                id: user._id,
                name: user.name,
                email: user.email,
                role: user.role
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
 * Delete user
 */
export const deleteUser = async (req: Request, res: Response) => {
    try {
        const { userId } = req.params;
        // @ts-ignore
        const currentUser = req.user;

        if (!userId) {
            throw new ApiError('User ID is required', 400);
        }

        const user = await User.findById(userId);
        if (!user) {
            throw new ApiError('User not found', 404);
        }

        // Check authorization: user can delete their own account or an admin can delete any user
        if (currentUser._id.toString() !== userId && currentUser.role !== 'admin') {
            throw new ApiError('You are not authorized to delete this user', 403);
        }

        await User.findByIdAndDelete(userId);

        res.status(200).json({
            message: 'User deleted successfully',
            deletedUserId: userId
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

