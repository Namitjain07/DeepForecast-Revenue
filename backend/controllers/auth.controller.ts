import { Request, Response } from 'express';
// @ts-ignore
import bcrypt from 'bcryptjs';
import User from '../models/User';
import Admin from '../models/Admin';
import { generateToken, hashPassword } from '../helpers/auth.helpers';
import { recordFailedLoginAttempt, recordSuccessfulLogin } from '../security/bruteForceProtection';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

// Login handler for both users and admins
export const login = async (req: Request, res: Response) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            throw new ApiError('Please provide email and password', 400);
        }

        // First check if it's an admin
        const admin = await Admin.findOne({ email });
        if (admin) {
            const isMatch = await bcrypt.compare(password, admin.password);
            if (!isMatch) {
                // Record failed attempt
                recordFailedLoginAttempt(email);
                throw new ApiError('Invalid credentials', 401);
            }

            // Record successful login (reset attempt counter)
            recordSuccessfulLogin(email);

            return res.status(200).json({
                message: 'Login successful',
                token: generateToken(admin._id.toString(), 'admin'),
                user: {
                    id: admin._id,
                    name: admin.name,
                    email: admin.email,
                    role: 'admin',
                    imageUrl: admin.imageUrl
                }
            });
        }

        // If not an admin, check if it's a user
        const user = await User.findOne({ email });
        if (!user) {
            // Record failed attempt
            recordFailedLoginAttempt(email);
            throw new ApiError('Invalid credentials', 401);
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            // Record failed attempt
            recordFailedLoginAttempt(email);
            throw new ApiError('Invalid credentials', 401);
        }

        // Record successful login (reset attempt counter)
        recordSuccessfulLogin(email);

        return res.status(200).json({
            message: 'Login successful',
            token: generateToken(user._id.toString(), user.role),
            user: {
                id: user._id,
                hotelId: user.hotelId,
                name: user.name,
                email: user.email,
                role: user.role,
                imageUrl: user.imageUrl
            }
        });

    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

// Add new user - Only admin can add owner, admin/owner can add manager
export const addUser = async (req: Request, res: Response) => {
    try {
        const { name, email, password, hotelId, role, imageUrl } = req.body;

        if (!name || !email || !password || !hotelId || !role) {
            throw new ApiError('Please provide all required fields', 400);
        }

        if (!['owner', 'manager'].includes(role)) {
            throw new ApiError('Invalid role specified', 400);
        }

        // Check if requester has permission to create this role
        // @ts-ignore
        const requester = req.user;
        if (role === 'owner' && requester.role !== 'admin') {
            throw new ApiError('Only admin can create owner accounts', 403);
        }

        if (role === 'manager' && !['admin', 'owner'].includes(requester.role)) {
            throw new ApiError('Only admin or owner can create manager accounts', 403);
        }

        // Check if user exists
        const userExists = await User.findOne({ email });
        if (userExists) {
            throw new ApiError('Email already exists', 400);
        }

        // Create user
        const user = await User.create({
            name,
            email,
            password: await hashPassword(password),
            hotelId,
            role,
            imageUrl
        });

        res.status(201).json({
            message: 'User created successfully',
            user: {
                id: user._id,
                name: user.name,
                email: user.email,
                role: user.role,
                imageUrl: user.imageUrl
            }
        });

    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

// Add new admin
export const addAdmin = async (req: Request, res: Response) => {
    try {
        const { name, email, password, imageUrl } = req.body;

        if (!name || !email || !password) {
            throw new ApiError('Please provide all required fields', 400);
        }

        // Check if admin exists
        const adminExists = await Admin.findOne({ email });
        if (adminExists) {
            throw new ApiError('Email already exists', 400);
        }

        // Create admin
        const admin = await Admin.create({
            name,
            email,
            password: await hashPassword(password),
            imageUrl
        });

        res.status(201).json({
            message: 'Admin created successfully',
            admin: {
                id: admin._id,
                name: admin.name,
                email: admin.email,
                imageUrl: admin.imageUrl
            }
        });

    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};
