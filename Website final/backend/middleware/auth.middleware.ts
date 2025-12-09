import { Request, Response, NextFunction } from 'express';
// @ts-ignore
import jwt from 'jsonwebtoken';
import User from '../models/User';
import Admin from '../models/Admin';

interface AuthRequest extends Request {
    user?: any;
}

interface JwtPayload {
    id: string;
    role: string;
}

class AuthError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

export const protect = async (req: AuthRequest, res: Response, next: NextFunction) => {
    try {
        const authHeader = req.headers.authorization;
        if (!authHeader?.startsWith('Bearer')) {
            throw new AuthError('No token provided', 401);
        }

        const token = authHeader.split(' ')[1];
        if (!token) {
            throw new AuthError('No token provided', 401);
        }
        const decoded = (jwt.verify(token, process.env.JWT_SECRET!) as unknown) as JwtPayload;

        // Check if user exists
        const user = await User.findById(decoded.id).select('-password');
        if (user) {
            req.user = { ...user.toObject(), role: decoded.role };
            return next();
        }

        // If not user, check if admin
        const admin = await Admin.findById(decoded.id).select('-password');
        if (admin) {
            req.user = { ...admin.toObject(), role: 'admin' };
            return next();
        }

        throw new AuthError('User not found', 401);
    } catch (error: any) {
        const statusCode = error.statusCode || 401;
        const message = error.message || 'Authentication failed';
        res.status(statusCode).json({ message });
    }
};

export const adminOnly = async (req: AuthRequest, res: Response, next: NextFunction) => {
    try {
        if (req.user?.role !== 'admin') {
            throw new AuthError('Admin access required', 403);
        }
        next();
    } catch (error: any) {
        const statusCode = error.statusCode || 403;
        const message = error.message || 'Admin access required';
        res.status(statusCode).json({ message });
    }
};

export const ownerOnly = async (req: AuthRequest, res: Response, next: NextFunction) => {
    try {
        if (req.user?.role !== 'owner' && req.user?.role !== 'admin') {
            throw new AuthError('Owner access required', 403);
        }
        next();
    } catch (error: any) {
        const statusCode = error.statusCode || 403;
        const message = error.message || 'Owner access required';
        res.status(statusCode).json({ message });
    }
};

export const ownerOrManagerOnly = async (req: AuthRequest, res: Response, next: NextFunction) => {
    try {
        if (!['owner', 'manager', 'admin'].includes(req.user?.role)) {
            throw new AuthError('Owner or Manager access required', 403);
        }
        next();
    } catch (error: any) {
        const statusCode = error.statusCode || 403;
        const message = error.message || 'Owner or Manager access required';
        res.status(statusCode).json({ message });
    }
};
