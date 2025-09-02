// @ts-ignore
import bcrypt from 'bcryptjs';
// @ts-ignore
import jwt from 'jsonwebtoken';
import { Types } from 'mongoose';
// @ts-ignore
import crypto from 'crypto';

interface TokenPayload {
    id: string;
    role: string;
}

export const generateToken = (id: string, role: string): string => {
    const payload: TokenPayload = { id, role };
    return jwt.sign(payload, process.env.JWT_SECRET!, {
        expiresIn: '30d',
    });
};

export const hashPassword = async (password: string): Promise<string> => {
    const salt = await bcrypt.genSalt(10);
    return await bcrypt.hash(password, salt);
};

export const generateUniqueId = (prefix: string): string => {
    const timestamp = Date.now().toString(36);
    const randomStr = crypto.randomBytes(4).toString('hex');
    return `${prefix}_${timestamp}${randomStr}`;
};
