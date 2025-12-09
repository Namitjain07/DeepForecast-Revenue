import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import { validateInputMiddleware } from './inputValidation';
import { loginRateLimitMiddleware } from './bruteForceProtection';

dotenv.config();

/**
 * Security Middleware Aggregator
 * Combines all security checks and protections
 */

/**
 * Global Rate Limiter
 * Limits the number of requests from the same IP
 */
export const globalRateLimitMiddleware = rateLimit({
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000'), // Default 15 minutes
    max: parseInt(process.env.RATE_LIMIT_MAX || '1000'), // Default 1000 requests
    message: {
        message: 'Too many requests from this IP, please try again later'
    },
    standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
    legacyHeaders: false, // Disable the `X-RateLimit-*` headers
});

/**
 * Express middleware for request size limiting
 */
export const requestSizeLimitMiddleware = (req: Request, res: Response, next: NextFunction): void => {
    const MAX_BODY_SIZE = 1024 * 1024; // 1MB

    let size = 0;
    req.on('data', (chunk) => {
        size += chunk.length;
        if (size > MAX_BODY_SIZE) {
            res.status(413).json({
                message: 'Request payload too large. Maximum size is 1MB.'
            });
            req.connection.destroy();
        }
    });

    next();
};

/**
 * Express middleware for security headers
 */
export const securityHeadersMiddleware = (_req: Request, res: Response, next: NextFunction): void => {
    // Prevent XSS attacks
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');

    // Prevent MIME type sniffing
    res.setHeader('Content-Security-Policy', "default-src 'self'");

    // Prevent clickjacking
    res.setHeader('X-Frame-Options', 'SAMEORIGIN');

    // Reference policy
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

    // Feature policy
    res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');

    next();
};

/**
 * Express middleware to prevent HTTP Parameter Pollution
 */
export const preventHTTPParameterPollutionMiddleware = (
    req: Request,
    res: Response,
    next: NextFunction
): void => {
    // Check for duplicate parameters
    if (req.url.includes('?')) {
        const queryParams = new URL(`http://localhost${req.url}`).searchParams;
        const paramMap = new Map<string, number>();

        const entries = Array.from(queryParams.keys());
        for (const key of entries) {
            const count = (paramMap.get(key) || 0) + 1;
            paramMap.set(key, count);

            if (count > 1) {
                res.status(400).json({
                    message: 'Duplicate query parameters detected',
                    parameter: key
                });
                return;
            }
        }
    }

    next();
};

/**
 * Express middleware for comprehensive security checks
 */
export const comprehensiveSecurityMiddleware = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        // Apply request size limit check
        requestSizeLimitMiddleware(req, res, () => {});

        // Apply security headers
        securityHeadersMiddleware(req, res, () => {});

        // Apply HTTP parameter pollution prevention
        preventHTTPParameterPollutionMiddleware(req, res, () => {});

        next();
    } catch (error) {
        res.status(500).json({
            message: 'Security check failed',
            error: (error as Error).message
        });
    }
};

/**
 * Middleware specifically for login endpoint security
 */
export const loginSecurityMiddleware = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        // First apply input validation
        await new Promise<void>((resolve) => {
            validateInputMiddleware(req, res, () => resolve());
        });

        // Then apply rate limiting
        await loginRateLimitMiddleware(req, res, next);
    } catch (error) {
        res.status(500).json({
            message: 'Login security check failed',
            error: (error as Error).message
        });
    }
};

/**
 * Middleware for API endpoint security
 */
export const apiSecurityMiddleware = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        // Apply general security checks
        securityHeadersMiddleware(req, res, () => {});
        preventHTTPParameterPollutionMiddleware(req, res, () => {});

        // Apply input validation for POST/PUT requests
        if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
            await new Promise<void>((resolve) => {
                validateInputMiddleware(req, res, () => resolve());
            });
        }

        next();
    } catch (error) {
        res.status(500).json({
            message: 'API security check failed',
            error: (error as Error).message
        });
    }
};

export default {
    requestSizeLimitMiddleware,
    securityHeadersMiddleware,
    preventHTTPParameterPollutionMiddleware,
    comprehensiveSecurityMiddleware,
    loginSecurityMiddleware,
    apiSecurityMiddleware,
    globalRateLimitMiddleware
};

