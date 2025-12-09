import { Request, Response, NextFunction } from 'express';

/**
 * Rate Limiting and Brute Force Protection
 * Implements delays between failed login attempts
 */

interface LoginAttempt {
    timestamp: number;
    attempts: number;
    lastAttemptTime: number;
}

// In-memory store for login attempts (use Redis in production)
const loginAttempts = new Map<string, LoginAttempt>();

// Configuration
const MAX_ATTEMPTS = 5; // Maximum failed attempts before lockout
const LOCKOUT_TIME = 15 * 60 * 1000; // 15 minutes lockout
const ATTEMPT_DELAY_BASE = 500; // Base delay in ms (500ms)
const DELAY_INCREMENT = 500; // Increase delay by 500ms for each attempt

/**
 * Calculate delay based on number of failed attempts
 * Exponential backoff: 500ms, 1s, 1.5s, 2s, 2.5s
 * @param attemptCount - Number of failed attempts
 * @returns Delay in milliseconds
 */
export const calculateDelay = (attemptCount: number): number => {
    // Cap the attempts to prevent overflow
    const cappedAttempts = Math.min(attemptCount, 10);
    return ATTEMPT_DELAY_BASE + cappedAttempts * DELAY_INCREMENT;
};

/**
 * Record a failed login attempt
 * @param identifier - Email or username of login attempt
 * @returns Object with delay info and whether account is locked
 */
export const recordFailedLoginAttempt = (
    identifier: string
): { delay: number; isLocked: boolean; attemptsRemaining: number } => {
    const now = Date.now();
    const key = `login:${identifier}`;

    let attempt = loginAttempts.get(key);

    if (!attempt) {
        // First attempt
        attempt = {
            timestamp: now,
            attempts: 1,
            lastAttemptTime: now
        };
        loginAttempts.set(key, attempt);
        const delay = calculateDelay(1);
        return {
            delay,
            isLocked: false,
            attemptsRemaining: MAX_ATTEMPTS - 1
        };
    }

    // Check if lockout period has expired
    const timeSinceFirstAttempt = now - attempt.timestamp;
    if (timeSinceFirstAttempt > LOCKOUT_TIME) {
        // Reset attempts after lockout period
        attempt = {
            timestamp: now,
            attempts: 1,
            lastAttemptTime: now
        };
        loginAttempts.set(key, attempt);
        const delay = calculateDelay(1);
        return {
            delay,
            isLocked: false,
            attemptsRemaining: MAX_ATTEMPTS - 1
        };
    }

    // Increment attempts
    attempt.attempts++;
    attempt.lastAttemptTime = now;
    loginAttempts.set(key, attempt);

    const isLocked = attempt.attempts >= MAX_ATTEMPTS;
    const delay = calculateDelay(attempt.attempts);

    return {
        delay,
        isLocked,
        attemptsRemaining: Math.max(0, MAX_ATTEMPTS - attempt.attempts)
    };
};

/**
 * Record a successful login (resets attempt counter)
 * @param identifier - Email or username
 */
export const recordSuccessfulLogin = (identifier: string): void => {
    const key = `login:${identifier}`;
    loginAttempts.delete(key);
};

/**
 * Check if account is locked
 * @param identifier - Email or username
 * @returns Object with lockout status and remaining time
 */
export const checkLoginLockout = (
    identifier: string
): { isLocked: boolean; remainingTime?: number } => {
    const key = `login:${identifier}`;
    const attempt = loginAttempts.get(key);

    if (!attempt) {
        return { isLocked: false };
    }

    if (attempt.attempts < MAX_ATTEMPTS) {
        return { isLocked: false };
    }

    const timeSinceFirstAttempt = Date.now() - attempt.timestamp;
    const remainingTime = LOCKOUT_TIME - timeSinceFirstAttempt;

    if (remainingTime <= 0) {
        loginAttempts.delete(key);
        return { isLocked: false };
    }

    return {
        isLocked: true,
        remainingTime: Math.ceil(remainingTime / 1000) // Return in seconds
    };
};

/**
 * Get current attempt count for an identifier
 * @param identifier - Email or username
 * @returns Number of current attempts (0 if none)
 */
export const getAttemptCount = (identifier: string): number => {
    const key = `login:${identifier}`;
    const attempt = loginAttempts.get(key);
    return attempt ? attempt.attempts : 0;
};

/**
 * Middleware for login rate limiting
 * Checks for account lockout and implements delays
 */
export const loginRateLimitMiddleware = async (
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> => {
    const { email } = req.body;

    if (!email) {
        res.status(400).json({ message: 'Email is required' });
        return;
    }

    // Check if account is locked
    const lockoutStatus = checkLoginLockout(email);

    if (lockoutStatus.isLocked) {
        res.status(429).json({
            message: `Account is locked due to too many failed login attempts. Please try again in ${lockoutStatus.remainingTime} seconds.`,
            retryAfter: lockoutStatus.remainingTime
        });
        return;
    }

    // Get current attempt count to apply delay
    const attemptCount = getAttemptCount(email);
    if (attemptCount > 0) {
        const delay = calculateDelay(attemptCount);
        // Apply delay to subsequent attempts
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    next();
};

/**
 * Cleanup function to remove old entries
 * Should be called periodically (e.g., every hour)
 */
export const cleanupOldAttempts = (): void => {
    const now = Date.now();
    const keysToDelete: string[] = [];

    loginAttempts.forEach((attempt, key) => {
        const age = now - attempt.timestamp;
        // Remove entries older than 24 hours
        if (age > 24 * 60 * 60 * 1000) {
            keysToDelete.push(key);
        }
    });

    keysToDelete.forEach(key => loginAttempts.delete(key));
};

// Run cleanup every hour
setInterval(cleanupOldAttempts, 60 * 60 * 1000);

export default {
    recordFailedLoginAttempt,
    recordSuccessfulLogin,
    checkLoginLockout,
    getAttemptCount,
    loginRateLimitMiddleware,
    cleanupOldAttempts,
    calculateDelay,
    MAX_ATTEMPTS,
    LOCKOUT_TIME
};

