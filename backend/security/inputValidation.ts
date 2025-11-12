import { Request, Response, NextFunction } from 'express';

/**
 * Input Validation Utilities
 * Prevents SQL injection and malicious input attacks
 */

// Blacklist of dangerous keywords (SQL injection vectors)
const DANGEROUS_KEYWORDS = [
    'UNION', 'union',
    'SELECT', 'select',
    'INSERT', 'insert',
    'UPDATE', 'update',
    'DELETE', 'delete',
    'DROP', 'drop',
    'EXEC', 'exec',
    'EXECUTE', 'execute',
    '--', '//',
    '/*', '*/',
    'xp_', 'sp_',
    'SCRIPT', 'script',
    'ALERT', 'alert',
    'ONERROR', 'onerror'
];

// Dangerous special characters for SQL injection
const DANGEROUS_CHARACTERS = /['";\\/-]/g;

// Patterns to detect SQL injection attempts
const SQL_INJECTION_PATTERNS = [
    /('|"|;|\/\*|\*\/|--|\|{2}|&&)/,
    /UNION.*SELECT/i,
    /INSERT.*INTO/i,
    /UPDATE.*SET/i,
    /DELETE.*FROM/i,
    /DROP.*TABLE/i,
    /EXEC.*\(/i
];

/**
 * Validate string input length
 * @param input - Input string to validate
 * @param minLength - Minimum allowed length
 * @param maxLength - Maximum allowed length
 * @returns Object with isValid flag and error message
 */
export const validateInputLength = (
    input: string,
    minLength: number = 1,
    maxLength: number = 255
): { isValid: boolean; error?: string } => {
    if (!input) {
        return { isValid: false, error: 'Input is required' };
    }

    if (input.length < minLength) {
        return {
            isValid: false,
            error: `Input must be at least ${minLength} characters long`
        };
    }

    if (input.length > maxLength) {
        return {
            isValid: false,
            error: `Input must not exceed ${maxLength} characters`
        };
    }

    return { isValid: true };
};

/**
 * Check for dangerous characters
 * @param input - Input string to validate
 * @returns Object with isValid flag and error message
 */
export const checkDangerousCharacters = (
    input: string
): { isValid: boolean; error?: string } => {
    if (!input) return { isValid: true };

    // Check for quote marks and apostrophes
    if (/['"`]/.test(input)) {
        return {
            isValid: false,
            error: 'Input contains forbidden characters: quotes or apostrophes'
        };
    }

    // Check for double hyphen (SQL comment)
    if (/--/.test(input)) {
        return {
            isValid: false,
            error: 'Input contains forbidden character sequence: double hyphen'
        };
    }

    // Check for dangerous SQL keywords
    for (const keyword of DANGEROUS_KEYWORDS) {
        if (new RegExp(`\\b${keyword}\\b`, 'i').test(input)) {
            return {
                isValid: false,
                error: `Input contains forbidden SQL keyword: ${keyword}`
            };
        }
    }

    // Check for SQL injection patterns
    for (const pattern of SQL_INJECTION_PATTERNS) {
        if (pattern.test(input)) {
            return {
                isValid: false,
                error: 'Input appears to contain SQL injection attempt'
            };
        }
    }

    return { isValid: true };
};

/**
 * Sanitize string input by removing dangerous characters
 * @param input - Input string to sanitize
 * @returns Sanitized string
 */
export const sanitizeInput = (input: string): string => {
    if (!input) return '';

    // Remove dangerous characters
    let sanitized = input.replace(DANGEROUS_CHARACTERS, '');

    // Trim whitespace
    sanitized = sanitized.trim();

    return sanitized;
};

/**
 * Validate email format
 * @param email - Email to validate
 * @returns Object with isValid flag and error message
 */
export const validateEmail = (
    email: string
): { isValid: boolean; error?: string } => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!email) {
        return { isValid: false, error: 'Email is required' };
    }

    if (email.length > 254) {
        return {
            isValid: false,
            error: 'Email must not exceed 254 characters'
        };
    }

    if (!emailRegex.test(email)) {
        return { isValid: false, error: 'Invalid email format' };
    }

    // Check for dangerous characters in email
    return checkDangerousCharacters(email);
};

/**
 * Validate password strength
 * @param password - Password to validate
 * @returns Object with isValid flag, score (0-5), and error message
 */
export const validatePasswordStrength = (
    password: string
): { isValid: boolean; score: number; error?: string } => {
    if (!password) {
        return { isValid: false, score: 0, error: 'Password is required' };
    }

    if (password.length < 8) {
        return {
            isValid: false,
            score: 0,
            error: 'Password must be at least 8 characters long'
        };
    }

    if (password.length > 128) {
        return {
            isValid: false,
            score: 0,
            error: 'Password must not exceed 128 characters'
        };
    }

    let score = 1; // Base score for meeting minimum length

    // Check for uppercase letters
    if (/[A-Z]/.test(password)) score++;

    // Check for lowercase letters
    if (/[a-z]/.test(password)) score++;

    // Check for numbers
    if (/\d/.test(password)) score++;

    // Check for special characters
    if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) score++;

    const isValid = score >= 3; // Require at least 3 out of 5 strength criteria

    return {
        isValid,
        score,
        error: isValid
            ? undefined
            : `Password is too weak. Score: ${score}/5. Include uppercase, lowercase, numbers, and special characters.`
    };
};

/**
 * Validate phone number
 * @param phone - Phone number to validate
 * @returns Object with isValid flag and error message
 */
export const validatePhoneNumber = (
    phone: string
): { isValid: boolean; error?: string } => {
    if (!phone) {
        return { isValid: false, error: 'Phone number is required' };
    }

    if (phone.length < 7 || phone.length > 15) {
        return {
            isValid: false,
            error: 'Phone number must be between 7 and 15 characters'
        };
    }

    // Allow only digits, spaces, hyphens, and plus sign
    const phoneRegex = /^[+]?[\d\s\-()]+$/;
    if (!phoneRegex.test(phone)) {
        return {
            isValid: false,
            error: 'Phone number contains invalid characters'
        };
    }

    return { isValid: true };
};

/**
 * Validate address fields
 * @param address - Address to validate
 * @returns Object with isValid flag and error message
 */
export const validateAddress = (
    address: string
): { isValid: boolean; error?: string } => {
    if (!address) {
        return { isValid: false, error: 'Address is required' };
    }

    if (address.length < 5 || address.length > 100) {
        return {
            isValid: false,
            error: 'Address must be between 5 and 100 characters'
        };
    }

    // Allow alphanumeric, spaces, hyphens, commas, and dots
    const addressRegex = /^[a-zA-Z0-9\s,.\-#/]+$/;
    if (!addressRegex.test(address)) {
        return {
            isValid: false,
            error: 'Address contains invalid characters'
        };
    }

    // Check for dangerous keywords
    return checkDangerousCharacters(address);
};

/**
 * Validate numeric input
 * @param value - Value to validate
 * @param min - Minimum allowed value
 * @param max - Maximum allowed value
 * @returns Object with isValid flag and error message
 */
export const validateNumericInput = (
    value: string | number,
    min: number = 0,
    max: number = 999999999
): { isValid: boolean; error?: string } => {
    const num = typeof value === 'string' ? parseInt(value, 10) : value;

    if (isNaN(num)) {
        return { isValid: false, error: 'Input must be a valid number' };
    }

    if (num < min) {
        return { isValid: false, error: `Value must be at least ${min}` };
    }

    if (num > max) {
        return { isValid: false, error: `Value must not exceed ${max}` };
    }

    return { isValid: true };
};

/**
 * Express middleware for input validation
 * Validates common fields in request body
 */
export const validateInputMiddleware = (
    req: Request,
    res: Response,
    next: NextFunction
): void => {
    const errors: { [key: string]: string } = {};

    // Validate email if present
    if (req.body.email) {
        const emailValidation = validateEmail(req.body.email);
        if (!emailValidation.isValid) {
            errors.email = emailValidation.error!;
        }
    }

    // Validate name if present
    if (req.body.name) {
        const nameValidation = validateInputLength(req.body.name, 2, 100);
        if (!nameValidation.isValid) {
            errors.name = nameValidation.error!;
        } else {
            const charValidation = checkDangerousCharacters(req.body.name);
            if (!charValidation.isValid) {
                errors.name = charValidation.error!;
            }
        }
    }

    // Validate password if present
    if (req.body.password) {
        const passwordValidation = validatePasswordStrength(req.body.password);
        if (!passwordValidation.isValid) {
            errors.password = passwordValidation.error!;
        }
    }

    // Validate contact number if present
    if (req.body.contactNumber) {
        const phoneValidation = validatePhoneNumber(req.body.contactNumber);
        if (!phoneValidation.isValid) {
            errors.contactNumber = phoneValidation.error!;
        }
    }

    // If there are validation errors, return 400
    if (Object.keys(errors).length > 0) {
        res.status(400).json({
            message: 'Input validation failed',
            errors
        });
        return;
    }

    next();
};

export default {
    validateInputLength,
    checkDangerousCharacters,
    sanitizeInput,
    validateEmail,
    validatePasswordStrength,
    validatePhoneNumber,
    validateAddress,
    validateNumericInput,
    validateInputMiddleware
};

