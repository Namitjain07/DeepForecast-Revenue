// @ts-ignore
import express from "express";
import connectDB from "./config/db";
// @ts-ignore
import cors from "cors";
// @ts-ignore
import dotenv from "dotenv";
// @ts-ignore
import compression from "compression";
// @ts-ignore
import swaggerUi from 'swagger-ui-express';
import { specs } from './config/swagger';
import authRoutes from "./routes/auth.route";
import hotelRoutes from "./routes/hotel.route";
import adminDashboardRoutes from "./routes/admin.dashboard.route";
import uploadRoutes from "./routes/upload.route";
import userRoutes from "./routes/user.route";
import recordRoutes from "./routes/records.route";
import forecastRoutes from "./routes/forcast.route";
import trainRoutes from "./routes/train.route";
import { comprehensiveSecurityMiddleware, globalRateLimitMiddleware } from './security/securityMiddleware';

dotenv.config();

const app = express();

// Trust proxy for rate limiting behind load balancers/proxies
app.set('trust proxy', 1);

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors({
    origin: process.env.CORS_ORIGIN || '*', // Allow configuring origin for security
    credentials: true
}));
app.use(compression() as any);
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Apply global rate limiting
app.use(globalRateLimitMiddleware as any);

// Apply comprehensive security middleware to all routes
app.use(comprehensiveSecurityMiddleware);

// Swagger Documentation
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(specs));

// Routes
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/hotels', hotelRoutes);
app.use('/api/v1/admin/dashboard', adminDashboardRoutes);
app.use('/api/v1/upload', uploadRoutes);
app.use('/api/v1/users', userRoutes);
app.use('/api/v1/records', recordRoutes);
app.use('/api/v1/forecast', forecastRoutes);
app.use('/api/v1/train', trainRoutes);

// Basic route
app.get("/", (_req, res) => {
    res.send("Server is running");
});

// Error handling middleware
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Something broke!', error: err.message });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(` Server running on http://localhost:${PORT}`);
    console.log(` API Documentation available at http://localhost:${PORT}/api-docs`);
    console.log(` Security middleware: Enabled (Input validation, Brute force protection, XSS prevention)`);
});
