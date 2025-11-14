/**
 * API Services - Main Entry Point
 *
 * This file re-exports all API functions organized by category:
 * - authApi.ts: Authentication
 * - dashboardApi.ts: Dashboard statistics
 * - hotelApi.ts: Hotel management
 * - usersApi.ts: User management
 * - recordsApi.ts: Records (general operations and CSV download)
 * - recordsMetricApi.ts: Records (metric-specific endpoints)
 * - forecastApi.ts: Forecasts (general operations and CSV download)
 * - forecastMetricApi.ts: Forecasts (metric-specific endpoints)
 */

// Re-export all API functions from individual modules
export * from './index';

