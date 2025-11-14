// ==================== AUTH APIs ====================
export * from './authApi';

// ==================== DASHBOARD APIs ====================
export * from './dashboardApi';

// ==================== HOTEL APIs ====================
export * from './hotelApi';

// ==================== USERS APIs ====================
export * from './usersApi';

// ==================== RECORDS APIs ====================
export * from './recordsApi';
export * from './recordsMetricApi';

// ==================== FORECAST APIs ====================
export * from './forecastApi';
export * from './forecastMetricApi';

// Export API URL and getToken for other uses if needed
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';
export const getToken = () => localStorage.getItem('token');

