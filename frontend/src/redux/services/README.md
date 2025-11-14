```
# Redux Services Architecture

This directory contains modularized API services organized by feature/domain.

## File Structure

```
services/
├── api.ts                      # Main entry point (re-exports from index.ts)
├── index.ts                    # Central export hub
├── authApi.ts                  # Authentication APIs
├── dashboardApi.ts             # Dashboard statistics APIs
├── hotelApi.ts                 # Hotel management APIs
├── usersApi.ts                 # User management APIs
├── recordsApi.ts               # Records general operations
├── recordsMetricApi.ts         # Records metric-specific (Revenue, RoomSold, Arrival, Departure, OOO)
├── forecastApi.ts              # Forecast general operations
└── forecastMetricApi.ts        # Forecast metric-specific (Revenue, RoomSold, Arrival, Departure, OOO)
```

## API Categories

### 1. **authApi.ts** - Authentication
- `loginUser(email, password)` - User login

### 2. **dashboardApi.ts** - Dashboard
- `fetchDashboardStats()` - Get dashboard statistics

### 3. **hotelApi.ts** - Hotel Management
- `fetchAllHotels(page, limit)` - Get all hotels with pagination
- `fetchRecentlyAddedHotels(limit)` - Get recently added hotels
- `searchHotels(searchTerm, page, limit)` - Search hotels
- `addHotel(hotelData)` - Add new hotel
- `fetchGeneralInfo(hotelId)` - Get hotel general information

### 4. **usersApi.ts** - User Management
- `fetchUsersByHotel(hotelId)` - Get users for a hotel
- `updateUserData(userId, userData)` - Update user
- `deleteUserData(userId)` - Delete user

### 5. **recordsApi.ts** - Records (General)
- `fetchAvailableDates(hotelId)` - Get available record dates
- `fetchRecentRecords(hotelId)` - Get 5 most recent records
- `fetchRecordsByDateRange(hotelId, startDate, endDate)` - Get records by date range
- `downloadRecordsCSV(hotelId, startDate, endDate)` - Download records as CSV

### 6. **recordsMetricApi.ts** - Records (Metric-Specific)
Each metric has its own fetch function with period support (1w, 1m, 3m, 6m, 12m):
- `fetchRevenueData(hotelId, period)` - Revenue records
- `fetchRoomSoldData(hotelId, period)` - Room sold records
- `fetchArrivalData(hotelId, period)` - Arrival records
- `fetchDepartureData(hotelId, period)` - Departure records
- `fetchOOOData(hotelId, period)` - OOO records

### 7. **forecastApi.ts** - Forecast (General)
- `fetchForecastAvailableDates(hotelId)` - Get available forecast dates
- `downloadForecastCSV(hotelId, startDate, endDate)` - Download forecast as CSV

### 8. **forecastMetricApi.ts** - Forecast (Metric-Specific)
Each metric has its own forecast function with period support (1w, 1m, 3m, 6m, 12m):
- `fetchRevenueForecasts(hotelId, period)` - Revenue forecast
- `fetchRoomSoldForecasts(hotelId, period)` - Room sold forecast
- `fetchArrivalForecasts(hotelId, period)` - Arrival forecast
- `fetchDepartureForecasts(hotelId, period)` - Departure forecast
- `fetchOOOForecasts(hotelId, period)` - OOO forecast

## Usage Examples

### Importing from index.ts (Recommended)
```typescript
import {
  loginUser,
  fetchDashboardStats,
  fetchAllHotels,
  fetchRevenueData,
  fetchRevenueForecasts,
  // ... other APIs
} from '@/redux/services';
```

### Using in Components
```typescript
import { useDispatch } from 'react-redux';
import { fetchRevenueData, fetchRevenueForecasts } from '@/redux/services';

export const MyComponent = () => {
  const dispatch = useDispatch();
  
  useEffect(() => {
    dispatch(fetchRevenueData(hotelId, '1m') as any);
    dispatch(fetchRevenueForecasts(hotelId, '1m') as any);
  }, [hotelId]);
  
  // Component code...
};
```

## Benefits of Modular Structure

✅ **Separation of Concerns** - Each file handles one domain/feature
✅ **Easier Maintenance** - Find and update APIs quickly
✅ **Better Scalability** - Add new API categories easily
✅ **Reduced Merge Conflicts** - Smaller files mean fewer conflicts
✅ **Improved Readability** - Clear organization and naming
✅ **Code Reusability** - Easy to import specific APIs where needed
✅ **Better Testing** - Can test individual API files separately

## Adding New APIs

1. Create a new file: `myFeatureApi.ts`
2. Add your API functions with proper Redux dispatch calls
3. Add export statement in `index.ts`
4. Import and use in components

Example:
```typescript
// newFeatureApi.ts
export const fetchMyFeature = (params) => async (dispatch: Dispatch) => {
  try {
    dispatch(myFeatureStart());
    const response = await axios.get(`${API_URL}/my-endpoint`, {
      headers: {
        'Authorization': `Bearer ${getToken()}`,
        'Content-Type': 'application/json',
      },
    });
    dispatch(myFeatureSuccess(response.data));
    return response.data;
  } catch (error: any) {
    dispatch(myFeatureFailure(error.message));
    throw error;
  }
};
```

Then in `index.ts`:
```typescript
export * from './newFeatureApi';
```

## Notes

- All API URLs use the `API_URL` constant from environment variables
- All requests include Bearer token authentication
- Redux dispatch is used for state management
- Error handling is implemented with try-catch
- CSV download functions handle pagination automatically

