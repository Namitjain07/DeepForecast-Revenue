# Revenue Prediction Website

A full-stack web application for hotel revenue prediction and management. The system allows hotel administrators, owners, and managers to track and forecast revenue metrics.

## Features

- **User Management**
  - Multi-role authentication (Admin, Owner, Manager)
  - Secure JWT-based authentication
  - Role-based access control

- **Hotel Management**
  - Add and manage hotel properties
  - Track hotel details and location information
  - Multiple hotels under single ownership

- **Revenue Tracking**
  - Daily revenue records
  - Room occupancy tracking
  - Guest statistics
  - Forecasting capabilities

## Tech Stack

### Backend
- Node.js + Express.js
- TypeScript
- MongoDB with Mongoose
- JWT Authentication
- Swagger API Documentation

### Frontend
- React.js with TypeScript
- Redux Toolkit for state management
- Vite as build tool
- Modern CSS with responsive design

## Project Structure

```
├── backend/
│   ├── config/         # Configuration files
│   ├── controllers/    # Request handlers
│   ├── helpers/        # Utility functions
│   ├── middleware/     # Express middlewares
│   ├── models/         # MongoDB models
│   └── routes/         # API routes
├── frontend/
│   ├── src/
│   │   ├── redux/     # State management
│   │   ├── components/# React components
│   │   └── assets/    # Static files
│   └── public/        # Public assets
```

## API Documentation

API documentation is available at `/api-docs` when running the server. Key endpoints include:

### Authentication
- POST `/api/v1/auth/login` - User login
- POST `/api/v1/auth/add_user` - Add new user
- POST `/api/v1/auth/add_admin` - Add new admin

### Hotels
- POST `/api/v1/hotels/add_hotel` - Add new hotel

## Getting Started

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
# Backend
cd backend
npm install

# Frontend
cd frontend
npm install
```

3. Create a `.env` file in the backend directory:


4. Start the development servers:
```bash
# Backend
cd backend
npm run dev

# Frontend
cd frontend
npm run dev
```

## User Roles and Permissions

- **Admin**: Full system access, can manage hotels and users
- **Owner**: Can manage their hotels and add managers
- **Manager**: Can manage daily operations and records

## Security Features

- Password hashing with bcrypt
- JWT token authentication
- Role-based access control
- Request validation and sanitization

## License

[MIT License](LICENSE)

