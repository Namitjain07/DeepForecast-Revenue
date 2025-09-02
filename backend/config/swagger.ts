import swaggerJsdoc from 'swagger-jsdoc';

const options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Revenue Prediction API',
      version: '1.0.0',
      description: 'API documentation for Revenue Prediction System',
    },
    components: {
      securitySchemes: {
        BearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
        },
      },
    },
    security: [{
      BearerAuth: [],
    }],
  },
  apis: ['./routes/*.ts', './models/*.ts'], // Path to the API docs
};

export const specs = swaggerJsdoc(options);
