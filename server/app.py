from flask import Flask, render_template, request, redirect, jsonify, session, url_for
import mysql.connector
from mysql.connector import Error
import os
import logging
from datetime import datetime
from functools import wraps
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Database configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'hotel_management_system'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': int(os.getenv('DB_PORT', 3306))
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            logger.info("Database connection successful")
            return connection
        else:
            raise Exception("Connection established but not active")
    except Error as e:
        logger.error(f"Database connection error: {e}")
        raise Exception(f"Database connection failed: {e}")

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    """Verify user credentials against database"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Hash the provided password
        password_hash = hash_password(password)
        
        query = """
        SELECT id, username, full_name, role, is_active 
        FROM users 
        WHERE username = %s AND password_hash = %s AND is_active = TRUE
        """
        cursor.execute(query, (username, password_hash))
        user = cursor.fetchone()
        
        if user:
            # Update last login
            update_query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = %s"
            cursor.execute(update_query, (username,))
            connection.commit()
            
            return {
                'id': user[0],
                'username': user[1],
                'full_name': user[2],
                'role': user[3],
                'is_active': user[4]
            }
        return None
        
    except Error as e:
        logger.error(f"Database error in verify_user: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in verify_user: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def require_login(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.is_json:
                return jsonify({
                    'status': 'error',
                    'message': 'Authentication required',
                    'error_code': 'AUTHENTICATION_REQUIRED'
                }), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    try:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            return render_template('login.html', error='Username and password are required')
        
        user = verify_user(username, password)
        if user:
            session['user'] = user['username']
            session['user_role'] = user['role']
            session['user_full_name'] = user['full_name']
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('form'))
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            return render_template('login.html', error='Invalid username or password')
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return render_template('login.html', error='Login failed. Please try again.')

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({
                'status': 'error',
                'message': 'Username and password are required',
                'error_code': 'MISSING_CREDENTIALS'
            }), 400
        
        user = verify_user(username, password)
        if user:
            session['user'] = user['username']
            session['user_role'] = user['role']
            session['user_full_name'] = user['full_name']
            logger.info(f"User {username} logged in successfully via API")
            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'user': user
            }), 200
        else:
            logger.warning(f"Failed API login attempt for username: {username}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid username or password',
                'error_code': 'INVALID_CREDENTIALS'
            }), 401
    
    except Exception as e:
        logger.error(f"API login error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Login failed',
            'error_code': 'LOGIN_ERROR'
        }), 500

@app.route('/logout')
def logout():
    user = session.get('user')
    session.pop('user', None)
    if user:
        logger.info(f"User {user} logged out")
    return redirect(url_for('login'))

@app.route('/api/logout', methods=['POST'])
def api_logout():
    user = session.get('user')
    session.pop('user', None)
    if user:
        logger.info(f"User {user} logged out via API")
    return jsonify({
        'status': 'success',
        'message': 'Logout successful'
    }), 200

@app.route('/')
@require_login
def form():
    return render_template('form.html', user=session.get('user'))

@app.route('/submit', methods=['POST'])
@require_login
def submit():
    try:
        # Validate form data
        required_fields = [
            "Total Room Inventory", "Rooms Sold", "Arrival Rooms", "Compliment Rooms",
            "House Use", "Individual Confirm", "Occupancy %", "Room Revenue", "ARR",
            "Departure Rooms", "OOO Rooms", "Pax", "snapshot_date", "arrival_date",
            "actual_or_forecast", "Day", "revenue_diff"
        ]
        
        missing_fields = [field for field in required_fields if field not in request.form or not request.form[field].strip()]
        if missing_fields:
            logger.warning(f"Form submission missing fields: {missing_fields}")
            return f"Error: Missing required fields: {', '.join(missing_fields)}", 400
        
        data = {field: request.form[field].strip() for field in required_fields}
        
        # Validate and parse date fields
        try:
            snapshot_date = datetime.strptime(data["snapshot_date"], "%Y-%m-%d").date()
            arrival_date = datetime.strptime(data["arrival_date"], "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Date parsing error in form submission: {str(e)}")
            return f"Error: Invalid date format. Please use YYYY-MM-DD format. Details: {str(e)}", 400
        
        # Validate numeric fields
        numeric_fields = ["Total Room Inventory", "Rooms Sold", "Arrival Rooms", "Compliment Rooms",
                         "House Use", "Individual Confirm", "Occupancy %", "Room Revenue", "ARR",
                         "Departure Rooms", "OOO Rooms", "Pax", "revenue_diff"]
        
        for field in numeric_fields:
            try:
                float(data[field])
            except ValueError:
                logger.error(f"Invalid numeric value for field {field}: {data[field]}")
                return f"Error: Field '{field}' must be a valid number. Got: {data[field]}", 400

        save_hotel_data(data, snapshot_date, arrival_date)
        logger.info(f"Form data successfully saved for date: {arrival_date}")
        return redirect('/')
        
    except Exception as e:
        logger.error(f"Unexpected error in form submission: {str(e)}")
        return f"Internal server error: {str(e)}", 500

@app.route('/api/submit', methods=['POST'])
@require_login
def api_submit():
    try:
        if not request.is_json:
            logger.warning("API request received without JSON content type")
            return jsonify({
                'status': 'error', 
                'message': 'Request must be JSON',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400

        data = request.get_json()
        
        if not data:
            logger.warning("API request received with empty JSON body")
            return jsonify({
                'status': 'error', 
                'message': 'Request body cannot be empty',
                'error_code': 'EMPTY_BODY'
            }), 400

        # Validate required fields
        required_fields = [
            "Total Room Inventory", "Rooms Sold", "Arrival Rooms", "Compliment Rooms",
            "House Use", "Individual Confirm", "Occupancy %", "Room Revenue", "ARR",
            "Departure Rooms", "OOO Rooms", "Pax", "snapshot_date", "arrival_date",
            "actual_or_forecast", "Day", "revenue_diff"
        ]
        
        missing = [field for field in required_fields if field not in data or not str(data[field]).strip()]
        if missing:
            logger.warning(f"API submission missing fields: {missing}")
            return jsonify({
                'status': 'error', 
                'message': f'Missing or empty required fields: {missing}',
                'error_code': 'MISSING_FIELDS',
                'missing_fields': missing
            }), 400

        # Clean string fields
        for field in data:
            if isinstance(data[field], str):
                data[field] = data[field].strip()

        # Validate and parse date fields
        try:
            snapshot_date = datetime.strptime(data["snapshot_date"], "%Y-%m-%d").date()
            arrival_date = datetime.strptime(data["arrival_date"], "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Date parsing error in API submission: {str(e)}")
            return jsonify({
                'status': 'error', 
                'message': f'Invalid date format. Use YYYY-MM-DD format. Details: {str(e)}',
                'error_code': 'INVALID_DATE_FORMAT'
            }), 400
        except Exception as e:
            logger.error(f"Unexpected date parsing error: {str(e)}")
            return jsonify({
                'status': 'error', 
                'message': f'Date parsing error: {str(e)}',
                'error_code': 'DATE_PARSING_ERROR'
            }), 400

        # Validate numeric fields
        numeric_fields = ["Total Room Inventory", "Rooms Sold", "Arrival Rooms", "Compliment Rooms",
                         "House Use", "Individual Confirm", "Occupancy %", "Room Revenue", "ARR",
                         "Departure Rooms", "OOO Rooms", "Pax", "revenue_diff"]
        
        invalid_numeric = []
        for field in numeric_fields:
            try:
                float(data[field])
            except (ValueError, TypeError):
                invalid_numeric.append(field)
        
        if invalid_numeric:
            logger.error(f"Invalid numeric values in API submission: {invalid_numeric}")
            return jsonify({
                'status': 'error',
                'message': f'Fields must be valid numbers: {invalid_numeric}',
                'error_code': 'INVALID_NUMERIC_VALUES',
                'invalid_fields': invalid_numeric
            }), 400

        save_hotel_data(data, snapshot_date, arrival_date)
        logger.info(f"API data successfully saved for date: {arrival_date}")
        return jsonify({
            'status': 'success', 
            'message': 'Data saved successfully',
            'arrival_date': str(arrival_date)
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in API submission: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error occurred',
            'error_code': 'INTERNAL_ERROR',
            'details': str(e)
        }), 500

def save_hotel_data(data, snapshot_date, arrival_date):
    """Save hotel data to MySQL database"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Validate occupancy percentage before database insertion
        occupancy_pct = float(data["Occupancy %"])
        if occupancy_pct < 0 or occupancy_pct > 100:
            raise Exception(f"Occupancy percentage must be between 0 and 100. Got: {occupancy_pct}%")
        
        # Validate that rooms sold doesn't exceed inventory
        total_inventory = int(float(data["Total Room Inventory"]))
        rooms_sold = int(float(data["Rooms Sold"]))
        if rooms_sold > total_inventory:
            raise Exception(f"Rooms sold ({rooms_sold}) cannot exceed total inventory ({total_inventory})")
        
        # Validate data using stored procedure if it exists
        try:
            # Call stored procedure with correct parameters (5 parameters: 3 IN, 2 OUT)
            cursor.callproc('ValidateHotelData', [
                total_inventory,
                rooms_sold,
                occupancy_pct,
                0,  # p_is_valid (OUT parameter placeholder)
                ''  # p_error_message (OUT parameter placeholder)
            ])
            
            # Get validation results
            for result in cursor.stored_results():
                validation_result = result.fetchall()
                if validation_result:
                    logger.info(f"Validation result: {validation_result}")
        except Error as proc_error:
            logger.warning(f"Stored procedure not available or error: {proc_error}")
            # Continue without stored procedure validation
        
        # Validate revenue and ARR
        room_revenue = float(data["Room Revenue"])
        arr = float(data["ARR"])
        if room_revenue < 0:
            raise Exception(f"Room revenue cannot be negative. Got: {room_revenue}")
        if arr < 0:
            raise Exception(f"ARR cannot be negative. Got: {arr}")
        
        query = """
        INSERT INTO hotel_data (
            total_room_inventory, rooms_sold, arrival_rooms, compliment_rooms,
            house_use, individual_confirm, occupancy_percentage, room_revenue,
            arr, departure_rooms, ooo_rooms, pax, snapshot_date, arrival_date,
            actual_or_forecast, day_of_week, revenue_diff, created_by
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        values = (
            total_inventory,
            rooms_sold,
            int(float(data["Arrival Rooms"])),
            int(float(data["Compliment Rooms"])),
            int(float(data["House Use"])),
            int(float(data["Individual Confirm"])),
            occupancy_pct,
            room_revenue,
            arr,
            int(float(data["Departure Rooms"])),
            int(float(data["OOO Rooms"])),
            int(float(data["Pax"])),
            snapshot_date,
            arrival_date,
            data["actual_or_forecast"],
            data["Day"],
            float(data["revenue_diff"]),
            session.get('user', 'unknown')
        )
        
        cursor.execute(query, values)
        connection.commit()
        logger.info(f"Data saved to database for arrival date: {arrival_date}")
        
    except mysql.connector.IntegrityError as e:
        if "unique_entry" in str(e):
            logger.warning(f"Duplicate entry attempted: {e}")
            raise Exception("A record for this snapshot date, arrival date, and forecast type already exists")
        elif "foreign key constraint" in str(e).lower():
            logger.error(f"Foreign key constraint error: {e}")
            raise Exception("Invalid user reference - please log in again")
        else:
            logger.error(f"Database integrity error: {e}")
            raise Exception(f"Database constraint violation: {e}")
    except mysql.connector.DataError as e:
        if "out of range" in str(e).lower():
            logger.error(f"Data out of range error: {e}")
            raise Exception("One or more values are out of acceptable range. Please check your input values.")
        else:
            logger.error(f"Data error: {e}")
            raise Exception(f"Data validation error: {e}")
    except Error as e:
        logger.error(f"Database error in save_hotel_data: {e}")
        raise Exception(f"Failed to save data to database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in save_hotel_data: {e}")
        raise Exception(f"Unexpected error occurred: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# Add a database health check endpoint
@app.route('/api/health')
def health_check():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Database connection healthy',
            'database_config': {
                'host': DB_CONFIG['host'],
                'database': DB_CONFIG['database'],
                'user': DB_CONFIG['user'],
                'port': DB_CONFIG['port']
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Database connection failed: {str(e)}',
            'database_config': {
                'host': DB_CONFIG['host'],
                'database': DB_CONFIG['database'],
                'user': DB_CONFIG['user'],
                'port': DB_CONFIG['port']
            }
        }), 500

@app.route('/api/data', methods=['GET'])
@require_login
def api_get_data():
    """
    Get hotel data with optional filtering and pagination
    Query parameters:
    - start_date: Filter by arrival date (YYYY-MM-DD)
    - end_date: Filter by arrival date (YYYY-MM-DD)
    - actual_or_forecast: Filter by actual/forecast
    - page: Page number (default: 1)
    - limit: Records per page (default: 50, max: 500)
    - sort_by: Sort field (default: arrival_date)
    - sort_order: asc/desc (default: desc)
    """
    connection = None
    cursor = None
    try:
        # Get query parameters with defaults
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        actual_or_forecast = request.args.get('actual_or_forecast')
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 50)), 500)  # Max 500 records
        sort_by = request.args.get('sort_by', 'arrival_date')
        sort_order = request.args.get('sort_order', 'desc').upper()
        
        # Validate parameters
        if page < 1:
            return jsonify({
                'status': 'error',
                'message': 'Page number must be positive',
                'error_code': 'INVALID_PAGE'
            }), 400
            
        if limit < 1:
            return jsonify({
                'status': 'error',
                'message': 'Limit must be positive',
                'error_code': 'INVALID_LIMIT'
            }), 400
            
        if sort_order not in ['ASC', 'DESC']:
            sort_order = 'DESC'
            
        # Valid sort fields
        valid_sort_fields = [
            'id', 'arrival_date', 'snapshot_date', 'occupancy_percentage',
            'room_revenue', 'arr', 'rooms_sold', 'total_room_inventory',
            'created_at', 'updated_at'
        ]
        
        if sort_by not in valid_sort_fields:
            sort_by = 'arrival_date'
        
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                where_conditions.append("arrival_date >= %s")
                params.append(start_date)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid start_date format. Use YYYY-MM-DD',
                    'error_code': 'INVALID_DATE_FORMAT'
                }), 400
        
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
                where_conditions.append("arrival_date <= %s")
                params.append(end_date)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid end_date format. Use YYYY-MM-DD',
                    'error_code': 'INVALID_DATE_FORMAT'
                }), 400
        
        if actual_or_forecast and actual_or_forecast in ['actual', 'forecast']:
            where_conditions.append("actual_or_forecast = %s")
            params.append(actual_or_forecast)
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Count total records
        count_query = f"SELECT COUNT(*) as total FROM hotel_data{where_clause}"
        cursor.execute(count_query, params)
        total_records = cursor.fetchone()['total']
        
        # Calculate pagination
        offset = (page - 1) * limit
        total_pages = (total_records + limit - 1) // limit
        
        # Main query
        query = f"""
        SELECT 
            id, total_room_inventory, rooms_sold, arrival_rooms, compliment_rooms,
            house_use, individual_confirm, occupancy_percentage, room_revenue,
            arr, departure_rooms, ooo_rooms, pax, snapshot_date, arrival_date,
            actual_or_forecast, day_of_week, revenue_diff, created_by,
            created_at, updated_at
        FROM hotel_data
        {where_clause}
        ORDER BY {sort_by} {sort_order}
        LIMIT %s OFFSET %s
        """
        
        params.extend([limit, offset])
        cursor.execute(query, params)
        records = cursor.fetchall()
        
        # Convert dates to strings for JSON serialization
        for record in records:
            if record['snapshot_date']:
                record['snapshot_date'] = record['snapshot_date'].strftime('%Y-%m-%d')
            if record['arrival_date']:
                record['arrival_date'] = record['arrival_date'].strftime('%Y-%m-%d')
            if record['created_at']:
                record['created_at'] = record['created_at'].isoformat()
            if record['updated_at']:
                record['updated_at'] = record['updated_at'].isoformat()
        
        return jsonify({
            'status': 'success',
            'data': records,
            'pagination': {
                'page': page,
                'limit': limit,
                'total_records': total_records,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'filters': {
                'start_date': start_date,
                'end_date': end_date,
                'actual_or_forecast': actual_or_forecast,
                'sort_by': sort_by,
                'sort_order': sort_order.lower()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching hotel data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch data',
            'error_code': 'FETCH_ERROR',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.route('/api/data/<int:record_id>', methods=['GET'])
@require_login
def api_get_single_record(record_id):
    """Get a single hotel data record by ID"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT 
            id, total_room_inventory, rooms_sold, arrival_rooms, compliment_rooms,
            house_use, individual_confirm, occupancy_percentage, room_revenue,
            arr, departure_rooms, ooo_rooms, pax, snapshot_date, arrival_date,
            actual_or_forecast, day_of_week, revenue_diff, created_by,
            created_at, updated_at
        FROM hotel_data
        WHERE id = %s
        """
        
        cursor.execute(query, (record_id,))
        record = cursor.fetchone()
        
        if not record:
            return jsonify({
                'status': 'error',
                'message': f'Record with ID {record_id} not found',
                'error_code': 'RECORD_NOT_FOUND'
            }), 404
        
        # Convert dates to strings for JSON serialization
        if record['snapshot_date']:
            record['snapshot_date'] = record['snapshot_date'].strftime('%Y-%m-%d')
        if record['arrival_date']:
            record['arrival_date'] = record['arrival_date'].strftime('%Y-%m-%d')
        if record['created_at']:
            record['created_at'] = record['created_at'].isoformat()
        if record['updated_at']:
            record['updated_at'] = record['updated_at'].isoformat()
        
        return jsonify({
            'status': 'success',
            'data': record
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching record {record_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch record',
            'error_code': 'FETCH_ERROR',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.route('/api/data/summary', methods=['GET'])
@require_login
def api_get_summary():
    """Get summarized hotel data using the database view"""
    connection = None
    cursor = None
    try:
        # Get query parameters
        start_month = request.args.get('start_month')  # Format: YYYY-MM
        end_month = request.args.get('end_month')    # Format: YYYY-MM
        actual_or_forecast = request.args.get('actual_or_forecast')
        
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Build WHERE clause for the view
        where_conditions = []
        params = []
        
        if start_month:
            try:
                datetime.strptime(start_month + '-01', '%Y-%m-%d')
                where_conditions.append("month_year >= %s")
                params.append(start_month)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid start_month format. Use YYYY-MM',
                    'error_code': 'INVALID_MONTH_FORMAT'
                }), 400
        
        if end_month:
            try:
                datetime.strptime(end_month + '-01', '%Y-%m-%d')
                where_conditions.append("month_year <= %s")
                params.append(end_month)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid end_month format. Use YYYY-MM',
                    'error_code': 'INVALID_MONTH_FORMAT'
                }), 400
        
        if actual_or_forecast and actual_or_forecast in ['actual', 'forecast']:
            where_conditions.append("actual_or_forecast = %s")
            params.append(actual_or_forecast)
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
        SELECT 
            month_year, actual_or_forecast, avg_occupancy, avg_room_rate,
            total_revenue, total_rooms_sold, avg_inventory, total_entries
        FROM hotel_data_summary
        {where_clause}
        ORDER BY month_year DESC, actual_or_forecast
        """
        
        cursor.execute(query, params)
        summary_data = cursor.fetchall()
        
        return jsonify({
            'status': 'success',
            'data': summary_data,
            'filters': {
                'start_month': start_month,
                'end_month': end_month,
                'actual_or_forecast': actual_or_forecast
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching summary data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch summary data',
            'error_code': 'FETCH_ERROR',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning(f"405 error: {request.method} {request.url}")
    return jsonify({'status': 'error', 'message': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
