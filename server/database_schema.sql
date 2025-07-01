-- Create database (uncomment if needed)
CREATE DATABASE hotel_management_system;
USE hotel_management_system;

-- Users table for authentication
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    full_name VARCHAR(100),
    role ENUM('admin', 'hotel_manager', 'staff') DEFAULT 'staff',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    
    INDEX idx_username (username),
    INDEX idx_role (role),
    INDEX idx_active (is_active)
);

-- Hotel data table for storing daily hotel metrics
CREATE TABLE hotel_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    total_room_inventory INT NOT NULL,
    rooms_sold INT NOT NULL,
    arrival_rooms INT NOT NULL,
    compliment_rooms INT NOT NULL,
    house_use INT NOT NULL,
    individual_confirm INT NOT NULL,
    occupancy_percentage DECIMAL(5,2) NOT NULL,
    room_revenue DECIMAL(10,2) NOT NULL,
    arr DECIMAL(8,2) NOT NULL COMMENT 'Average Room Rate',
    departure_rooms INT NOT NULL,
    ooo_rooms INT NOT NULL COMMENT 'Out of Order Rooms',
    pax INT NOT NULL COMMENT 'Number of Guests',
    snapshot_date DATE NOT NULL,
    arrival_date DATE NOT NULL,
    actual_or_forecast VARCHAR(20) NOT NULL,
    day_of_week VARCHAR(20) NOT NULL,
    revenue_diff DECIMAL(10,2) NOT NULL,
    
    -- Audit fields
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_occupancy CHECK (occupancy_percentage >= 0 AND occupancy_percentage <= 100),
    CONSTRAINT chk_revenue CHECK (room_revenue >= 0),
    CONSTRAINT chk_arr CHECK (arr >= 0),
    CONSTRAINT chk_rooms_positive CHECK (
        total_room_inventory >= 0 AND 
        rooms_sold >= 0 AND 
        arrival_rooms >= 0 AND 
        compliment_rooms >= 0 AND 
        house_use >= 0 AND 
        individual_confirm >= 0 AND 
        departure_rooms >= 0 AND 
        ooo_rooms >= 0 AND 
        pax >= 0
    ),
    CONSTRAINT chk_actual_forecast CHECK (actual_or_forecast IN ('actual', 'forecast')),
    
    -- Indexes
    INDEX idx_snapshot_date (snapshot_date),
    INDEX idx_arrival_date (arrival_date),
    INDEX idx_actual_forecast (actual_or_forecast),
    INDEX idx_created_by (created_by),
    INDEX idx_date_range (snapshot_date, arrival_date),
    UNIQUE KEY unique_entry (snapshot_date, arrival_date, actual_or_forecast),
    
    -- Foreign key
    FOREIGN KEY (created_by) REFERENCES users(username) ON UPDATE CASCADE
);

-- Default users (passwords should be hashed securely in production)
INSERT INTO users (username, password_hash, full_name, role, email) VALUES
('admin', SHA2('password123', 256), 'System Administrator', 'admin', 'admin@hotel.com'),
('hotel_manager', SHA2('hotel123', 256), 'Hotel Manager', 'hotel_manager', 'manager@hotel.com');

-- View for reporting
CREATE VIEW hotel_data_summary AS
SELECT 
    DATE_FORMAT(arrival_date, '%Y-%m') AS month_year,
    actual_or_forecast,
    AVG(occupancy_percentage) AS avg_occupancy,
    AVG(arr) AS avg_room_rate,
    SUM(room_revenue) AS total_revenue,
    SUM(rooms_sold) AS total_rooms_sold,
    AVG(total_room_inventory) AS avg_inventory,
    COUNT(*) AS total_entries
FROM hotel_data 
GROUP BY month_year, actual_or_forecast
ORDER BY month_year DESC, actual_or_forecast;

-- Performance indexes
-- Add virtual/generated columns
ALTER TABLE hotel_data 
ADD COLUMN arrival_year INT GENERATED ALWAYS AS (YEAR(arrival_date)) STORED,
ADD COLUMN arrival_month INT GENERATED ALWAYS AS (MONTH(arrival_date)) STORED,
ADD COLUMN arrival_quarter INT GENERATED ALWAYS AS (QUARTER(arrival_date)) STORED;

-- Now index the generated columns
CREATE INDEX idx_arrival_year ON hotel_data (arrival_year);
CREATE INDEX idx_arrival_month ON hotel_data (arrival_month);
CREATE INDEX idx_arrival_year_month ON hotel_data (arrival_year, arrival_month);
CREATE INDEX idx_arrival_quarter_year ON hotel_data (arrival_quarter, arrival_year);


-- Enhanced data validation stored procedure
DROP PROCEDURE IF EXISTS ValidateHotelData;
DROP PROCEDURE IF EXISTS ValidateHotelDataComplete;

DELIMITER //
CREATE PROCEDURE ValidateHotelDataComplete(
    IN p_total_inventory INT,
    IN p_rooms_sold INT,
    IN p_arrival_rooms INT,
    IN p_compliment_rooms INT,
    IN p_house_use INT,
    IN p_individual_confirm INT,
    IN p_occupancy DECIMAL(5,2),
    IN p_room_revenue DECIMAL(10,2),
    IN p_arr DECIMAL(8,2),
    IN p_departure_rooms INT,
    IN p_ooo_rooms INT,
    IN p_pax INT,
    IN p_snapshot_date DATE,
    IN p_arrival_date DATE,
    IN p_actual_or_forecast VARCHAR(20),
    IN p_day_of_week VARCHAR(20),
    IN p_revenue_diff DECIMAL(10,2),
    OUT p_is_valid BOOLEAN,
    OUT p_errors JSON
)
BEGIN
    DECLARE calculated_occupancy DECIMAL(5,2);
    DECLARE errors JSON DEFAULT JSON_ARRAY();
    DECLARE error_count INT DEFAULT 0;

    SET p_is_valid = TRUE;

    -- Validate required fields
    IF p_total_inventory IS NULL THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'total_room_inventory', 'message', 'Total room inventory is required'));
        SET error_count = error_count + 1;
    END IF;

    IF p_rooms_sold IS NULL THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'rooms_sold', 'message', 'Rooms sold is required'));
        SET error_count = error_count + 1;
    END IF;

    IF p_snapshot_date IS NULL THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'snapshot_date', 'message', 'Snapshot date is required'));
        SET error_count = error_count + 1;
    END IF;

    IF p_arrival_date IS NULL THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'arrival_date', 'message', 'Arrival date is required'));
        SET error_count = error_count + 1;
    END IF;

    -- Validate positive numbers
    IF p_total_inventory IS NOT NULL AND p_total_inventory < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'total_room_inventory', 'message', 'Total room inventory must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_rooms_sold IS NOT NULL AND p_rooms_sold < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'rooms_sold', 'message', 'Rooms sold must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_arrival_rooms IS NOT NULL AND p_arrival_rooms < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'arrival_rooms', 'message', 'Arrival rooms must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_compliment_rooms IS NOT NULL AND p_compliment_rooms < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'compliment_rooms', 'message', 'Compliment rooms must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_house_use IS NOT NULL AND p_house_use < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'house_use', 'message', 'House use must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_individual_confirm IS NOT NULL AND p_individual_confirm < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'individual_confirm', 'message', 'Individual confirm must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_departure_rooms IS NOT NULL AND p_departure_rooms < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'departure_rooms', 'message', 'Departure rooms must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_ooo_rooms IS NOT NULL AND p_ooo_rooms < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'ooo_rooms', 'message', 'Out of order rooms must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_pax IS NOT NULL AND p_pax < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'pax', 'message', 'Number of guests must be positive'));
        SET error_count = error_count + 1;
    END IF;

    -- Validate occupancy percentage range
    IF p_occupancy IS NOT NULL AND (p_occupancy < 0 OR p_occupancy > 100) THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'occupancy_percentage', 'message', CONCAT('Occupancy percentage must be between 0 and 100. Got: ', p_occupancy, '%')));
        SET error_count = error_count + 1;
    END IF;

    -- Validate revenue fields
    IF p_room_revenue IS NOT NULL AND p_room_revenue < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'room_revenue', 'message', 'Room revenue must be positive'));
        SET error_count = error_count + 1;
    END IF;

    IF p_arr IS NOT NULL AND p_arr < 0 THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'arr', 'message', 'Average room rate must be positive'));
        SET error_count = error_count + 1;
    END IF;

    -- Validate actual_or_forecast enum
    IF p_actual_or_forecast IS NOT NULL AND p_actual_or_forecast NOT IN ('actual', 'forecast') THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'actual_or_forecast', 'message', 'Must be either "actual" or "forecast"'));
        SET error_count = error_count + 1;
    END IF;

    -- Validate day_of_week
    IF p_day_of_week IS NOT NULL AND p_day_of_week NOT IN ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday') THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'day_of_week', 'message', 'Invalid day of week'));
        SET error_count = error_count + 1;
    END IF;

    -- Validate date logic
    IF p_snapshot_date IS NOT NULL AND p_arrival_date IS NOT NULL AND p_snapshot_date > p_arrival_date THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'arrival_date', 'message', 'Arrival date cannot be before snapshot date'));
        SET error_count = error_count + 1;
    END IF;

    -- Business logic validations
    IF p_rooms_sold IS NOT NULL AND p_total_inventory IS NOT NULL AND p_rooms_sold > p_total_inventory THEN
        SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'rooms_sold', 'message', CONCAT('Rooms sold (', p_rooms_sold, ') cannot exceed total inventory (', p_total_inventory, ')')));
        SET error_count = error_count + 1;
    END IF;

    -- Calculate and compare occupancy if both values are present and valid
    IF p_total_inventory IS NOT NULL AND p_rooms_sold IS NOT NULL AND p_occupancy IS NOT NULL 
       AND p_total_inventory > 0 AND p_rooms_sold >= 0 AND p_occupancy >= 0 AND p_occupancy <= 100 THEN
        SET calculated_occupancy = (p_rooms_sold / p_total_inventory) * 100;

        IF ABS(calculated_occupancy - p_occupancy) > 1.0 THEN
            SET errors = JSON_ARRAY_APPEND(errors, '$', JSON_OBJECT('field', 'occupancy_percentage', 'message', CONCAT('Occupancy mismatch: Expected ', ROUND(calculated_occupancy, 2), '%, but got ', p_occupancy, '%')));
            SET error_count = error_count + 1;
        END IF;
    END IF;

    -- Set final validation result
    IF error_count > 0 THEN
        SET p_is_valid = FALSE;
    END IF;

    SET p_errors = errors;
END //

-- Simplified validation procedure (backward compatibility)
CREATE PROCEDURE ValidateHotelData(
    IN p_total_inventory INT,
    IN p_rooms_sold INT,
    IN p_occupancy DECIMAL(5,2),
    OUT p_is_valid BOOLEAN,
    OUT p_error_message TEXT
)
BEGIN
    DECLARE validation_errors JSON;
    DECLARE error_array_length INT;
    DECLARE i INT DEFAULT 0;
    DECLARE current_error JSON;
    DECLARE error_messages TEXT DEFAULT '';

    CALL ValidateHotelDataComplete(
        p_total_inventory, p_rooms_sold, NULL, NULL, NULL, NULL,
        p_occupancy, NULL, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL, NULL, NULL,
        p_is_valid, validation_errors
    );

    -- Convert JSON errors to text for backward compatibility
    SET error_array_length = JSON_LENGTH(validation_errors);
    
    WHILE i < error_array_length DO
        SET current_error = JSON_EXTRACT(validation_errors, CONCAT('$[', i, ']'));
        SET error_messages = CONCAT(error_messages, 
                                   IF(error_messages = '', '', '; '),
                                   JSON_UNQUOTE(JSON_EXTRACT(current_error, '$.message')));
        SET i = i + 1;
    END WHILE;

    SET p_error_message = error_messages;
END //
DELIMITER ;

-- Audit log table
CREATE TABLE audit_log (
    id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(50) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    record_id INT NOT NULL,
    old_values JSON,
    new_values JSON,
    changed_by VARCHAR(50) NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_table_operation (table_name, operation),
    INDEX idx_changed_at (changed_at),
    INDEX idx_record_id (record_id)
);

-- Trigger for hotel_data updates
DELIMITER //
CREATE TRIGGER hotel_data_audit_trigger
AFTER UPDATE ON hotel_data
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, operation, record_id, old_values, new_values, changed_by)
    VALUES (
        'hotel_data',
        'UPDATE',
        NEW.id,
        JSON_OBJECT(
            'total_room_inventory', OLD.total_room_inventory,
            'rooms_sold', OLD.rooms_sold,
            'occupancy_percentage', OLD.occupancy_percentage,
            'room_revenue', OLD.room_revenue,
            'arr', OLD.arr
        ),
        JSON_OBJECT(
            'total_room_inventory', NEW.total_room_inventory,
            'rooms_sold', NEW.rooms_sold,
            'occupancy_percentage', NEW.occupancy_percentage,
            'room_revenue', NEW.room_revenue,
            'arr', NEW.arr
        ),
        OLD.created_by
    );
END //
DELIMITER ;

-- Optional: Table partitioning
-- ALTER TABLE hotel_data PARTITION BY RANGE (YEAR(arrival_date)) (
--     PARTITION p2023 VALUES LESS THAN (2024),
--     PARTITION p2024 VALUES LESS THAN (2025),
--     PARTITION p2025 VALUES LESS THAN (2026),
--     PARTITION p_future VALUES LESS THAN MAXVALUE
-- );

-- Permissions (optional for deployment)
-- CREATE USER 'hotel_app'@'localhost' IDENTIFIED BY 'secure_password_here';
-- GRANT SELECT, INSERT, UPDATE ON hotel_management_system.* TO 'hotel_app'@'localhost';
-- FLUSH PRIVILEGES;
