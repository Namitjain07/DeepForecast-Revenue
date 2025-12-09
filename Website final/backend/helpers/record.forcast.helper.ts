/**
 * Helper function to calculate date range and aggregation for RECORDS (backward from last date)
 * @param period - Time period (1w, 1m, 3m, 6m, 12m)
 * @param lastRecordDate - The last (most recent) date of records for the hotel
 */
export const getPeriodConfig = (period: '1w' | '1m' | '3m' | '6m' | '12m', lastRecordDate: Date) => {
    let startDate = new Date(lastRecordDate);
    let aggregationDays = 1;

    switch (period) {
        case '1w':
            startDate.setDate(lastRecordDate.getDate() - 7);
            aggregationDays = 1;
            break;
        case '1m':
            startDate.setMonth(lastRecordDate.getMonth() - 1);
            aggregationDays = 1;
            break;
        case '3m':
            startDate.setMonth(lastRecordDate.getMonth() - 3);
            aggregationDays = 3;
            break;
        case '6m':
            startDate.setMonth(lastRecordDate.getMonth() - 6);
            aggregationDays = 7;
            break;
        case '12m':
            startDate.setFullYear(lastRecordDate.getFullYear() - 1);
            aggregationDays = 30;
            break;
    }

    return { startDate, aggregationDays };
};

/**
 * Helper function to calculate date range and aggregation for FORECAST (forward from last date)
 * @param period - Time period (1w, 1m, 3m, 6m, 12m)
 * @param lastForecastDate - The last (most recent) date of forecasts for the hotel
 */
export const getPeriodConfigForecast = (period: '1w' | '1m' | '3m' | '6m' | '12m', lastForecastDate: Date) => {
    let endDate = new Date(lastForecastDate);
    let aggregationDays = 1;

    switch (period) {
        case '1w':
            endDate.setDate(lastForecastDate.getDate() + 7);
            aggregationDays = 1;
            break;
        case '1m':
            endDate.setMonth(lastForecastDate.getMonth() + 1);
            aggregationDays = 1;
            break;
        case '3m':
            endDate.setMonth(lastForecastDate.getMonth() + 3);
            aggregationDays = 3;
            break;
        case '6m':
            endDate.setMonth(lastForecastDate.getMonth() + 6);
            aggregationDays = 7;
            break;
        case '12m':
            endDate.setFullYear(lastForecastDate.getFullYear() + 1);
            aggregationDays = 30;
            break;
    }

    return { endDate, aggregationDays };
};

/**
 * Helper function to aggregate records by metric
 */
export const aggregateMetric = (records: any[], metric: string, aggregationDays: number) => {
    if (aggregationDays === 1) {
        return records.map(r => ({
            date: r.date,
            value: r[metric]
        }));
    }

    const aggregated: any[] = [];
    const buckets: { [key: string]: any[] } = {};

    records.forEach(record => {
        const date = new Date(record.date);
        const bucketKey = new Date(date.getFullYear(), date.getMonth(), Math.floor(date.getDate() / aggregationDays) * aggregationDays);
        const key = bucketKey.toISOString().split('T')[0]!;

        if (!buckets[key]) {
            buckets[key] = [];
        }
        buckets[key]!.push(record);
    });

    Object.keys(buckets).sort().forEach(key => {
        const bucket = buckets[key]!;
        const sum = bucket.reduce((acc: number, r: any) => acc + (r[metric] || 0), 0);
        const avg = Math.round(sum / bucket.length);
        aggregated.push({
            date: new Date(key),
            value: avg
        });
    });

    return aggregated;
};