import React, { useState, useMemo, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../../redux/store';
import { fetchRevenueData, fetchRevenueForecasts } from '../../redux/services/api';
import {
    LineChart,
    Line,
    BarChart,
    Bar,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import '../../stylesheet/ui/component-ui-revenue-graph.css';

interface RevenueGraphProps {
    hotelId: string;
}

interface ChartDataPoint {
    date: string;
    actual?: number;
    predicted?: number;
    isPredicted?: boolean;
}

const RevenueGraph: React.FC<RevenueGraphProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const [timePeriod, setTimePeriod] = useState<'1w' | '1m' | '3m' | '6m' | '12m'>('1m');
    const { revenue: revenueRecords } = useSelector((state: RootState) => state.records);
    const { revenue: revenueForecasts } = useSelector((state: RootState) => state.forecast);

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchRevenueData(hotelId, timePeriod) as any);
            dispatch(fetchRevenueForecasts(hotelId, timePeriod) as any);
        }
    }, [hotelId, timePeriod, dispatch]);

    const { revenueData, chartHeight, chartType, xAxisInterval } = useMemo(() => {
        const data: ChartDataPoint[] = [];

        if (revenueRecords && revenueRecords.length > 0) {
            revenueRecords.forEach(record => {
                const formattedDate = new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                data.push({
                    date: formattedDate,
                    actual: record.value,
                    isPredicted: false
                });
            });
        }

        if (revenueForecasts && revenueForecasts.length > 0) {
            revenueForecasts.forEach(forecast => {
                const formattedDate = new Date(forecast.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                const existingIndex = data.findIndex(d => d.date === formattedDate);

                if (existingIndex >= 0) {
                    data[existingIndex].predicted = forecast.value;
                } else {
                    data.push({
                        date: formattedDate,
                        predicted: forecast.value,
                        isPredicted: true
                    });
                }
            });
        }

        let chartHeight = 300;
        let chartType: 'bar' | 'line' | 'area' = 'bar';
        let xAxisInterval = 0;

        switch (timePeriod) {
            case '1w':
                chartHeight = 300;
                chartType = 'bar';
                xAxisInterval = 0;
                break;
            case '1m':
                chartHeight = 350;
                chartType = 'bar';
                xAxisInterval = 2;
                break;
            case '3m':
                chartHeight = 350;
                chartType = 'area';
                xAxisInterval = 5;
                break;
            case '6m':
                chartHeight = 400;
                chartType = 'area';
                xAxisInterval = 10;
                break;
            case '12m':
                chartHeight = 400;
                chartType = 'line';
                xAxisInterval = 15;
                break;
        }

        return {
            revenueData: data,
            chartHeight,
            chartType,
            xAxisInterval
        };
    }, [revenueRecords, revenueForecasts, timePeriod]);

    return (
        <div className="component-ui-revenue-graph-container">
            <div className="component-ui-revenue-graph-header">
                <h3 className="component-ui-revenue-graph-title">Revenue Analysis</h3>
                <div className="component-ui-revenue-graph-toggle">
                    {(['1w', '1m', '3m', '6m', '12m'] as const).map(period => (
                        <button
                            key={period}
                            className={`component-ui-revenue-graph-toggle-btn ${timePeriod === period ? 'active' : ''}`}
                            onClick={() => setTimePeriod(period)}
                        >
                            {period === '1w' ? '1 Week' : period === '1m' ? '1 Month' : period === '3m' ? '3 Months' : period === '6m' ? '6 Months' : '12 Months'}
                        </button>
                    ))}
                </div>
            </div>
            <ResponsiveContainer width="100%" height={chartHeight}>
                {chartType === 'bar' ? (
                    <BarChart data={revenueData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="date" tick={{ fill: '#666', fontSize: 12 }} interval={xAxisInterval} />
                        <YAxis tick={{ fill: '#666', fontSize: 12 }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#fff',
                                border: '1px solid #e0e0e0',
                                borderRadius: '4px',
                                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }}
                            formatter={(value, name) => {
                                if (name === 'actual') return [value, 'Actual Rooms'];
                                return [value, 'Predicted Rooms'];
                            }}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                        <Bar dataKey="actual" fill="#ff6384" name="Actual Rooms" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="predicted" fill="#36a2eb" name="Predicted Rooms" radius={[4, 4, 0, 0]} />
                    </BarChart>
                ) : (
                    <LineChart data={revenueData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="date" tick={{ fill: '#666', fontSize: 12 }} interval={xAxisInterval} />
                        <YAxis tick={{ fill: '#666', fontSize: 12 }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#fff',
                                border: '1px solid #e0e0e0',
                                borderRadius: '4px',
                                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }}
                            formatter={(value) => `$${value}`}
                            labelFormatter={(label) => `Date: ${label}`}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="line" />
                        <Line type="monotone" dataKey="actual" stroke="rgb(255, 99, 132)" dot={false} strokeWidth={2} name="Actual Revenue" />
                        <Line type="monotone" dataKey="predicted" stroke="rgb(54, 162, 235)" dot={false} strokeWidth={2} strokeDasharray="5 5" name="Predicted Revenue" />
                    </LineChart>
                )}
            </ResponsiveContainer>
        </div>
    );
};

export default RevenueGraph;
