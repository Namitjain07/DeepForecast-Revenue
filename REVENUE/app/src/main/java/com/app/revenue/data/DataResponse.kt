package com.app.revenue.data

import com.google.gson.annotations.SerializedName

data class DataResponse(
    val data: List<HotelData>,
    val pagination: PaginationMetadata
)

data class PaginationMetadata(
    val page: Int,
    val pageSize: Int,
    val totalPages: Int,
    val totalRecords: Int,
    val hasNext: Boolean,
    val hasPrevious: Boolean
)

data class SummaryData(
    @SerializedName("month")
    val month: String,
    @SerializedName("total_rooms_sold")
    val totalRoomsSold: Int,
    @SerializedName("total_revenue")
    val totalRevenue: Double,
    @SerializedName("avg_occupancy")
    val avgOccupancy: Double,
    @SerializedName("avg_arr")
    val avgArr: Double,
    @SerializedName("actual_or_forecast")
    val actualOrForecast: String
)

data class SummaryResponse(
    val data: List<SummaryData>
)
