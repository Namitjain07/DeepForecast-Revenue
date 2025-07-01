package com.app.revenue.data

import com.google.gson.annotations.SerializedName

data class HotelData(
    @SerializedName("Total Room Inventory")
    val totalRoomInventory: Int,
    @SerializedName("Rooms Sold")
    val roomsSold: Int,
    @SerializedName("Arrival Rooms")
    val arrivalRooms: Int,
    @SerializedName("Compliment Rooms")
    val complimentRooms: Int,
    @SerializedName("House Use")
    val houseUse: Int,
    @SerializedName("Individual Confirm")
    val individualConfirm: Int,
    @SerializedName("Occupancy %")
    val occupancyPercentage: Double,
    @SerializedName("Room Revenue")
    val roomRevenue: Double,
    @SerializedName("ARR")
    val arr: Double,
    @SerializedName("Departure Rooms")
    val departureRooms: Int,
    @SerializedName("OOO Rooms")
    val oooRooms: Int,
    @SerializedName("Pax")
    val pax: Int,
    @SerializedName("snapshot_date")
    val snapshotDate: String,
    @SerializedName("arrival_date")
    val arrivalDate: String,
    @SerializedName("actual_or_forecast")
    val actualOrForecast: String,
    @SerializedName("Day")
    val day: String,
    @SerializedName("revenue_diff")
    val revenueDiff: Double
)
