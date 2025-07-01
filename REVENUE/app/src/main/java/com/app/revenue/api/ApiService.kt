package com.app.revenue.api

import com.app.revenue.data.DataResponse
import com.app.revenue.data.HotelData
import com.app.revenue.data.LoginRequest
import com.app.revenue.data.SummaryResponse
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

interface ApiService {
    @POST("/api/login")
    fun login(@Body loginRequest: LoginRequest): Call<Any>
    
    @POST("/api/logout")
    fun logout(): Call<Any>
    
    @POST("/api/submit")
    fun submitData(@Body hotelData: HotelData): Call<Any>
    
    @GET("/api/data")
    fun getData(
        @Query("page") page: Int = 1,
        @Query("page_size") pageSize: Int = 20,
        @Query("start_date") startDate: String? = null,
        @Query("end_date") endDate: String? = null,
        @Query("actual_or_forecast") actualOrForecast: String? = null,
        @Query("sort_by") sortBy: String? = null,
        @Query("sort_order") sortOrder: String? = null
    ): Call<DataResponse>
    
    @GET("/api/data/{id}")
    fun getDataById(@Path("id") id: Int): Call<HotelData>
    
    @GET("/api/data/summary")
    fun getSummary(
        @Query("start_month") startMonth: String? = null,
        @Query("end_month") endMonth: String? = null,
        @Query("actual_or_forecast") actualOrForecast: String? = null
    ): Call<SummaryResponse>
}

