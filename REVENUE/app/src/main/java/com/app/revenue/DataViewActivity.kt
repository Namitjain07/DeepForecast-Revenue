package com.app.revenue

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.app.revenue.api.RetrofitClient
import com.app.revenue.data.DataResponse
import com.app.revenue.data.HotelData
import com.app.revenue.data.SummaryData
import com.app.revenue.data.SummaryResponse
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class DataViewActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    DataViewScreen(
                        onBack = { finish() }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DataViewScreen(onBack: () -> Unit) {
    var selectedTab by remember { mutableStateOf(0) }
    val tabs = listOf("Records", "Summary")

    Column(modifier = Modifier.fillMaxSize()) {
        // Top App Bar
        TopAppBar(
            title = { Text("Data View") },
            navigationIcon = {
                TextButton(onClick = onBack) {
                    Text("← Back")
                }
            }
        )

        // Tab Row
        TabRow(selectedTabIndex = selectedTab) {
            tabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTab == index,
                    onClick = { selectedTab = index },
                    text = { Text(title) }
                )
            }
        }

        // Tab Content
        when (selectedTab) {
            0 -> RecordsTab()
            1 -> SummaryTab()
        }
    }
}

@Composable
fun RecordsTab() {
    var hotelData by remember { mutableStateOf<List<HotelData>>(emptyList()) }
    var isLoading by remember { mutableStateOf(false) }
    var currentPage by remember { mutableStateOf(1) }
    var totalPages by remember { mutableStateOf(1) }
    var filterType by remember { mutableStateOf("") }

    fun loadData(page: Int = 1, actualOrForecast: String? = null) {
        isLoading = true
        RetrofitClient.instance.getData(
            page = page,
            pageSize = 20,
            actualOrForecast = actualOrForecast?.takeIf { it.isNotEmpty() }
        ).enqueue(object : Callback<DataResponse> {
            override fun onResponse(call: Call<DataResponse>, response: Response<DataResponse>) {
                isLoading = false
                if (response.isSuccessful) {
                    response.body()?.let { dataResponse ->
                        hotelData = dataResponse.data
                        currentPage = dataResponse.pagination.page
                        totalPages = dataResponse.pagination.totalPages
                    }
                }
            }

            override fun onFailure(call: Call<DataResponse>, t: Throwable) {
                isLoading = false
            }
        })
    }

    LaunchedEffect(Unit) {
        loadData()
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        // Filter Section
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = filterType,
                onValueChange = { filterType = it },
                label = { Text("Filter (Actual/Forecast)") },
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = { loadData(1, filterType) }
            ) {
                Text("Filter")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (isLoading) {
            Box(
                modifier = Modifier.fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(hotelData) { data ->
                    HotelDataCard(data)
                }
            }
        }

        // Pagination
        if (totalPages > 1) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = { loadData(currentPage - 1, filterType.takeIf { it.isNotEmpty() }) },
                    enabled = currentPage > 1
                ) {
                    Text("Previous")
                }
                
                Text("Page $currentPage of $totalPages")
                
                Button(
                    onClick = { loadData(currentPage + 1, filterType.takeIf { it.isNotEmpty() }) },
                    enabled = currentPage < totalPages
                ) {
                    Text("Next")
                }
            }
        }
    }
}

@Composable
fun SummaryTab() {
    var summaryData by remember { mutableStateOf<List<SummaryData>>(emptyList()) }
    var isLoading by remember { mutableStateOf(false) }

    fun loadSummary() {
        isLoading = true
        RetrofitClient.instance.getSummary().enqueue(object : Callback<SummaryResponse> {
            override fun onResponse(call: Call<SummaryResponse>, response: Response<SummaryResponse>) {
                isLoading = false
                if (response.isSuccessful) {
                    response.body()?.let { summaryResponse ->
                        summaryData = summaryResponse.data
                    }
                }
            }

            override fun onFailure(call: Call<SummaryResponse>, t: Throwable) {
                isLoading = false
            }
        })
    }

    LaunchedEffect(Unit) {
        loadSummary()
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        if (isLoading) {
            Box(
                modifier = Modifier.fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(summaryData) { data ->
                    SummaryDataCard(data)
                }
            }
        }
    }
}

@Composable
fun HotelDataCard(data: HotelData) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Date: ${data.arrivalDate}",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text("Type: ${data.actualOrForecast}")
            Text("Rooms Sold: ${data.roomsSold}")
            Text("Occupancy: ${data.occupancyPercentage}%")
            Text("Revenue: $${String.format("%.2f", data.roomRevenue)}")
            Text("ARR: $${String.format("%.2f", data.arr)}")
        }
    }
}

@Composable
fun SummaryDataCard(data: SummaryData) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Month: ${data.month}",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text("Type: ${data.actualOrForecast}")
            Text("Total Rooms Sold: ${data.totalRoomsSold}")
            Text("Total Revenue: $${String.format("%.2f", data.totalRevenue)}")
            Text("Avg Occupancy: ${String.format("%.2f", data.avgOccupancy)}%")
            Text("Avg ARR: $${String.format("%.2f", data.avgArr)}")
        }
    }
}
