package com.app.revenue

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.app.revenue.api.RetrofitClient
import com.app.revenue.data.HotelData
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    HotelDataForm(
                        onSubmit = { hotelData ->
                            submitData(hotelData)
                        },
                        onLogout = {
                            performLogout()
                        },
                        onViewData = {
                            val intent = Intent(this@MainActivity, DataViewActivity::class.java)
                            startActivity(intent)
                        }
                    )
                }
            }
        }
    }

    private fun submitData(hotelData: HotelData) {
        RetrofitClient.instance.submitData(hotelData).enqueue(object : Callback<Any> {
            override fun onResponse(call: Call<Any>, response: Response<Any>) {
                if (response.isSuccessful) {
                    Toast.makeText(this@MainActivity, "Data submitted!", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, "Error occurred: ${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Any>, t: Throwable) {
                Toast.makeText(this@MainActivity, "Error occurred: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun performLogout() {
        RetrofitClient.instance.logout().enqueue(object : Callback<Any> {
            override fun onResponse(call: Call<Any>, response: Response<Any>) {
                Toast.makeText(this@MainActivity, "Logged out successfully", Toast.LENGTH_SHORT).show()
                val intent = Intent(this@MainActivity, LoginActivity::class.java)
                startActivity(intent)
                finish()
            }

            override fun onFailure(call: Call<Any>, t: Throwable) {
                Toast.makeText(this@MainActivity, "Logout error: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HotelDataForm(onSubmit: (HotelData) -> Unit, onLogout: () -> Unit, onViewData: () -> Unit) {
    var totalRoomInventory by remember { mutableStateOf("") }
    var roomsSold by remember { mutableStateOf("") }
    var arrivalRooms by remember { mutableStateOf("") }
    var complimentRooms by remember { mutableStateOf("") }
    var houseUse by remember { mutableStateOf("") }
    var individualConfirm by remember { mutableStateOf("") }
    var occupancyPercentage by remember { mutableStateOf("") }
    var roomRevenue by remember { mutableStateOf("") }
    var arr by remember { mutableStateOf("") }
    var departureRooms by remember { mutableStateOf("") }
    var oooRooms by remember { mutableStateOf("") }
    var pax by remember { mutableStateOf("") }
    var snapshotDate by remember { mutableStateOf("") }
    var arrivalDate by remember { mutableStateOf("") }
    var actualOrForecast by remember { mutableStateOf("") }
    var day by remember { mutableStateOf("") }
    var revenueDiff by remember { mutableStateOf("") }

    var formError by remember { mutableStateOf("") }

    // Define validateForm function outside of the button click handler
    fun validateForm(): Boolean {
        return totalRoomInventory.isNotEmpty() && 
                roomsSold.isNotEmpty() && 
                arrivalRooms.isNotEmpty() && 
                complimentRooms.isNotEmpty() && 
                houseUse.isNotEmpty() && 
                individualConfirm.isNotEmpty() && 
                occupancyPercentage.isNotEmpty() && 
                roomRevenue.isNotEmpty() && 
                arr.isNotEmpty() && 
                departureRooms.isNotEmpty() && 
                oooRooms.isNotEmpty() && 
                pax.isNotEmpty() && 
                snapshotDate.isNotEmpty() && 
                arrivalDate.isNotEmpty() && 
                actualOrForecast.isNotEmpty() && 
                day.isNotEmpty() && 
                revenueDiff.isNotEmpty()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Hotel Revenue Form",
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = onViewData,
                modifier = Modifier.padding(end = 8.dp)
            ) {
                Text("View Data")
            }
            OutlinedButton(onClick = onLogout) {
                Text("Logout")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (formError.isNotEmpty()) {
            Text(
                text = formError,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(vertical = 8.dp)
            )
        }

        NumberField("Total Room Inventory", totalRoomInventory) { totalRoomInventory = it }
        NumberField("Rooms Sold", roomsSold) { roomsSold = it }
        NumberField("Arrival Rooms", arrivalRooms) { arrivalRooms = it }
        NumberField("Compliment Rooms", complimentRooms) { complimentRooms = it }
        NumberField("House Use", houseUse) { houseUse = it }
        NumberField("Individual Confirm", individualConfirm) { individualConfirm = it }
        DecimalField("Occupancy %", occupancyPercentage) { occupancyPercentage = it }
        DecimalField("Room Revenue", roomRevenue) { roomRevenue = it }
        DecimalField("ARR", arr) { arr = it }
        NumberField("Departure Rooms", departureRooms) { departureRooms = it }
        NumberField("OOO Rooms", oooRooms) { oooRooms = it }
        NumberField("Pax", pax) { pax = it }
        
        TextField("Snapshot Date (YYYY-MM-DD)", snapshotDate) { snapshotDate = it }
        TextField("Arrival Date (YYYY-MM-DD)", arrivalDate) { arrivalDate = it }
        TextField("Actual or Forecast", actualOrForecast) { actualOrForecast = it }
        TextField("Day", day) { day = it }
        DecimalField("Revenue Diff", revenueDiff) { revenueDiff = it }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                if (validateForm()) {
                    try {
                        val hotelData = HotelData(
                            totalRoomInventory = totalRoomInventory.toInt(),
                            roomsSold = roomsSold.toInt(),
                            arrivalRooms = arrivalRooms.toInt(),
                            complimentRooms = complimentRooms.toInt(),
                            houseUse = houseUse.toInt(),
                            individualConfirm = individualConfirm.toInt(),
                            occupancyPercentage = occupancyPercentage.toDouble(),
                            roomRevenue = roomRevenue.toDouble(),
                            arr = arr.toDouble(),
                            departureRooms = departureRooms.toInt(),
                            oooRooms = oooRooms.toInt(),
                            pax = pax.toInt(),
                            snapshotDate = snapshotDate,
                            arrivalDate = arrivalDate,
                            actualOrForecast = actualOrForecast,
                            day = day,
                            revenueDiff = revenueDiff.toDouble()
                        )
                        onSubmit(hotelData)
                        formError = ""
                    } catch (e: NumberFormatException) {
                        formError = "Please enter valid numbers in all numeric fields."
                    }
                } else {
                    formError = "All fields are required and must be valid."
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp)
        ) {
            Text("Submit")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NumberField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = { newValue ->
            if (newValue.isEmpty() || newValue.all { it.isDigit() }) {
                onValueChange(newValue)
            }
        },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DecimalField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = { newValue ->
            if (newValue.isEmpty() || newValue.matches(Regex("^\\d*\\.?\\d*$"))) {
                onValueChange(newValue)
            }
        },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    )
}