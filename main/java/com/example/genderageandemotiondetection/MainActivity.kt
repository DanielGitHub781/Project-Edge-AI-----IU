package com.example.genderageandemotiondetection

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min
import androidx.activity.compose.rememberLauncherForActivityResult
import android.util.Log
import androidx.core.graphics.scale
import androidx.core.graphics.get

class MainActivity : ComponentActivity() {

    private lateinit var ageInterpreter: Interpreter
    private lateinit var genderInterpreter: Interpreter
    private lateinit var emotionInterpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load models
        ageInterpreter = Interpreter(loadModelFile("age_model.tflite"))
        genderInterpreter = Interpreter(loadModelFile("gender_model.tflite"))
        emotionInterpreter = Interpreter(loadModelFile("emotion_model.tflite"))

        setContent {
            MaterialTheme {
                var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
                var age by remember { mutableStateOf<String>("") }
                var gender by remember { mutableStateOf<String>("") }
                var emotion by remember { mutableStateOf<String>("") }
                var errorMsg by remember { mutableStateOf<String?>(null) }

                // New states for inference times
                var ageTime by remember { mutableStateOf<Long?>(null) }
                var genderTime by remember { mutableStateOf<Long?>(null) }
                var emotionTime by remember { mutableStateOf<Long?>(null) }

                val pickImageLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.GetContent()
                ) { uri: Uri? ->
                    try {
                        uri?.let {
                            val original = getBitmapFromUri(it)
                            selectedBitmap = original.copy(Bitmap.Config.ARGB_8888, true)

                            val (a, at) = predictAge(original)
                            val (g, gt) = predictGender(original)
                            val (e, et) = predictEmotion(original)

                            age = a
                            gender = g
                            emotion = e
                            ageTime = at
                            genderTime = gt
                            emotionTime = et

                            errorMsg = null
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                        errorMsg = "Error: ${e.message}"
                    }
                }

                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Button(onClick = { pickImageLauncher.launch("image/*") }) {
                        Text("Choose Image from Gallery")
                    }

                    selectedBitmap?.let { bmp ->
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "Selected Image",
                            modifier = Modifier.size(200.dp)
                        )
                        Text("Age Group: $age (${ageTime ?: "--"} ms)")
                        Text("Gender: $gender (${genderTime ?: "--"} ms)")
                        Text("Emotion: $emotion (${emotionTime ?: "--"} ms)")
                    }

                    errorMsg?.let {
                        Text(" $it", color = androidx.compose.ui.graphics.Color.Red)
                    }
                }
            }
        }
    }
    private fun loadModelFile(modelFile: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun getBitmapFromUri(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT < 28) {
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        } else {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source)
        }
    }

    private fun prepareBitmap(bitmap: Bitmap): Bitmap {
        val bmp = if (bitmap.config != Bitmap.Config.ARGB_8888 || !bitmap.isMutable) {
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            bitmap
        }

        val width = bmp.width
        val height = bmp.height
        val size = min(width, height)
        val xOffset = (width - size) / 2
        val yOffset = (height - size) / 2
        val squareBitmap = Bitmap.createBitmap(bmp, xOffset, yOffset, size, size)

        return squareBitmap.scale(96, 96)
    }

    private fun bitmapToFloatArray(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resized = prepareBitmap(bitmap)
        val input = Array(1) { Array(96) { Array(96) { FloatArray(3) } } }

        for (y in 0 until 96) {
            for (x in 0 until 96) {
                val px = resized[x, y]
                input[0][y][x][0] = ((px shr 16) and 0xFF) / 255.0f
                input[0][y][x][1] = ((px shr 8) and 0xFF) / 255.0f
                input[0][y][x][2] = (px and 0xFF) / 255.0f
            }
        }
        return input
    }

    private fun predictAge(bitmap: Bitmap): Pair<String, Long> {
        val input = bitmapToFloatArray(bitmap)
        val numAgeGroups = 5
        val output = Array(1) { FloatArray(numAgeGroups) }

        val start = System.nanoTime()
        ageInterpreter.run(input, output)
        val end = System.nanoTime()
        val timeMs = (end - start) / 1_000_000

        val predictedIdx = output[0].indices.maxByOrNull { output[0][it] } ?: 0
        val ageGroups = arrayOf("Child", "Teen", "Young Adult", "Adult", "Senior")

        Log.d("AgePred", "Probs: ${output[0].joinToString(", ")} | Time: $timeMs ms")

        return Pair(ageGroups[predictedIdx], timeMs)
    }

    private fun predictGender(bitmap: Bitmap): Pair<String, Long> {
        val input = bitmapToFloatArray(bitmap)
        val output = Array(1) { FloatArray(1) }

        val start = System.nanoTime()
        genderInterpreter.run(input, output)
        val end = System.nanoTime()
        val timeMs = (end - start) / 1_000_000

        val gender = if (output[0][0] < 0.5) "Male" else "Female"
        return Pair(gender, timeMs)
    }

    private fun predictEmotion(bitmap: Bitmap): Pair<String, Long> {
        val input = bitmapToFloatArray(bitmap)
        val numEmotions = 3
        val output = Array(1) { FloatArray(numEmotions) }

        val start = System.nanoTime()
        emotionInterpreter.run(input, output)
        val end = System.nanoTime()
        val timeMs = (end - start) / 1_000_000

        val predictedIdx = output[0].indices.maxByOrNull { output[0][it] } ?: 0
        val emotions = arrayOf("Happy", "Neutral", "Sad")

        Log.d("EmotionPred", "Probs: ${output[0].joinToString(", ")} | Time: $timeMs ms")

        return Pair(emotions[predictedIdx], timeMs)
    }
}









