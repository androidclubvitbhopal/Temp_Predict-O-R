# Weather-Prediction

![Android-App-Homepage](/images/appMainActivity.jpg)

It is going to be an Android application that is going to predict weather conditions depending on the parameters given to the integrated machine learning model, that is, temperature and humidity. 

## Prerequisites
+ Your PC should have Android Studio installed.
+ An Android Phone (or you may install and test the app on a virtual device)

## Features

+ Predicts weather conditions based on temperature and humidity
+ Integrates TensorFlow Lite models on Android Devices
+ Provides visual feedback with weather icons for predicted conditions

## Installation

+ Clone this repository to your local machine using 'git clone'.
+ Open the project in Android Studio.
+ Build and run the application on an Android device or emulator.

## Usage

+ Enter the temperature and humidity values.
+ Tap the "Predict" button to obtain the predicted weather condition.
+ View the predicted weather condition along with a corresponding weather icon.

## Libraries Used

+ TensorFlow Lite: For running machine learning models on Android.
+ All other minor dependencies are specified in the [build.gradle file](https://github.com/adsmehra/IOT-Weather-Predictor/blob/main/app/build.gradle.kts).

## Flow diagram of the model
![Flow-Diagram](/images/mlAndroidFlowDiagram.jpg)

## Dependencies 
### For TensorFlow

```
    implementation("org.tensorflow:tensorflow-lite-support:0.1.0")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.1.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.3.0")
````

## Import TensorFlow Interpreter
```kotlin
import org.tensorflow.lite.Interpreter
```

# Weather-Predictor Model access
You can get the model from "asset" section or follow the link -> [Tflite file](WeatherPrediction/app/src/main/assets/Weather_predictor.tflite)

## Import the ml file from the assets directory 
```kotlin
    import com.example.demo.ml.WeatherPredictor
```


## Main onCreate Method

Here, We:
+ Initialize our UI Components,
+ We then provide input data from the user to our TFLite file for processing,
+ Throughout, we use Exception Handling to make sure we avoid errors.

### Loading Tensorflow model
+ Used to access the the tflite file from assets folder and create an inputstream for it
+ Creates a channel to pass the inputs to the model
+ Declares startOffset and length of the file
+ Returns a MappedByteBuffer for the data in the model
```kotlin
private fun loadModelFile(): ByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd("Weather_predictor.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declareLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength)
    }
```

### Predict Weather fun
+ Takes temperature (in Celsius) and humidity (in percentage) as inputs.
+ Uses a TensorFlow Lite model to predict the weather condition.
+ Returns the predicted weather condition (Sunny, Cloudy, Partly Cloudy, Rainy, Cold).

```kotlin
    private fun predictWeather(temperatureC: Float, humidityPer: Float): String {
    val byteBuffer =
        ByteBuffer.allocateDirect(2 * 4) // Assuming 2 input features and 4 bytes per float
    byteBuffer.order(ByteOrder.nativeOrder())
    val inputFeature0 = byteBuffer.asFloatBuffer()
    inputFeature0.put(floatArrayOf(temperatureC, humidityPer))
    byteBuffer.rewind()

    // Runs model inference and gets result.
    val outputs = Array(1) { FloatArray(5) }
    tflite.run(inputFeature0, outputs)

    val maxIndex = outputs[0].indices.maxByOrNull { outputs[0][it] } ?: -1
    val predictedClassIndex = if (maxIndex != -1) maxIndex else 0

    val weatherConditions =
        arrayOf("Cloudy", "Cold", "Rainy", "Sunny", "Partly Cloudy")
    val predictedWeather = weatherConditions[predictedClassIndex]
    Log.d("Weather", "Predicted: $predictedClassIndex")

    return predictedWeather

}


