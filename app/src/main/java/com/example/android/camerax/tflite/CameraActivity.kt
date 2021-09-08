/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camerax.tflite

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.drawable.ColorDrawable
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.view.ViewGroup
import android.webkit.MimeTypeMap
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.android.example.camerax.tflite.databinding.ActivityCameraBinding
import com.example.android.camera.utils.YuvToRgbConverter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random


/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

    private lateinit var activityCameraBinding: ActivityCameraBinding

    private lateinit var bitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer = TensorImage(DataType.UINT8)


    private lateinit var outputDirectory: Uri

    /** Milliseconds used for UI animations */
     val ANIMATION_FAST_MILLIS = 50L
     val ANIMATION_SLOW_MILLIS = 100L
    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService
    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(
                tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(NnApiDelegate()))
    }

    private val detector by lazy {
        ObjectDetectionHelper(
            tflite,
            FileUtil.loadLabels(this, LABELS_PATH)
        )
    }

    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        outputDirectory = createNewUriDirect(this,"LaneVision",  "LV_${System.currentTimeMillis()}.mp4")!!
        // Initialize our background executor
        cameraExecutor = Executors.newSingleThreadExecutor()
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

        activityCameraBinding.cameraCaptureButton.setOnClickListener {
            saveImage()
        }
    }
 //

  //
    // We can only change the foreground Drawable using API level 23+ API

    private fun saveImage(){
        // Get a stable reference of the modifiable image capture use case
        // Get a stable reference of the modifiable image capture use case
        imageCapture?.let { imageCapture ->

            // Create output file to hold the image
            val uriCV = createNewUri(this,"LaneVision",  "LV_${System.currentTimeMillis()}.jpeg")!!

            // Setup image capture metadata
            val metadata = ImageCapture.Metadata().apply {

                // Mirror image when using the front camera
                isReversedHorizontal = lensFacing == CameraSelector.LENS_FACING_FRONT
            }
            val resolver = applicationContext.contentResolver
           // resolver.openOutputStream(outputDirectory, "rw").use {
                // Create output options object which contains file + metadata
                val outputOptions = ImageCapture.OutputFileOptions.Builder(resolver, uriCV.first!!, uriCV.second )
                    .setMetadata(metadata)
                    .build()

            // Setup image capture listener which is triggered after photo has been taken
            imageCapture.takePicture(
                outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
                    override fun onError(exc: ImageCaptureException) {
                        Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    }

                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {

                        Log.d(TAG, "Photo capture succeeded: $outputDirectory")

                        // We can only change the foreground Drawable using API level 23+ API
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            // Update the gallery thumbnail with latest picture taken
                          //  setGalleryThumbnail(savedUri)
                        }

                        // Implicit broadcasts will be ignored for devices running API level >= 24
                        // so if you only target API level 24+ you can remove this statement
                        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
                            sendBroadcast(
                                Intent(android.hardware.Camera.ACTION_NEW_PICTURE, outputDirectory)
                            )
                        }

                    }
                })

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {

                // Display flash animation to indicate that photo was captured
                activityCameraBinding.root.postDelayed({
                    activityCameraBinding.root.foreground = ColorDrawable(Color.WHITE)
                    activityCameraBinding.root.postDelayed(
                        { activityCameraBinding.root.foreground = null }, ANIMATION_FAST_MILLIS)
                }, ANIMATION_SLOW_MILLIS)
            }
        }
    }
    fun analytics(it: View){

        // Disable all camera controls
        it.isEnabled = false

        if (pauseAnalysis) {
            // If image analysis is in paused state, resume it
            pauseAnalysis = false
            activityCameraBinding.imagePredicted.visibility = View.GONE

        } else {
            // Otherwise, pause image analysis and freeze image
            pauseAnalysis = true
            val matrix = Matrix().apply {
                postRotate(imageRotationDegrees.toFloat())
                if (isFrontFacing) postScale(-1f, 1f)
            }
            val uprightImage = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
            activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)
            activityCameraBinding.imagePredicted.visibility = View.VISIBLE
        }

        // Re-enable camera controls
        it.isEnabled = true
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({

            // Camera provider is now guaranteed to be available
            cameraProvider = cameraProviderFuture.get()
            bindCameraCapture()

        }, ContextCompat.getMainExecutor(this))
    }


    private fun bindCameraCapture(){

        // Get screen metrics used to setup camera for full screen resolution
        val metrics = windowManager.getCurrentWindowMetrics().bounds
        Log.d(TAG, "Screen metrics: ${metrics.width()} x ${metrics.height()}")

        val screenAspectRatio = aspectRatio(metrics.width(), metrics.height())
        Log.d(TAG, "Preview aspect ratio: $screenAspectRatio")
        val rotation = activityCameraBinding.viewFinder.display.rotation


        var frameCounter = 0
        var lastFpsTimestamp = System.currentTimeMillis()
        val converter = YuvToRgbConverter(this)

        // Create a new camera selector each time, enforcing lens facing
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        // Set up the view finder use case to display camera preview
        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        // ImageCapture
        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            // We request aspect ratio but no resolution to match preview config, but letting
            // CameraX optimize for whatever specific resolution best fits our use cases
            .setTargetAspectRatio(screenAspectRatio)
            // Set initial target rotation, we will have to call this again if rotation changes
            // during the lifecycle of this use case
            .setTargetRotation(rotation)
            .build()

        // Set up the image analysis use case which will process frames in real time
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                    if (!::bitmapBuffer.isInitialized) {
                        // The image rotation and RGB image buffer are initialized only once
                        // the analyzer has started running
                        imageRotationDegrees = image.imageInfo.rotationDegrees
                        bitmapBuffer = Bitmap.createBitmap(
                            image.width, image.height, Bitmap.Config.ARGB_8888)
                    }

                    // Early exit: image analysis is in paused state
                    if (pauseAnalysis) {
                        image.close()
                        return@Analyzer
                    }

                    // Convert the image to RGB and place it in our shared buffer
                    image.use { converter.yuvToRgb(image.image!!, bitmapBuffer) }

                    // Process the image in Tensorflow
                    val tfImage =  tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })

                    // Perform the object detection for the current frame
                    val predictions = detector.predict(tfImage)

                    // Report only the top prediction
                    reportPrediction(predictions.maxByOrNull { it.score })

                    // Compute the FPS of the entire pipeline
                    val frameCount = 10
                    if (++frameCounter % frameCount == 0) {
                        frameCounter = 0
                        val now = System.currentTimeMillis()
                        val delta = now - lastFpsTimestamp
                        val fps = 1000 * frameCount.toFloat() / delta
                        Log.d(TAG, "FPS: ${"%.02f".format(fps)}")
                        lastFpsTimestamp = now
                    }
                })

            }





        // Apply declared configs to CameraX using the same lifecycle owner
        cameraProvider!!.unbindAll()
        camera = cameraProvider!!.bindToLifecycle(
            this as LifecycleOwner, cameraSelector, preview,imageCapture, imageAnalyzer)

        // Use the camera object to link our preview use case with the view
        preview!!.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)
    }

    private fun reportPrediction(
        prediction: ObjectDetectionHelper.ObjectPrediction?
    ) = activityCameraBinding.viewFinder.post {

        // Early exit: if prediction is not good enough, don't report it
        if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
            activityCameraBinding.boxPrediction.visibility = View.GONE
            activityCameraBinding.textPrediction.visibility = View.GONE
            return@post
        }

        // Location has to be mapped to our local coordinates
        val location = mapOutputCoordinates(prediction.location)

        // Update the text and UI
        activityCameraBinding.textPrediction.text = "${"%.2f".format(prediction.score)} ${prediction.label}"
        (activityCameraBinding.boxPrediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = location.top.toInt()
            leftMargin = location.left.toInt()
            width = min(activityCameraBinding.viewFinder.width, location.right.toInt() - location.left.toInt())
            height = min(activityCameraBinding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
        }

        // Make sure all UI elements are visible
        activityCameraBinding.boxPrediction.visibility = View.VISIBLE
        activityCameraBinding.textPrediction.visibility = View.VISIBLE
    }

    /**
     * Helper function used to map the coordinates for objects coming out of
     * the model into the coordinates that the user sees on the screen.
     */
    private fun mapOutputCoordinates(location: RectF): RectF {

        // Step 1: map location to the preview coordinates
        val previewLocation = RectF(
            location.left * activityCameraBinding.viewFinder.width,
            location.top * activityCameraBinding.viewFinder.height,
            location.right * activityCameraBinding.viewFinder.width,
            location.bottom * activityCameraBinding.viewFinder.height
        )

        // Step 2: compensate for camera sensor orientation and mirroring
        val isFrontFacing = lensFacing == CameraSelector.LENS_FACING_FRONT
        val correctedLocation = if (isFrontFacing) {
            RectF(
                activityCameraBinding.viewFinder.width - previewLocation.right,
                previewLocation.top,
                activityCameraBinding.viewFinder.width - previewLocation.left,
                previewLocation.bottom)
        } else {
            previewLocation
        }

        // Step 3: compensate for 1:1 to 4:3 aspect ratio conversion + small margin
        val margin = 0.1f
        val requestedRatio = 4f / 3f
        val midX = (correctedLocation.left + correctedLocation.right) / 2f
        val midY = (correctedLocation.top + correctedLocation.bottom) / 2f
        return if (activityCameraBinding.viewFinder.width < activityCameraBinding.viewFinder.height) {
            RectF(
                midX - (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY - (1f - margin) * correctedLocation.height() / 2f,
                midX + (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY + (1f - margin) * correctedLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * correctedLocation.width() / 2f,
                midY - (1f + margin) * requestedRatio * correctedLocation.height() / 2f,
                midX + (1f - margin) * correctedLocation.width() / 2f,
                midY + (1f + margin) * requestedRatio * correctedLocation.height() / 2f
            )
        }
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode)
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }



    /**
     *  [androidx.camera.core.ImageAnalysis.Builder] requires enum value of
     *  [androidx.camera.core.AspectRatio]. Currently it has values of 4:3 & 16:9.
     *
     *  Detecting the most suitable ratio for dimensions provided in @params by counting absolute
     *  of preview ratio to one of the provided values.
     *
     *  @param width - preview width
     *  @param height - preview height
     *  @return suitable aspect ratio
     */
    private fun aspectRatio(width: Int, height: Int): Int {
        val previewRatio = max(width, height).toDouble() / min(width, height)
        if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
            return AspectRatio.RATIO_4_3
        }
        return AspectRatio.RATIO_16_9
    }
    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val ACCURACY_THRESHOLD = 0.5f
        private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
        private const val RATIO_4_3_VALUE = 4.0 / 3.0
        private const val RATIO_16_9_VALUE = 16.0 / 9.0
    }
}
