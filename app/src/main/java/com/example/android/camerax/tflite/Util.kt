package com.example.android.camerax.tflite

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File

fun createNewUriDirect(
    context: Context,
    folderName: String,
    fileName: String = "",
    sharedDirectory: String = Environment.DIRECTORY_MOVIES
): Uri? {
    val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
    } else {
        MediaStore.Video.Media.EXTERNAL_CONTENT_URI
    }
    val dirDest = File(sharedDirectory, folderName)
    val date = System.currentTimeMillis()
    val extension = "mp4"
    var fileNameIs = "${folderName}_${System.currentTimeMillis()}.$extension"
    if (fileName.isNotBlank()) {
        fileNameIs = fileName
    }

    val newImage = ContentValues().apply {
        put(MediaStore.Video.Media.DISPLAY_NAME, fileNameIs)
        put(MediaStore.MediaColumns.MIME_TYPE, "video/$extension")
        put(MediaStore.MediaColumns.DATE_ADDED, date)
        put(MediaStore.MediaColumns.DATE_MODIFIED, date)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.MediaColumns.RELATIVE_PATH, "$dirDest${File.separator}")

        }
    }
    return context.contentResolver.insert(collection, newImage)
}


fun createNewUri(
    context: Context,
    folderName: String,
    fileName: String = "",
    sharedDirectory: String = Environment.DIRECTORY_PICTURES
): Pair<Uri?, ContentValues> {
    val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
    } else {
        MediaStore.Images.Media.EXTERNAL_CONTENT_URI
    }
    val dirDest = File(sharedDirectory, folderName)
    val date = System.currentTimeMillis()
    val extension = "jpeg"
    var fileNameIs = "${folderName}_${System.currentTimeMillis()}.$extension"
    if (fileName.isNotBlank()) {
        fileNameIs = fileName
    }

    val newImage = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, fileNameIs)
        put(MediaStore.MediaColumns.MIME_TYPE, "image/$extension")
        put(MediaStore.MediaColumns.DATE_ADDED, date)
        put(MediaStore.MediaColumns.DATE_MODIFIED, date)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.MediaColumns.RELATIVE_PATH, "$dirDest${File.separator}")
            //todo need to change
         //   put(MediaStore.Video.Media.IS_PENDING, 1)
        }
    }
    val uri = context.contentResolver.insert(collection, newImage)
    return Pair(uri, newImage)
}