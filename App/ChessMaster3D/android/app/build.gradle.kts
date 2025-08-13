plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.chess_master_3d"
    compileSdk = flutter.compileSdkVersion
    // Override Flutter's default NDK with plugin-required version
    ndkVersion = "27.0.12077973"
    // Limit ABIs during development to speed up builds (emulator is x86_64) while still producing a single APK name.

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.chess_master_3d"
        // Configure SDK & versioning from Flutter
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        ndk {
            // Focus on physical device performance: build only arm64 for fastest iteration.
            // To test on x86_64 emulator again, temporarily change this list to listOf("x86_64") or add back both.
                abiFilters.clear()
                // Include emulator (x86_64) plus physical device ABIs for local testing.
                abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}
