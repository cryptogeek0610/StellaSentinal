# Mobile App Architecture Plan

## Overview

This document outlines the architecture for deploying StellaSentinal anomaly detection to mobile devices (iOS/Android) using ONNX Runtime Mobile SDK.

## Use Cases

### Primary: Field Technician App

**Scenario:** IT support technicians diagnose device issues on-site

**Features:**
- Scan device QR code / enter device ID
- Offline anomaly scoring (no network required)
- Historical trend visualization
- Troubleshooting recommendations (via local LLM or cached responses)
- Flag for detailed investigation
- Sync results when online

**Value Proposition:**
- **Instant results:** No waiting for API calls
- **Works anywhere:** Airport tarmac, warehouse floor, remote locations
- **Privacy-friendly:** Device data stays local
- **Cost-effective:** No inference API costs

### Secondary: Executive Dashboard App

**Scenario:** Management views fleet health on mobile

**Features:**
- Fleet overview (device count, anomaly rate)
- Critical alerts
- Geographic heatmap
- Drill-down to device details
- Offline mode with sync

## Technology Stack

### Framework: React Native

**Rationale:**
- Single codebase for iOS + Android
- Team already knows React/TypeScript
- Large ecosystem of packages
- Good performance for business apps

**Alternative Considered:** Flutter
- Rejected: Team unfamiliar with Dart, React expertise already exists

### ML Inference: ONNX Runtime Mobile

**iOS:**
```
onnxruntime-react-native >= 1.16.0
  └─ ONNX Runtime Objective-C SDK
     └─ CoreML acceleration (Apple Neural Engine)
```

**Android:**
```
onnxruntime-react-native >= 1.16.0
  └─ ONNX Runtime Android SDK
     └─ NNAPI acceleration (Android Neural API)
```

### State Management: Redux Toolkit + React Query

**Rationale:**
- Redux: Global state for models, settings, offline queue
- React Query: Server sync, cache management
- Persistence: redux-persist for offline data

### Local Storage: SQLite (via react-native-sqlite-storage)

**Schema:**
```sql
CREATE TABLE models (
  id TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  model_data BLOB NOT NULL,
  feature_count INTEGER,
  created_at INTEGER,
  last_used INTEGER
);

CREATE TABLE devices (
  device_id TEXT PRIMARY KEY,
  metadata TEXT, -- JSON blob
  last_scored INTEGER,
  cached_score REAL,
  cached_label INTEGER
);

CREATE TABLE anomaly_results (
  id TEXT PRIMARY KEY,
  device_id TEXT,
  score REAL,
  label INTEGER,
  timestamp INTEGER,
  synced INTEGER DEFAULT 0,
  FOREIGN KEY(device_id) REFERENCES devices(device_id)
);

CREATE TABLE sync_queue (
  id TEXT PRIMARY KEY,
  operation TEXT, -- 'score', 'resolve', 'note'
  payload TEXT,
  created_at INTEGER,
  retry_count INTEGER DEFAULT 0
);
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Mobile App (React Native)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Scanner    │  │   Device     │  │  Dashboard   │      │
│  │   Screen     │  │   Detail     │  │   Screen     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                           │                                   │
│         ┌─────────────────▼─────────────────┐                │
│         │    State Management (Redux)       │                │
│         │  - Model state                    │                │
│         │  - Device cache                   │                │
│         │  - Offline queue                  │                │
│         └────────┬──────────────────────────┘                │
│                  │                                            │
│    ┌─────────────┼─────────────┐                             │
│    │             │             │                             │
│    ▼             ▼             ▼                             │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐                       │
│ │ ONNX    │  │ SQLite  │  │  API    │                       │
│ │ Runtime │  │ Storage │  │ Client  │                       │
│ └─────────┘  └─────────┘  └─────────┘                       │
│                                │                              │
├────────────────────────────────┼──────────────────────────────┤
│       Device Features          │                              │
├────────────────────────────────┼──────────────────────────────┤
│  - CoreML (iOS)                │                              │
│  - NNAPI (Android)             │                              │
│  - Camera (QR scanning)        │                              │
│  - Background Sync             │                              │
└────────────────────────────────┼──────────────────────────────┘
                                 │
                                 │ Sync when online
                                 │
                        ┌────────▼────────┐
                        │  Backend API    │
                        │  /api/models    │
                        │  /api/devices   │
                        │  /api/anomalies │
                        └─────────────────┘
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Project Setup

```bash
# Initialize React Native project
npx react-native init StellaSentinalMobile --template react-native-template-typescript

# Install dependencies
npm install @reduxjs/toolkit react-redux redux-persist
npm install @tanstack/react-query
npm install onnxruntime-react-native
npm install react-native-sqlite-storage
npm install react-native-fs  # For model file management
npm install react-native-camera  # QR code scanning
npm install react-native-chart-kit  # Charting
npm install @react-navigation/native @react-navigation/stack
```

#### 1.2 ONNX Runtime Integration

**Module:** `src/services/OnnxInference.ts`

```typescript
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

export class OnnxInferenceService {
  private session: InferenceSession | null = null;
  private featureCount: number = 0;

  async loadModel(modelPath: string): Promise<void> {
    try {
      // Load ONNX model from local storage
      const modelData = await RNFS.readFile(modelPath, 'base64');

      this.session = await InferenceSession.create(modelData, {
        executionProviders: ['coreml', 'nnapi', 'cpu'],  // Try hardware acceleration first
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
      });

      // Extract feature count from model metadata
      this.featureCount = this.getFeatureCount();

      console.log('ONNX model loaded successfully');
      console.log('Execution providers:', this.session.executionProviders);
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  async predict(features: number[]): Promise<{ score: number; label: number }> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    if (features.length !== this.featureCount) {
      throw new Error(
        `Expected ${this.featureCount} features, got ${features.length}`
      );
    }

    // Convert to Float32Array and create tensor
    const inputTensor = new Tensor('float32', new Float32Array(features), [
      1,
      this.featureCount,
    ]);

    // Run inference
    const feeds = { float_input: inputTensor };
    const results = await this.session.run(feeds);

    // Extract outputs
    const predictions = results.output_label.data as Float32Array;
    const scores = results.output_score?.data as Float32Array;

    return {
      label: predictions[0],
      score: scores ? scores[0] : 0,
    };
  }

  async predictBatch(
    featuresArray: number[][]
  ): Promise<Array<{ score: number; label: number }>> {
    // Batch inference for multiple devices
    const results = await Promise.all(
      featuresArray.map(features => this.predict(features))
    );
    return results;
  }

  getModelInfo(): { featureCount: number; providers: string[] } {
    return {
      featureCount: this.featureCount,
      providers: this.session?.executionProviders || [],
    };
  }

  private getFeatureCount(): number {
    // Extract from model input shape
    const inputMeta = this.session?.inputNames[0];
    // Parse metadata or return default
    return 25; // TODO: Extract from ONNX metadata
  }

  async dispose(): void {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}

// Singleton instance
export const onnxInference = new OnnxInferenceService();
```

#### 1.3 Model Sync Service

**Module:** `src/services/ModelSync.ts`

```typescript
import { onnxInference } from './OnnxInference';
import { apiClient } from './ApiClient';
import RNFS from 'react-native-fs';
import { SQLiteDatabase } from './Database';

export interface ModelManifest {
  version: string;
  modelUrl: string;
  featureCount: number;
  checksum: string;
  sizeBytes: number;
}

export class ModelSyncService {
  private modelDir = `${RNFS.DocumentDirectoryPath}/models`;
  private currentVersion: string | null = null;

  async initialize(): Promise<void> {
    // Create model directory
    await RNFS.mkdir(this.modelDir);

    // Load existing model if available
    const localModel = await this.getLocalModel();
    if (localModel) {
      await this.loadModel(localModel.path);
      this.currentVersion = localModel.version;
    }
  }

  async checkForUpdates(): Promise<boolean> {
    try {
      // Fetch latest model manifest from server
      const manifest: ModelManifest = await apiClient.get('/api/models/latest');

      // Compare versions
      if (manifest.version !== this.currentVersion) {
        console.log(`New model available: ${manifest.version}`);
        return true;
      }

      return false;
    } catch (error) {
      console.error('Failed to check for model updates:', error);
      return false;
    }
  }

  async downloadAndInstall(
    manifest: ModelManifest,
    onProgress?: (progress: number) => void
  ): Promise<void> {
    const modelPath = `${this.modelDir}/model_${manifest.version}.onnx`;

    // Download model
    console.log('Downloading model:', manifest.modelUrl);

    const download = RNFS.downloadFile({
      fromUrl: manifest.modelUrl,
      toFile: modelPath,
      progress: res => {
        const progress = res.bytesWritten / res.contentLength;
        onProgress?.(progress);
      },
    });

    await download.promise;

    // Verify checksum
    const checksum = await RNFS.hash(modelPath, 'sha256');
    if (checksum !== manifest.checksum) {
      await RNFS.unlink(modelPath);
      throw new Error('Model checksum verification failed');
    }

    // Load new model
    await this.loadModel(modelPath);

    // Save metadata to database
    await SQLiteDatabase.exec(
      `INSERT OR REPLACE INTO models (id, version, model_data, feature_count, created_at, last_used)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [
        manifest.version,
        manifest.version,
        modelPath,
        manifest.featureCount,
        Date.now(),
        Date.now(),
      ]
    );

    this.currentVersion = manifest.version;

    // Clean up old models
    await this.cleanupOldModels();

    console.log(`Model ${manifest.version} installed successfully`);
  }

  private async loadModel(modelPath: string): Promise<void> {
    await onnxInference.loadModel(modelPath);
  }

  private async getLocalModel(): Promise<{ version: string; path: string } | null> {
    const result = await SQLiteDatabase.query(
      'SELECT version, model_data FROM models ORDER BY created_at DESC LIMIT 1'
    );

    if (result.length > 0) {
      return {
        version: result[0].version,
        path: result[0].model_data,
      };
    }

    return null;
  }

  private async cleanupOldModels(): Promise<void> {
    // Keep only the latest 2 model versions
    const models = await SQLiteDatabase.query(
      'SELECT id, model_data FROM models ORDER BY created_at DESC'
    );

    for (let i = 2; i < models.length; i++) {
      const oldModel = models[i];

      // Delete file
      if (await RNFS.exists(oldModel.model_data)) {
        await RNFS.unlink(oldModel.model_data);
      }

      // Delete from database
      await SQLiteDatabase.exec('DELETE FROM models WHERE id = ?', [oldModel.id]);
    }
  }
}

export const modelSync = new ModelSyncService();
```

### Phase 2: Offline-First Architecture (Week 3-4)

#### 2.1 Redux State Management

**Store:** `src/store/index.ts`

```typescript
import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';

import modelReducer from './slices/modelSlice';
import deviceReducer from './slices/deviceSlice';
import syncReducer from './slices/syncSlice';

const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
  whitelist: ['model', 'devices', 'sync'], // Persist these reducers
};

const rootReducer = {
  model: modelReducer,
  devices: deviceReducer,
  sync: syncReducer,
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
  middleware: getDefaultMiddleware =>
    getDefaultMiddleware({
      serializableCheck: false, // Required for redux-persist
    }),
});

export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

#### 2.2 Offline Queue System

**Service:** `src/services/OfflineQueue.ts`

```typescript
export interface QueuedOperation {
  id: string;
  type: 'score' | 'resolve' | 'note';
  deviceId: string;
  payload: any;
  timestamp: number;
  retryCount: number;
  status: 'pending' | 'processing' | 'failed';
}

export class OfflineQueueService {
  private queue: QueuedOperation[] = [];
  private processing = false;

  async addToQueue(operation: Omit<QueuedOperation, 'id' | 'retryCount' | 'status'>): Promise<void> {
    const queueItem: QueuedOperation = {
      ...operation,
      id: uuid(),
      retryCount: 0,
      status: 'pending',
    };

    // Save to database
    await SQLiteDatabase.exec(
      `INSERT INTO sync_queue (id, operation, payload, created_at, retry_count)
       VALUES (?, ?, ?, ?, ?)`,
      [queueItem.id, queueItem.type, JSON.stringify(queueItem.payload), queueItem.timestamp, 0]
    );

    this.queue.push(queueItem);

    // Try to sync immediately if online
    if (await this.isOnline()) {
      this.processQueue();
    }
  }

  async processQueue(): Promise<void> {
    if (this.processing) return;

    this.processing = true;

    try {
      while (this.queue.length > 0 && (await this.isOnline())) {
        const item = this.queue[0];

        try {
          await this.processItem(item);

          // Success - remove from queue
          this.queue.shift();
          await SQLiteDatabase.exec('DELETE FROM sync_queue WHERE id = ?', [item.id]);
        } catch (error) {
          item.retryCount++;

          if (item.retryCount >= 3) {
            // Max retries reached - mark as failed
            item.status = 'failed';
            await SQLiteDatabase.exec(
              'UPDATE sync_queue SET retry_count = ?, status = ? WHERE id = ?',
              [item.retryCount, 'failed', item.id]
            );
            this.queue.shift();
          } else {
            // Retry later
            await SQLiteDatabase.exec(
              'UPDATE sync_queue SET retry_count = ? WHERE id = ?',
              [item.retryCount, item.id]
            );
            break;
          }
        }
      }
    } finally {
      this.processing = false;
    }
  }

  private async processItem(item: QueuedOperation): Promise<void> {
    switch (item.type) {
      case 'score':
        await apiClient.post('/api/anomalies/score', item.payload);
        break;
      case 'resolve':
        await apiClient.patch(`/api/anomalies/${item.payload.id}/resolve`);
        break;
      case 'note':
        await apiClient.post(`/api/anomalies/${item.payload.anomalyId}/notes`, item.payload);
        break;
    }
  }

  private async isOnline(): Promise<boolean> {
    const state = await NetInfo.fetch();
    return state.isConnected && state.isInternetReachable;
  }
}

export const offlineQueue = new OfflineQueueService();
```

### Phase 3: UI Implementation (Week 5-6)

#### 3.1 Scanner Screen

```typescript
// src/screens/ScannerScreen.tsx

import React, { useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { RNCamera } from 'react-native-camera';
import { useNavigation } from '@react-navigation/native';
import { onnxInference } from '../services/OnnxInference';

export const ScannerScreen: React.FC = () => {
  const navigation = useNavigation();
  const [scanning, setScanning] = useState(true);

  const handleBarCodeRead = async ({ data }: { data: string }) => {
    if (!scanning) return;

    setScanning(false);

    try {
      // Parse device ID from QR code
      const deviceId = parseDeviceId(data);

      // Fetch device data (cached or from API)
      const deviceData = await fetchDeviceData(deviceId);

      // Extract features
      const features = extractFeatures(deviceData);

      // Run ONNX inference
      const result = await onnxInference.predict(features);

      // Navigate to results
      navigation.navigate('DeviceDetail', {
        deviceId,
        anomalyScore: result.score,
        anomalyLabel: result.label,
      });
    } catch (error) {
      console.error('Scan error:', error);
      Alert.alert('Error', 'Failed to process device');
    } finally {
      setScanning(true);
    }
  };

  return (
    <View style={styles.container}>
      <RNCamera
        style={styles.camera}
        onBarCodeRead={handleBarCodeRead}
        barCodeTypes={[RNCamera.Constants.BarCodeType.qr]}
      />
      <View style={styles.overlay}>
        <Text style={styles.instructions}>Scan device QR code</Text>
      </View>
    </View>
  );
};
```

### Phase 4: Performance Optimization (Week 7-8)

#### 4.1 Model Warm-up

```typescript
// Warm up model on app launch for faster first inference
export async function warmUpModel(): Promise<void> {
  const dummyFeatures = new Array(25).fill(0);
  await onnxInference.predict(dummyFeatures);
  console.log('Model warmed up');
}
```

#### 4.2 Background Sync

```typescript
// Background sync using react-native-background-fetch
import BackgroundFetch from 'react-native-background-fetch';

BackgroundFetch.configure(
  {
    minimumFetchInterval: 15, // minutes
    stopOnTerminate: false,
    startOnBoot: true,
  },
  async (taskId) => {
    console.log('[BackgroundFetch] Event received:', taskId);

    // Check for model updates
    if (await modelSync.checkForUpdates()) {
      // Download in background (WiFi only)
      if (await isWifiConnected()) {
        const manifest = await apiClient.get('/api/models/latest');
        await modelSync.downloadAndInstall(manifest);
      }
    }

    // Process offline queue
    await offlineQueue.processQueue();

    BackgroundFetch.finish(taskId);
  },
  (taskId) => {
    console.log('[BackgroundFetch] TIMEOUT:', taskId);
    BackgroundFetch.finish(taskId);
  }
);
```

## Deployment

### iOS Deployment

**Requirements:**
- Xcode 14+
- iOS 13+ target
- CoreML support for acceleration

**Build:**
```bash
cd ios
pod install
cd ..
npx react-native run-ios --configuration Release
```

**App Store Submission:**
- App size: ~15MB (before models)
- Model download: 3MB (INT8 quantized)
- Total footprint: ~18MB

### Android Deployment

**Requirements:**
- Android SDK 24+ (Android 7.0+)
- NNAPI support for acceleration

**Build:**
```bash
cd android
./gradlew assembleRelease
```

**Google Play Submission:**
- APK size: ~18MB (before models)
- Model download: 3MB
- Total footprint: ~21MB

## Testing Strategy

### Unit Tests
- ONNX inference accuracy
- Model sync logic
- Offline queue operations
- Feature extraction

### Integration Tests
- End-to-end scoring flow
- Sync process
- Offline → Online transition

### Performance Tests
- Inference latency (target: <100ms)
- Model load time (target: <1s)
- Battery impact (target: <5% per hour)

## Success Metrics

- **Inference Speed:** <100ms per device
- **Offline Capability:** 100% functional offline
- **Sync Success Rate:** >99%
- **Model Update Adoption:** >90% within 24hrs
- **Crash Rate:** <0.5%
- **App Store Rating:** >4.5 stars

## Future Enhancements

- **Federated Learning:** Train models on device data
- **Push Notifications:** Alert on critical anomalies
- **AR Mode:** Point camera at device, overlay anomaly info
- **Voice Interface:** "Check device 12345"
- **Smartwatch Support:** Quick device check on watch

## Conclusion

This mobile architecture enables truly offline anomaly detection with enterprise-grade performance and reliability. ONNX Runtime makes it possible to run production-quality ML models on resource-constrained mobile devices.
