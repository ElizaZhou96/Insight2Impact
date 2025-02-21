import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Volume2, VolumeX, Eye, EyeOff, Palette, Scan } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

function App() {
  const [isVisionImpaired, setIsVisionImpaired] = useState(false);
  const [isHearingImpaired, setIsHearingImpaired] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [colorBlindMode, setColorBlindMode] = useState(false);
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<Array<{
    class: string;
    score: number;
  }>>([]);
  const previousObjectsRef = useRef<Set<string>>(new Set());
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number>();
  const isDetectionRunningRef = useRef(false);

  // Speech synthesis utility
  const speak = useCallback((text: string) => {
    // Only speak if hearing is not impaired and detection is active
    if (!isHearingImpaired && isDetectionActive && isDetectionRunningRef.current) {
      const utterance = new SpeechSynthesisUtterance(text);
      speechSynthesis.speak(utterance);
    }
  }, [isHearingImpaired, isDetectionActive]);

  // Effect to handle hearing impairment changes
  useEffect(() => {
    if (isHearingImpaired) {
      // Cancel any ongoing speech when hearing impairment is enabled
      speechSynthesis.cancel();
    }
  }, [isHearingImpaired]);

  // Initialize TensorFlow.js and load COCO-SSD model
  useEffect(() => {
    const initTF = async () => {
      try {
        await tf.ready();
        await tf.setBackend('webgl');
        const loadedModel = await cocoSsd.load({
          base: 'mobilenet_v2'
        });
        setModel(loadedModel);
      } catch (error) {
        console.error('Error initializing TensorFlow.js or loading COCO-SSD model:', error);
      }
    };
    
    initTF();
  }, []);

  // Cleanup animation frame and speech synthesis on unmount
  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, []);

  // Object detection colors based on class - using more distinct colors
  const colorMap: { [key: string]: string } = {
    person: '#FF3B30',           // Bright Red
    car: '#5856D6',             // Purple
    bicycle: '#34C759',         // Green
    chair: '#FF9500',           // Orange
    couch: '#007AFF',           // Blue
    tv: '#AF52DE',             // Pink
    laptop: '#FFD60A',         // Yellow
    'cell phone': '#FF2D55',   // Rose
    book: '#64D2FF',           // Light Blue
    clock: '#5AC8FA',          // Sky Blue
    'traffic light': '#30D158', // Lime Green
    'stop sign': '#BF5AF2',    // Violet
    default: '#8E8E93'         // Gray
  };

  const commonObjects = [
    'person',
    'car',
    'bicycle',
    'chair',
    'couch',
    'tv',
    'laptop',
    'cell phone',
    'book',
    'clock',
    'traffic light',
    'stop sign'
  ];

  const getColorForClass = (className: string) => {
    return colorMap[className] || colorMap.default;
  };

  // Function to stop detection and cleanup
  const stopDetection = useCallback(() => {
    isDetectionRunningRef.current = false;
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = undefined;
    }
    speechSynthesis.cancel();
    setDetectedObjects([]);
    previousObjectsRef.current.clear();
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, []);

  // Detect objects in the video stream
  const detectObjects = useCallback(async () => {
    if (!model || !webcamRef.current || !canvasRef.current || !isDetectionRunningRef.current) {
      return;
    }

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!video || !ctx) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    try {
      // Detect objects
      const predictions = await model.detect(video, undefined, 0.6);

      if (!isDetectionRunningRef.current) return;

      // Get current objects for speech synthesis
      const currentObjects = new Set(predictions.map(pred => pred.class));
      
      // Announce new objects that weren't in the previous frame
      currentObjects.forEach(obj => {
        if (!previousObjectsRef.current.has(obj)) {
          speak(`Detected ${obj}`);
        }
      });

      // Update previous objects for next frame
      previousObjectsRef.current = currentObjects;

      // Update detected objects state
      setDetectedObjects(predictions.map(pred => ({
        class: pred.class,
        score: pred.score
      })));

      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw bounding boxes and labels
      predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const color = getColorForClass(prediction.class);

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw label background
        ctx.fillStyle = color;
        const padding = 8;
        const textWidth = ctx.measureText(prediction.class).width;
        ctx.fillRect(x - padding/2, y - 30, textWidth + padding * 2, 30);

        // Draw label text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 16px Inter, system-ui, sans-serif';
        ctx.fillText(
          `${prediction.class} ${(prediction.score * 100).toFixed(1)}%`,
          x + padding/2,
          y - 10
        );
      });

      // Request next frame only if detection is still active
      if (isDetectionRunningRef.current) {
        animationFrameRef.current = requestAnimationFrame(detectObjects);
      }
    } catch (error) {
      console.error('Error during object detection:', error);
      setIsDetectionActive(false);
      stopDetection();
    }
  }, [model, speak, stopDetection]);

  // Start/stop detection loop when active state changes
  useEffect(() => {
    if (isDetectionActive) {
      isDetectionRunningRef.current = true;
      detectObjects();
    } else {
      stopDetection();
    }
  }, [isDetectionActive, detectObjects, stopDetection]);

  const handleUserMedia = useCallback(() => {
    setIsCameraReady(true);
  }, []);

  // Combine visual impairment effects
  const getVisualEffects = () => {
    if (!isVisionImpaired) return '';
    const effects = ['blur-sm', 'brightness-50'];
    return effects.join(' ');
  };

  return (
    <>
      <svg xmlns="http://www.w3.org/2000/svg" className="hidden">
        <filter id="red-green-blind">
          <feColorMatrix
            type="matrix"
            values="
              0.625, 0.375, 0,   0, 0,
              0.7,   0.3,   0,   0, 0,
              0,     0.3,   0.7, 0, 0,
              0,     0,     0,   1, 0"
          />
        </filter>
      </svg>

      <div className={`min-h-screen bg-gray-900 text-white ${colorBlindMode && isVisionImpaired ? '[filter:url(#red-green-blind)]' : ''}`}>
        {/* Header */}
        <header className="bg-gray-800 p-4 sticky top-0 z-50 shadow-lg">
          <div className="container mx-auto">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center space-x-2">
                <Camera className="w-6 h-6" />
                <h1 className="text-xl font-bold">Accessibility Simulator</h1>
              </div>
              <div className="flex items-center space-x-4 flex-wrap gap-2">
                <button
                  onClick={() => setIsDetectionActive(!isDetectionActive)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition ${
                    isDetectionActive
                      ? 'bg-blue-600 hover:bg-blue-700'
                      : 'bg-gray-600 hover:bg-gray-700'
                  }`}
                  disabled={!model}
                >
                  <Scan className="w-4 h-4" />
                  <span>{isDetectionActive ? 'Stop Detection' : 'Start Detection'}</span>
                </button>
                <button
                  onClick={() => setIsVisionImpaired(!isVisionImpaired)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition ${
                    isVisionImpaired
                      ? 'bg-yellow-600 hover:bg-yellow-700'
                      : 'bg-gray-600 hover:bg-gray-700'
                  }`}
                >
                  {isVisionImpaired ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  <span>Vision Impairment</span>
                </button>
                {isVisionImpaired && (
                  <button
                    onClick={() => setColorBlindMode(!colorBlindMode)}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition ${
                      colorBlindMode
                        ? 'bg-purple-600 hover:bg-purple-700'
                        : 'bg-gray-600 hover:bg-gray-700'
                    }`}
                  >
                    <Palette className="w-4 h-4" />
                    <span>Color Blind Mode</span>
                  </button>
                )}
                <button
                  onClick={() => setIsHearingImpaired(!isHearingImpaired)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition ${
                    isHearingImpaired
                      ? 'bg-red-600 hover:bg-red-700'
                      : 'bg-gray-600 hover:bg-gray-700'
                  }`}
                >
                  {isHearingImpaired ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                  <span>Hearing Impairment</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="container mx-auto p-4">
          <div className="max-w-6xl mx-auto">
            {/* Info Cards */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-800 p-6 rounded-xl">
                <div className="flex items-center space-x-2 mb-4">
                  <Eye className="w-6 h-6 text-blue-400" />
                  <h2 className="text-xl font-semibold">Visual Impairment</h2>
                </div>
                <p className="text-gray-300 mb-4">
                  The simulation applies blur and reduced brightness to demonstrate common
                  visual impairments. This helps understand the challenges faced by
                  visually impaired individuals.
                </p>
                <div className="bg-gray-700/50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Palette className="w-5 h-5 text-purple-400" />
                    <h3 className="font-medium">Color Blindness Simulation</h3>
                  </div>
                  <p className="text-sm text-gray-400">
                    Toggle the Color Blind Mode to experience red-green color blindness
                    (deuteranopia). This simulation uses a precise color matrix to accurately
                    represent how colors appear to individuals with this condition.
                  </p>
                </div>
              </div>

              <div className="bg-gray-800 p-6 rounded-xl">
                <div className="flex items-center space-x-2 mb-4">
                  <Volume2 className="w-6 h-6 text-blue-400" />
                  <h2 className="text-xl font-semibold">Hearing Impairment</h2>
                </div>
                <p className="text-gray-300">
                  When active, the simulation mutes all audio output, providing insight
                  into the experience of those with hearing impairments in their daily
                  interactions.
                </p>
              </div>
            </div>

            <div className="grid lg:grid-cols-4 gap-6">
              {/* Detectable Objects List */}
              <div className="lg:col-span-1 bg-gray-800 p-6 rounded-xl h-[calc(100vh-24rem)] sticky top-28 overflow-y-auto">
                <div className="flex items-center space-x-2 mb-4">
                  <Scan className="w-6 h-6 text-blue-400" />
                  <h2 className="text-xl font-semibold">Detectable Objects</h2>
                </div>
                <div className="space-y-2">
                  {commonObjects.map((obj) => (
                    <div
                      key={obj}
                      className="flex items-center space-x-2 p-2 rounded bg-gray-700/50"
                    >
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: getColorForClass(obj) }}
                      />
                      <span className="capitalize">{obj}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Camera and Detection Section */}
              <div className="lg:col-span-3 space-y-6">
                {/* Camera View */}
                <div className="relative rounded-xl overflow-hidden bg-black h-[calc(100vh-24rem)]">
                  <Webcam
                    ref={webcamRef}
                    onUserMedia={handleUserMedia}
                    className={`w-full h-full object-contain ${getVisualEffects()}`}
                    mirrored
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  />
                  
                  {!isCameraReady && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                      <p className="text-lg">Please allow camera access...</p>
                    </div>
                  )}

                  {!model && isCameraReady && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
                      <p className="text-lg">Loading object detection model...</p>
                    </div>
                  )}

                  {/* Active Impairments Overlay */}
                  <div className="absolute top-4 right-4">
                    <div className="flex flex-col gap-2">
                      {isVisionImpaired && (
                        <div className="flex items-center space-x-2 bg-gray-800/80 px-4 py-2 rounded-lg">
                          <EyeOff className="w-5 h-5 text-yellow-400" />
                          <span>Vision Impaired</span>
                        </div>
                      )}
                      {isHearingImpaired && (
                        <div className="flex items-center space-x-2 bg-gray-800/80 px-4 py-2 rounded-lg">
                          <VolumeX className="w-5 h-5 text-red-400" />
                          <span>Hearing Impaired</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Detection Results */}
                {isDetectionActive && detectedObjects.length > 0 && (
                  <div className="bg-gray-800 p-6 rounded-xl">
                    <div className="flex items-center space-x-2 mb-4">
                      <Scan className="w-6 h-6 text-blue-400" />
                      <h2 className="text-xl font-semibold">Detection Results</h2>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {detectedObjects.map((obj, index) => (
                        <div
                          key={index}
                          className="bg-gray-700/50 p-4 rounded-lg"
                          style={{ borderLeft: `4px solid ${getColorForClass(obj.class)}` }}
                        >
                          <h3 className="font-medium capitalize">{obj.class}</h3>
                          <p className="text-sm text-gray-400">
                            Confidence: {(obj.score * 100).toFixed(1)}%
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}

export default App;