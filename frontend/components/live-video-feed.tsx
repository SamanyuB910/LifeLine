"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Video,
  VideoOff,
  Volume2,
  VolumeX,
  Maximize2,
  RotateCcw,
  Camera,
  AlertTriangle,
  Eye,
  Activity,
} from "lucide-react"

export function LiveVideoFeed() {
  const [isVideoActive, setIsVideoActive] = useState(true)
  const [isAudioActive, setIsAudioActive] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [snapshotUrl, setSnapshotUrl] = useState<string>("")
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Real AI analysis data from backend
  const [aiAnalysis, setAiAnalysis] = useState({
    fallRisk: 0,
    emotionalState: "No Face",
    posture: "Not detected",
    movement: "Unknown",
    attention: "Unknown",
    lastUpdate: new Date().toLocaleTimeString(),
    confidence: 0,
    cameraStatus: "inactive",
  })

  // Fetch real data from backend API
  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/patients/patient_001')
        if (response.ok) {
          const data = await response.json()
          setAiAnalysis({
            fallRisk: data.vitals.fall_risk || 0,
            emotionalState: data.emotion.current === 'no_face' ? 'No Face' : 
                           data.emotion.current.charAt(0).toUpperCase() + data.emotion.current.slice(1),
            posture: data.vitals.posture || "Not detected",
            movement: data.vitals.movement_activity || "Unknown",
            attention: data.vitals.attention_level ? `${data.vitals.attention_level}%` : "Unknown",
            lastUpdate: new Date(data.last_update).toLocaleTimeString(),
            confidence: data.emotion.confidence || 0,
            cameraStatus: data.camera_status || "inactive",
          })
        }
        
        // Update snapshot URL with timestamp to force refresh
        setSnapshotUrl(`http://localhost:5000/api/latest-snapshot?t=${Date.now()}`)
      } catch (error) {
        console.error('Error fetching patient data:', error)
      }
    }

    // Initial fetch
    fetchPatientData()

    // Update every 2 seconds
    const interval = setInterval(fetchPatientData, 2000)

    return () => clearInterval(interval)
  }, [])

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  const handleVideoToggle = () => {
    setIsVideoActive(!isVideoActive)
  }

  const handleAudioToggle = () => {
    setIsAudioActive(!isAudioActive)
  }

  const getRiskColor = (risk: number) => {
    if (risk < 30) return "text-accent"
    if (risk < 60) return "text-chart-3"
    return "text-destructive"
  }

  const getRiskBadgeVariant = (risk: number) => {
    if (risk < 30) return "secondary"
    if (risk < 60) return "outline"
    return "destructive"
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Camera className="w-5 h-5" />
            <span>Live Patient Monitor</span>
            <div className="w-2 h-2 bg-accent rounded-full pulse-dot"></div>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant="secondary">Room ICU-204</Badge>
            <Badge variant="outline">HD 1080p</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Video Feed Container */}
        <div ref={containerRef} className="relative bg-muted/20 rounded-lg overflow-hidden aspect-video">
          {isVideoActive ? (
            <div className="relative w-full h-full">
              {/* Actual camera snapshot */}
              {snapshotUrl ? (
                <img
                  src={snapshotUrl}
                  alt="Live camera feed"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    // Fallback to placeholder if image fails to load
                    e.currentTarget.style.display = 'none'
                  }}
                />
              ) : (
                <div className="w-full h-full bg-gradient-to-br from-muted/40 to-muted/60 flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <Camera className="w-12 h-12 mx-auto mb-2" />
                    <p>Loading camera feed...</p>
                  </div>
                </div>
              )}

              {/* Overlay for camera status */}
              <div className="absolute inset-0">
                <div className="absolute top-4 left-4 flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${aiAnalysis.cameraStatus === 'active' ? 'bg-green-500 pulse-dot' : 'bg-red-500'}`}></div>
                  <span className="text-sm font-medium text-white drop-shadow-lg">
                    {aiAnalysis.cameraStatus === 'active' ? 'LIVE' : 'CAMERA INACTIVE'}
                  </span>
                </div>

                {/* High Risk Alert for No Face Detected */}
                {aiAnalysis.emotionalState === 'No Face' && (
                  <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
                    <div className="bg-red-600/90 backdrop-blur-sm rounded-lg px-4 py-2 border-2 border-red-400 pulse-dot">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className="w-5 h-5 text-white" />
                        <span className="text-sm font-bold text-white">HIGH RISK - PATIENT NOT VISIBLE</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Real AI Analysis Overlay */}
                <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm rounded-lg p-3 space-y-2">
                  <div className="flex items-center space-x-2">
                    <Eye className="w-4 h-4 text-primary" />
                    <span className="text-xs text-white font-medium">AI Analysis</span>
                  </div>
                  <div className="space-y-1 text-xs text-white">
                    <div className="flex justify-between">
                      <span>Fall Risk:</span>
                      <span className={getRiskColor(aiAnalysis.fallRisk)}>{aiAnalysis.fallRisk}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Emotion:</span>
                      <span className="text-accent">{aiAnalysis.emotionalState}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Confidence:</span>
                      <span className="text-chart-2">{(aiAnalysis.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Posture:</span>
                      <span className="text-chart-2">{aiAnalysis.posture}</span>
                    </div>
                  </div>
                </div>

                {/* Real Movement Detection Indicators */}
                <div className="absolute bottom-4 left-4 flex space-x-2">
                  <div className="bg-black/60 backdrop-blur-sm rounded-full px-3 py-1">
                    <span className="text-xs text-white drop-shadow-lg">Movement: {aiAnalysis.movement}</span>
                  </div>
                  <div className="bg-black/60 backdrop-blur-sm rounded-full px-3 py-1">
                    <span className="text-xs text-white drop-shadow-lg">Attention: {aiAnalysis.attention}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-muted/40">
              <div className="text-center">
                <VideoOff className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
                <p className="text-muted-foreground">Video feed disabled</p>
              </div>
            </div>
          )}

          {/* Video Controls Overlay */}
          <div className="absolute bottom-4 right-4 flex items-center space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleVideoToggle}
              className="bg-black/60 backdrop-blur-sm hover:bg-black/80"
            >
              {isVideoActive ? <Video className="w-4 h-4" /> : <VideoOff className="w-4 h-4" />}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleAudioToggle}
              className="bg-black/60 backdrop-blur-sm hover:bg-black/80"
            >
              {isAudioActive ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={toggleFullscreen}
              className="bg-black/60 backdrop-blur-sm hover:bg-black/80"
            >
              <Maximize2 className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* AI Analysis Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <AlertTriangle className={`w-4 h-4 ${getRiskColor(aiAnalysis.fallRisk)}`} />
                <span className="text-sm font-medium">Fall Risk</span>
              </div>
              <Badge variant={getRiskBadgeVariant(aiAnalysis.fallRisk)}>{aiAnalysis.fallRisk}%</Badge>
            </div>
            <p className="text-xs text-muted-foreground">Based on posture and movement analysis</p>
          </div>

          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-chart-2" />
                <span className="text-sm font-medium">Emotional State</span>
              </div>
              <Badge variant="secondary">{aiAnalysis.emotionalState}</Badge>
            </div>
            <p className="text-xs text-muted-foreground">Facial expression recognition</p>
          </div>

          <div className={`rounded-lg p-4 ${aiAnalysis.posture === 'not detected' ? 'bg-red-50 border border-red-200' : 'bg-muted/50'}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Eye className={`w-4 h-4 ${aiAnalysis.posture === 'not detected' ? 'text-red-600' : 'text-primary'}`} />
                <span className="text-sm font-medium">Posture</span>
              </div>
              <Badge variant={aiAnalysis.posture === 'not detected' ? 'destructive' : 'secondary'}>
                {aiAnalysis.posture === 'not detected' ? 'Patient Not Visible' : 'Sitting Upright'}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              {aiAnalysis.posture === 'not detected' ? 'High risk - immediate attention required' : 'Normal posture detected'}
            </p>
          </div>
        </div>

        {/* Video Feed Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset View
            </Button>
            <Button variant="outline" size="sm">
              Save Snapshot
            </Button>
          </div>
          <div className="text-xs text-muted-foreground">Last updated: {aiAnalysis.lastUpdate}</div>
        </div>
      </CardContent>
    </Card>
  )
}
