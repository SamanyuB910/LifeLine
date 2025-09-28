"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Camera, Download, RefreshCw, Clock, Eye, AlertTriangle, Activity } from "lucide-react"

export function LatestSnapshot() {
  const [lastCaptured, setLastCaptured] = useState(new Date())
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Simulated snapshot data
  const [snapshotData, setSnapshotData] = useState({
    fallRisk: 23,
    emotionalState: "Calm",
    posture: "Sitting upright",
    movement: "Minimal",
    attention: "Alert",
  })

  // Auto-refresh snapshot every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setLastCaptured(new Date())
      setSnapshotData((prev) => ({
        ...prev,
        fallRisk: Math.floor(Math.random() * 40) + 10,
      }))
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const handleRefresh = () => {
    setIsRefreshing(true)
    setTimeout(() => {
      setLastCaptured(new Date())
      setSnapshotData((prev) => ({
        ...prev,
        fallRisk: Math.floor(Math.random() * 40) + 10,
      }))
      setIsRefreshing(false)
    }, 1000)
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
            <span>Latest Patient Snapshot</span>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant="secondary">Room ICU-204</Badge>
            <Badge variant="outline" className="flex items-center space-x-1">
              <Clock className="w-3 h-3" />
              <span>{lastCaptured.toLocaleTimeString()}</span>
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Snapshot Image Container */}
        <div className="relative bg-muted/20 rounded-lg overflow-hidden aspect-video">
          <div className="relative w-full h-full bg-gradient-to-br from-muted/40 to-muted/60 flex items-center justify-center">
            {/* Simulated snapshot placeholder */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-purple-900/20">
              <div className="absolute top-4 left-4 flex items-center space-x-2">
                <div className="w-3 h-3 bg-primary rounded-full"></div>
                <span className="text-sm font-medium text-white">SNAPSHOT</span>
              </div>

              {/* Simulated patient silhouette */}
              <div className="absolute bottom-1/4 left-1/2 transform -translate-x-1/2">
                <div className="w-32 h-48 bg-gradient-to-t from-blue-400/30 to-transparent rounded-t-full"></div>
              </div>

              {/* AI Analysis Overlay */}
              <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm rounded-lg p-3 space-y-2">
                <div className="flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-primary" />
                  <span className="text-xs text-white font-medium">AI Analysis</span>
                </div>
                <div className="space-y-1 text-xs text-white">
                  <div className="flex justify-between">
                    <span>Fall Risk:</span>
                    <span className={getRiskColor(snapshotData.fallRisk)}>{snapshotData.fallRisk}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Emotion:</span>
                    <span className="text-accent">{snapshotData.emotionalState}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Posture:</span>
                    <span className="text-chart-2">{snapshotData.posture}</span>
                  </div>
                </div>
              </div>

              {/* Timestamp overlay */}
              <div className="absolute bottom-4 left-4">
                <div className="bg-black/60 backdrop-blur-sm rounded-full px-3 py-1">
                  <span className="text-xs text-white">Captured: {lastCaptured.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Snapshot Analysis Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <AlertTriangle className={`w-4 h-4 ${getRiskColor(snapshotData.fallRisk)}`} />
                <span className="text-sm font-medium">Fall Risk</span>
              </div>
              <Badge variant={getRiskBadgeVariant(snapshotData.fallRisk)}>{snapshotData.fallRisk}%</Badge>
            </div>
            <p className="text-xs text-muted-foreground">At time of capture</p>
          </div>

          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-chart-2" />
                <span className="text-sm font-medium">Emotional State</span>
              </div>
              <Badge variant="secondary">{snapshotData.emotionalState}</Badge>
            </div>
            <p className="text-xs text-muted-foreground">Facial expression analysis</p>
          </div>

          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Eye className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">Attention Level</span>
              </div>
              <Badge variant="secondary">{snapshotData.attention}</Badge>
            </div>
            <p className="text-xs text-muted-foreground">Awareness assessment</p>
          </div>
        </div>

        {/* Snapshot Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isRefreshing}>
              <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
              Refresh Snapshot
            </Button>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          </div>
          <div className="text-xs text-muted-foreground">Auto-refresh: 30s intervals</div>
        </div>
      </CardContent>
    </Card>
  )
}
