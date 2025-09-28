"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from "recharts"
import { Heart, Activity, Thermometer, Droplets, TrendingUp, TrendingDown, Minus, RefreshCw } from "lucide-react"

interface VitalSignData {
  time: string
  heartRate: number
  bloodPressureSys: number
  bloodPressureDia: number
  temperature: number
  oxygenSat: number
  respiratoryRate: number
}

export function VitalSignsCharts() {
  const [vitalData, setVitalData] = useState<VitalSignData[]>([])
  const [isLive, setIsLive] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState("1h")

  // Generate initial data
  useEffect(() => {
    const generateInitialData = () => {
      const data: VitalSignData[] = []
      const now = new Date()

      for (let i = 59; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60000) // Every minute
        data.push({
          time: time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          heartRate: 75 + Math.random() * 10 - 5,
          bloodPressureSys: 120 + Math.random() * 20 - 10,
          bloodPressureDia: 80 + Math.random() * 10 - 5,
          temperature: 98.6 + Math.random() * 2 - 1,
          oxygenSat: 97 + Math.random() * 3,
          respiratoryRate: 16 + Math.random() * 4 - 2,
        })
      }
      return data
    }

    setVitalData(generateInitialData())
  }, [])

  // Simulate real-time updates
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      setVitalData((prevData) => {
        const newData = [...prevData]
        const now = new Date()
        const newPoint: VitalSignData = {
          time: now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          heartRate: 75 + Math.random() * 10 - 5,
          bloodPressureSys: 120 + Math.random() * 20 - 10,
          bloodPressureDia: 80 + Math.random() * 10 - 5,
          temperature: 98.6 + Math.random() * 2 - 1,
          oxygenSat: 97 + Math.random() * 3,
          respiratoryRate: 16 + Math.random() * 4 - 2,
        }

        // Keep only last 60 points
        newData.push(newPoint)
        if (newData.length > 60) {
          newData.shift()
        }

        return newData
      })
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [isLive])

  const currentVitals = vitalData[vitalData.length - 1] || {
    heartRate: 78,
    bloodPressureSys: 120,
    bloodPressureDia: 80,
    temperature: 98.6,
    oxygenSat: 97,
    respiratoryRate: 16,
  }

  const getTrend = (data: VitalSignData[], key: keyof VitalSignData) => {
    if (data.length < 2) return "stable"
    const recent = data.slice(-5)
    const avg1 = recent.slice(0, 2).reduce((sum, item) => sum + (item[key] as number), 0) / 2
    const avg2 = recent.slice(-2).reduce((sum, item) => sum + (item[key] as number), 0) / 2
    const diff = avg2 - avg1
    if (Math.abs(diff) < 1) return "stable"
    return diff > 0 ? "up" : "down"
  }

  const TrendIcon = ({ trend }: { trend: string }) => {
    switch (trend) {
      case "up":
        return <TrendingUp className="w-4 h-4 text-chart-3" />
      case "down":
        return <TrendingDown className="w-4 h-4 text-destructive" />
      default:
        return <Minus className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getVitalStatus = (vital: string, value: number) => {
    switch (vital) {
      case "heartRate":
        if (value < 60 || value > 100) return "warning"
        return "normal"
      case "temperature":
        if (value < 97 || value > 100.4) return "warning"
        return "normal"
      case "oxygenSat":
        if (value < 95) return "critical"
        if (value < 97) return "warning"
        return "normal"
      default:
        return "normal"
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "critical":
        return "text-destructive"
      case "warning":
        return "text-chart-3"
      default:
        return "text-accent"
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Activity className="w-5 h-5" />
            <span>Vital Signs Monitoring</span>
            {isLive && <div className="w-2 h-2 bg-accent rounded-full pulse-dot"></div>}
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button
              variant={isLive ? "default" : "outline"}
              size="sm"
              onClick={() => setIsLive(!isLive)}
              className="flex items-center space-x-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLive ? "animate-spin" : ""}`} />
              <span>{isLive ? "Live" : "Paused"}</span>
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Vitals Summary */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Heart className="w-5 h-5 text-chart-1" />
              <TrendIcon trend={getTrend(vitalData, "heartRate")} />
            </div>
            <div className="space-y-1">
              <p className="text-2xl font-bold text-foreground">{Math.round(currentVitals.heartRate)}</p>
              <p className="text-sm text-muted-foreground">Heart Rate (BPM)</p>
              <Badge variant="outline" className={getStatusColor(getVitalStatus("heartRate", currentVitals.heartRate))}>
                {getVitalStatus("heartRate", currentVitals.heartRate)}
              </Badge>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Activity className="w-5 h-5 text-primary" />
              <TrendIcon trend={getTrend(vitalData, "bloodPressureSys")} />
            </div>
            <div className="space-y-1">
              <p className="text-2xl font-bold text-foreground">
                {Math.round(currentVitals.bloodPressureSys)}/{Math.round(currentVitals.bloodPressureDia)}
              </p>
              <p className="text-sm text-muted-foreground">Blood Pressure</p>
              <Badge variant="outline" className="text-accent">
                normal
              </Badge>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Thermometer className="w-5 h-5 text-chart-3" />
              <TrendIcon trend={getTrend(vitalData, "temperature")} />
            </div>
            <div className="space-y-1">
              <p className="text-2xl font-bold text-foreground">{currentVitals.temperature.toFixed(1)}°F</p>
              <p className="text-sm text-muted-foreground">Temperature</p>
              <Badge
                variant="outline"
                className={getStatusColor(getVitalStatus("temperature", currentVitals.temperature))}
              >
                {getVitalStatus("temperature", currentVitals.temperature)}
              </Badge>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Droplets className="w-5 h-5 text-chart-2" />
              <TrendIcon trend={getTrend(vitalData, "oxygenSat")} />
            </div>
            <div className="space-y-1">
              <p className="text-2xl font-bold text-foreground">{Math.round(currentVitals.oxygenSat)}%</p>
              <p className="text-sm text-muted-foreground">Oxygen Saturation</p>
              <Badge variant="outline" className={getStatusColor(getVitalStatus("oxygenSat", currentVitals.oxygenSat))}>
                {getVitalStatus("oxygenSat", currentVitals.oxygenSat)}
              </Badge>
            </div>
          </div>
        </div>

        {/* Charts Tabs */}
        <Tabs defaultValue="heartRate" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="heartRate">Heart Rate</TabsTrigger>
            <TabsTrigger value="bloodPressure">Blood Pressure</TabsTrigger>
            <TabsTrigger value="temperature">Temperature</TabsTrigger>
            <TabsTrigger value="oxygenSat">Oxygen Sat</TabsTrigger>
          </TabsList>

          <TabsContent value="heartRate" className="space-y-4">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={vitalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[60, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="heartRate"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "hsl(var(--chart-1))" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="bloodPressure" className="space-y-4">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={vitalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[60, 160]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="bloodPressureSys"
                    stackId="1"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                  />
                  <Area
                    type="monotone"
                    dataKey="bloodPressureDia"
                    stackId="2"
                    stroke="hsl(var(--chart-2))"
                    fill="hsl(var(--chart-2))"
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="temperature" className="space-y-4">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={vitalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[96, 102]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="temperature"
                    stroke="hsl(var(--chart-3))"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "hsl(var(--chart-3))" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="oxygenSat" className="space-y-4">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={vitalData.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[90, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="oxygenSat" fill="hsl(var(--chart-2))" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
