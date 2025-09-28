"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle,
  X,
  Bell,
  BellOff,
  Clock,
  User,
  Heart,
  Activity,
  Eye,
  Shield,
} from "lucide-react"

interface Alert {
  id: string
  type: "critical" | "high" | "medium" | "low"
  category: "fall_risk" | "vital_signs" | "behavior" | "system" | "security"
  title: string
  message: string
  timestamp: Date
  acknowledged: boolean
  patient?: string
  room?: string
}

export function AlertsPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filter, setFilter] = useState<"all" | "critical" | "high" | "medium" | "low">("all")
  const [soundEnabled, setSoundEnabled] = useState(true)

  // Generate initial alerts
  useEffect(() => {
    const initialAlerts: Alert[] = [
      {
        id: "1",
        type: "critical",
        category: "fall_risk",
        title: "High Fall Risk Detected",
        message: "Patient showing unstable posture and increased movement patterns",
        timestamp: new Date(Date.now() - 2 * 60000),
        acknowledged: false,
        patient: "Sarah Johnson",
        room: "ICU-204",
      },
      {
        id: "2",
        type: "high",
        category: "vital_signs",
        title: "Heart Rate Elevated",
        message: "Heart rate consistently above 100 BPM for 5 minutes",
        timestamp: new Date(Date.now() - 5 * 60000),
        acknowledged: false,
        patient: "Sarah Johnson",
        room: "ICU-204",
      },
      {
        id: "3",
        type: "medium",
        category: "behavior",
        title: "Emotional State Change",
        message: "Facial expression analysis indicates increased stress levels",
        timestamp: new Date(Date.now() - 8 * 60000),
        acknowledged: true,
        patient: "Sarah Johnson",
        room: "ICU-204",
      },
      {
        id: "4",
        type: "low",
        category: "system",
        title: "Camera Quality Reduced",
        message: "Video feed quality automatically reduced due to network conditions",
        timestamp: new Date(Date.now() - 15 * 60000),
        acknowledged: true,
        patient: "Sarah Johnson",
        room: "ICU-204",
      },
    ]
    setAlerts(initialAlerts)
  }, [])

  // Simulate new alerts
  useEffect(() => {
    const interval = setInterval(() => {
      const alertTypes = ["critical", "high", "medium", "low"] as const
      const categories = ["fall_risk", "vital_signs", "behavior", "system", "security"] as const
      const messages = {
        fall_risk: [
          "Sudden movement detected",
          "Patient attempting to leave bed",
          "Posture instability detected",
          "Fall risk score increased",
        ],
        vital_signs: [
          "Blood pressure spike detected",
          "Temperature fluctuation",
          "Oxygen saturation below threshold",
          "Irregular heart rhythm",
        ],
        behavior: [
          "Agitation levels increased",
          "Sleep pattern disruption",
          "Reduced responsiveness",
          "Unusual movement patterns",
        ],
        system: [
          "Network connectivity issue",
          "Model accuracy degraded",
          "Storage space low",
          "Backup system activated",
        ],
        security: [
          "Unauthorized access attempt",
          "Data encryption verified",
          "Audit log updated",
          "Security scan completed",
        ],
      }

      if (Math.random() < 0.3) {
        // 30% chance of new alert
        const type = alertTypes[Math.floor(Math.random() * alertTypes.length)]
        const category = categories[Math.floor(Math.random() * categories.length)]
        const messageList = messages[category]
        const message = messageList[Math.floor(Math.random() * messageList.length)]

        const newAlert: Alert = {
          id: Date.now().toString(),
          type,
          category,
          title: `${category.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase())} Alert`,
          message,
          timestamp: new Date(),
          acknowledged: false,
          patient: "Sarah Johnson",
          room: "ICU-204",
        }

        setAlerts((prev) => [newAlert, ...prev].slice(0, 20)) // Keep only last 20 alerts
      }
    }, 10000) // Check every 10 seconds

    return () => clearInterval(interval)
  }, [])

  const acknowledgeAlert = (id: string) => {
    setAlerts((prev) => prev.map((alert) => (alert.id === id ? { ...alert, acknowledged: true } : alert)))
  }

  const dismissAlert = (id: string) => {
    setAlerts((prev) => prev.filter((alert) => alert.id !== id))
  }

  const getAlertIcon = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return <AlertTriangle className="w-4 h-4 text-destructive" />
      case "high":
        return <AlertCircle className="w-4 h-4 text-chart-3" />
      case "medium":
        return <Info className="w-4 h-4 text-primary" />
      case "low":
        return <CheckCircle className="w-4 h-4 text-accent" />
    }
  }

  const getCategoryIcon = (category: Alert["category"]) => {
    switch (category) {
      case "fall_risk":
        return <User className="w-4 h-4" />
      case "vital_signs":
        return <Heart className="w-4 h-4" />
      case "behavior":
        return <Eye className="w-4 h-4" />
      case "system":
        return <Activity className="w-4 h-4" />
      case "security":
        return <Shield className="w-4 h-4" />
    }
  }

  const getAlertBadgeVariant = (type: Alert["type"]) => {
    switch (type) {
      case "critical":
        return "destructive"
      case "high":
        return "outline"
      case "medium":
        return "secondary"
      case "low":
        return "outline"
    }
  }

  const filteredAlerts = alerts.filter((alert) => filter === "all" || alert.type === filter)
  const unacknowledgedCount = alerts.filter((alert) => !alert.acknowledged).length

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date()
    const diffMs = now.getTime() - timestamp.getTime()
    const diffMins = Math.floor(diffMs / 60000)

    if (diffMins < 1) return "Just now"
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    return timestamp.toLocaleDateString()
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Bell className="w-5 h-5" />
            <span>Active Alerts</span>
            {unacknowledgedCount > 0 && (
              <Badge variant="destructive" className="text-xs">
                {unacknowledgedCount}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={soundEnabled ? "text-primary" : "text-muted-foreground"}
            >
              {soundEnabled ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Alert Filters */}
        <div className="flex flex-wrap gap-2">
          {["all", "critical", "high", "medium", "low"].map((filterType) => (
            <Button
              key={filterType}
              variant={filter === filterType ? "default" : "outline"}
              size="sm"
              onClick={() => setFilter(filterType as typeof filter)}
              className="text-xs"
            >
              {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
              {filterType !== "all" && (
                <Badge variant="secondary" className="ml-2 text-xs">
                  {alerts.filter((a) => a.type === filterType).length}
                </Badge>
              )}
            </Button>
          ))}
        </div>

        {/* Alerts List */}
        <ScrollArea className="h-96">
          <div className="space-y-3">
            {filteredAlerts.length === 0 ? (
              <div className="text-center py-8">
                <CheckCircle className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">No alerts to display</p>
              </div>
            ) : (
              filteredAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`
                    p-4 rounded-lg border transition-smooth
                    ${
                      alert.acknowledged
                        ? "bg-muted/30 border-border opacity-60"
                        : "bg-card border-border hover:bg-muted/50"
                    }
                    ${alert.type === "critical" && !alert.acknowledged ? "border-destructive/50" : ""}
                  `}
                >
                  <div className="flex items-start justify-between space-x-3">
                    <div className="flex items-start space-x-3 flex-1">
                      <div className="flex items-center space-x-2 mt-1">
                        {getAlertIcon(alert.type)}
                        {getCategoryIcon(alert.category)}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-semibold text-foreground">{alert.title}</h4>
                          <Badge variant={getAlertBadgeVariant(alert.type)} className="text-xs">
                            {alert.type}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{alert.message}</p>
                        <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                          <div className="flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>{formatTimeAgo(alert.timestamp)}</span>
                          </div>
                          {alert.patient && (
                            <div className="flex items-center space-x-1">
                              <User className="w-3 h-3" />
                              <span>{alert.patient}</span>
                            </div>
                          )}
                          {alert.room && <span>Room {alert.room}</span>}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-1">
                      {!alert.acknowledged && (
                        <Button variant="ghost" size="sm" onClick={() => acknowledgeAlert(alert.id)}>
                          <CheckCircle className="w-4 h-4" />
                        </Button>
                      )}
                      <Button variant="ghost" size="sm" onClick={() => dismissAlert(alert.id)}>
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Alert Summary */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
          <div className="text-center">
            <p className="text-2xl font-bold text-foreground">{unacknowledgedCount}</p>
            <p className="text-xs text-muted-foreground">Unacknowledged</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-foreground">{alerts.length}</p>
            <p className="text-xs text-muted-foreground">Total Alerts</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
