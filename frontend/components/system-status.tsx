"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Server,
  Wifi,
  Database,
  Shield,
  Camera,
  Brain,
  HardDrive,
  Cpu,
  MemoryStick,
  CheckCircle,
  AlertTriangle,
  XCircle,
} from "lucide-react"

interface SystemMetric {
  name: string
  value: number
  status: "healthy" | "warning" | "critical"
  unit: string
  icon: React.ElementType
}

interface ServiceStatus {
  name: string
  status: "online" | "degraded" | "offline"
  uptime: string
  lastCheck: string
  icon: React.ElementType
}

export function SystemStatus() {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([
    { name: "CPU Usage", value: 45, status: "healthy", unit: "%", icon: Cpu },
    { name: "Memory", value: 62, status: "healthy", unit: "%", icon: MemoryStick },
    { name: "Storage", value: 78, status: "warning", unit: "%", icon: HardDrive },
    { name: "Network", value: 95, status: "healthy", unit: "%", icon: Wifi },
  ])

  const [services, setServices] = useState<ServiceStatus[]>([
    {
      name: "Video Processing",
      status: "online",
      uptime: "99.9%",
      lastCheck: "30s ago",
      icon: Camera,
    },
    {
      name: "AI Models",
      status: "online",
      uptime: "99.7%",
      lastCheck: "45s ago",
      icon: Brain,
    },
    {
      name: "Database",
      status: "online",
      uptime: "100%",
      lastCheck: "15s ago",
      icon: Database,
    },
    {
      name: "Security",
      status: "online",
      uptime: "100%",
      lastCheck: "60s ago",
      icon: Shield,
    },
  ])

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics((prev) =>
        prev.map((metric) => ({
          ...metric,
          value: Math.max(0, Math.min(100, metric.value + (Math.random() - 0.5) * 10)),
          status: metric.value > 90 ? "critical" : metric.value > 75 ? "warning" : "healthy",
        })),
      )

      setServices((prev) =>
        prev.map((service) => ({
          ...service,
          lastCheck: `${Math.floor(Math.random() * 60)}s ago`,
          status: Math.random() > 0.95 ? "degraded" : "online", // 5% chance of degraded status
        })),
      )
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = (status: ServiceStatus["status"]) => {
    switch (status) {
      case "online":
        return <CheckCircle className="w-4 h-4 text-accent" />
      case "degraded":
        return <AlertTriangle className="w-4 h-4 text-chart-3" />
      case "offline":
        return <XCircle className="w-4 h-4 text-destructive" />
    }
  }

  const getStatusBadge = (status: ServiceStatus["status"]) => {
    switch (status) {
      case "online":
        return <Badge variant="secondary">Online</Badge>
      case "degraded":
        return <Badge variant="outline">Degraded</Badge>
      case "offline":
        return <Badge variant="destructive">Offline</Badge>
    }
  }

  const getMetricColor = (status: SystemMetric["status"]) => {
    switch (status) {
      case "healthy":
        return "text-accent"
      case "warning":
        return "text-chart-3"
      case "critical":
        return "text-destructive"
    }
  }

  const getProgressColor = (status: SystemMetric["status"]) => {
    switch (status) {
      case "healthy":
        return "bg-accent"
      case "warning":
        return "bg-chart-3"
      case "critical":
        return "bg-destructive"
    }
  }

  const overallHealth = services.every((s) => s.status === "online") ? "healthy" : "warning"

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Server className="w-5 h-5" />
            <span>System Status</span>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <div
              className={`w-2 h-2 rounded-full ${overallHealth === "healthy" ? "bg-accent" : "bg-chart-3"} pulse-dot`}
            ></div>
            <Badge variant={overallHealth === "healthy" ? "secondary" : "outline"}>
              {overallHealth === "healthy" ? "All Systems Operational" : "Some Issues Detected"}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* System Metrics */}
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">System Resources</h4>
          <div className="grid grid-cols-1 gap-4">
            {systemMetrics.map((metric) => (
              <div key={metric.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <metric.icon className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-foreground">{metric.name}</span>
                  </div>
                  <span className={`text-sm font-medium ${getMetricColor(metric.status)}`}>
                    {Math.round(metric.value)}
                    {metric.unit}
                  </span>
                </div>
                <Progress value={metric.value} className="h-2" />
              </div>
            ))}
          </div>
        </div>

        {/* Service Status */}
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Services</h4>
          <div className="space-y-3">
            {services.map((service) => (
              <div key={service.name} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <service.icon className="w-4 h-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium text-foreground">{service.name}</p>
                    <p className="text-xs text-muted-foreground">Uptime: {service.uptime}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {getStatusIcon(service.status)}
                  {getStatusBadge(service.status)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Model Performance */}
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">AI Model Performance</h4>
          <div className="grid grid-cols-1 gap-3">
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-foreground">Fall Detection</p>
                <p className="text-xs text-muted-foreground">Accuracy: 85.2%</p>
              </div>
              <Badge variant="secondary">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-foreground">Emotion Recognition</p>
                <p className="text-xs text-muted-foreground">Accuracy: 78.9%</p>
              </div>
              <Badge variant="secondary">Active</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-foreground">Vital Signs Analysis</p>
                <p className="text-xs text-muted-foreground">Accuracy: 92.1%</p>
              </div>
              <Badge variant="secondary">Active</Badge>
            </div>
          </div>
        </div>

        {/* Security Status */}
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Security & Compliance</h4>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">HIPAA Compliance</span>
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-accent" />
                <Badge variant="secondary">Verified</Badge>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Data Encryption</span>
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-accent" />
                <Badge variant="secondary">AES-256</Badge>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Access Control</span>
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-accent" />
                <Badge variant="secondary">Active</Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Last System Check */}
        <div className="pt-4 border-t border-border">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Last system check</span>
            <span>{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
