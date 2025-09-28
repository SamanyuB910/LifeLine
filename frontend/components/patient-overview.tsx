"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Heart, Activity, Thermometer, Droplets, User, AlertTriangle } from "lucide-react"

export function PatientOverview() {
  const patientData = {
    name: "Sarah Johnson",
    id: "P-2024-001",
    room: "ICU-204",
    age: 67,
    condition: "Post-surgical monitoring",
    riskLevel: "Medium",
    lastUpdate: "2 minutes ago",
  }

  const vitalSigns = [
    {
      label: "Heart Rate",
      value: "78",
      unit: "BPM",
      status: "normal",
      icon: Heart,
      color: "text-accent",
    },
    {
      label: "Blood Pressure",
      value: "120/80",
      unit: "mmHg",
      status: "normal",
      icon: Activity,
      color: "text-primary",
    },
    {
      label: "Temperature",
      value: "98.6",
      unit: "°F",
      status: "normal",
      icon: Thermometer,
      color: "text-chart-3",
    },
    {
      label: "Oxygen Sat",
      value: "97",
      unit: "%",
      status: "normal",
      icon: Droplets,
      color: "text-chart-2",
    },
  ]

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <User className="w-5 h-5" />
            <span>Patient Overview</span>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant={patientData.riskLevel === "High" ? "destructive" : "secondary"}>
              {patientData.riskLevel} Risk
            </Badge>
            <div className="w-2 h-2 bg-accent rounded-full pulse-dot"></div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Patient Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-muted-foreground">Patient Name</p>
            <p className="font-semibold text-foreground">{patientData.name}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Patient ID</p>
            <p className="font-semibold text-foreground">{patientData.id}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Room</p>
            <p className="font-semibold text-foreground">{patientData.room}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Age</p>
            <p className="font-semibold text-foreground">{patientData.age} years</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Condition</p>
            <p className="font-semibold text-foreground">{patientData.condition}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Last Update</p>
            <p className="font-semibold text-foreground">{patientData.lastUpdate}</p>
          </div>
        </div>

        {/* Vital Signs Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {vitalSigns.map((vital) => (
            <div key={vital.label} className="bg-muted/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <vital.icon className={`w-5 h-5 ${vital.color}`} />
                <Badge variant="outline" className="text-xs">
                  {vital.status}
                </Badge>
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {vital.value}
                  <span className="text-sm text-muted-foreground ml-1">{vital.unit}</span>
                </p>
                <p className="text-sm text-muted-foreground">{vital.label}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm">
            View History
          </Button>
          <Button variant="outline" size="sm">
            Update Notes
          </Button>
          <Button variant="outline" size="sm">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Emergency Alert
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
