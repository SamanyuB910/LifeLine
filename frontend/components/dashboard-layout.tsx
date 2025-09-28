"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Activity, Heart, AlertTriangle, Users, Settings, Shield, Video, BarChart3, Bell, Menu, X } from "lucide-react"

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const navigation = [
    { name: "Patient Monitor", icon: Activity, current: true, badge: null, section: "patient-monitor" },
    { name: "Vital Signs", icon: Heart, current: false, badge: null, section: "vital-signs" },
    { name: "Video Feeds", icon: Video, current: false, badge: "4", section: "video-feeds" },
    { name: "Analytics", icon: BarChart3, current: false, badge: null, section: "analytics" },
    { name: "Alerts", icon: AlertTriangle, current: false, badge: "3", section: "alerts" },
    { name: "Patients", icon: Users, current: false, badge: null, section: "patients" },
  ]

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
    setSidebarOpen(false) // Close sidebar on mobile after navigation
  }

  const systemNavigation = [
    { name: "Security", icon: Shield, current: false },
    { name: "Settings", icon: Settings, current: false },
  ]

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden bg-black/50" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <div
        className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
      `}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-border">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Heart className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-foreground">LifeLine</h1>
                <p className="text-xs text-muted-foreground">Patient Monitoring</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="lg:hidden" onClick={() => setSidebarOpen(false)}>
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-1">
            <div className="space-y-1">
              {navigation.map((item) => (
                <button
                  key={item.name}
                  onClick={() => scrollToSection(item.section)}
                  className={`
                    group flex items-center justify-between px-3 py-2 text-sm font-medium rounded-lg transition-smooth w-full text-left
                    ${
                      item.current
                        ? /* Updated selected tab styling to use red primary with subtle glow effect */
                          "bg-primary text-primary-foreground shadow-lg shadow-primary/20 border border-primary/30"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }
                  `}
                >
                  <div className="flex items-center space-x-3">
                    <item.icon className="w-5 h-5" />
                    <span>{item.name}</span>
                  </div>
                  {item.badge && (
                    <Badge variant="secondary" className="text-xs">
                      {item.badge}
                    </Badge>
                  )}
                </button>
              ))}
            </div>

            <div className="pt-6 mt-6 border-t border-border">
              <p className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">System</p>
              <div className="mt-2 space-y-1">
                {systemNavigation.map((item) => (
                  <a
                    key={item.name}
                    href="#"
                    className="group flex items-center px-3 py-2 text-sm font-medium text-muted-foreground rounded-lg hover:text-foreground hover:bg-muted transition-smooth"
                  >
                    <item.icon className="w-5 h-5 mr-3" />
                    {item.name}
                  </a>
                ))}
              </div>
            </div>
          </nav>

          {/* System status */}
          <div className="p-4 border-t border-border">
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-accent rounded-full pulse-dot"></div>
              <div>
                <p className="text-sm font-medium text-foreground">System Online</p>
                <p className="text-xs text-muted-foreground">All systems operational</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="sticky top-0 z-30 flex h-16 items-center justify-between bg-background/95 backdrop-blur-sm border-b border-border px-6">
          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm" className="lg:hidden" onClick={() => setSidebarOpen(true)}>
              <Menu className="w-5 h-5" />
            </Button>
            <div>
              <h2 className="text-lg font-semibold text-foreground">Patient Monitoring Dashboard</h2>
              <p className="text-sm text-muted-foreground">Real-time health monitoring and alerts</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm" className="relative">
              <Bell className="w-5 h-5" />
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-destructive rounded-full text-xs flex items-center justify-center text-destructive-foreground">
                3
              </span>
            </Button>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center">
                <span className="text-sm font-medium">DR</span>
              </div>
              <div className="hidden sm:block">
                <p className="text-sm font-medium text-foreground">Dr. Smith</p>
                <p className="text-xs text-muted-foreground">Attending Physician</p>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="min-h-[calc(100vh-4rem)]">{children}</main>
      </div>
    </div>
  )
}
