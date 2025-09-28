import { DashboardLayout } from "@/components/dashboard-layout"
import { PatientOverview } from "@/components/patient-overview"
import { LiveVideoFeed } from "@/components/live-video-feed"
import { VitalSignsCharts } from "@/components/vital-signs-charts"
import { AlertsPanel } from "@/components/alerts-panel"
import { SystemStatus } from "@/components/system-status"
import { LatestSnapshot } from "@/components/latest-snapshot"

export default function HomePage() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Patient Monitor Section */}
        <section id="patient-monitor" className="scroll-mt-16">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-6">Patient Monitor</h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <LiveVideoFeed />
              </div>
              <div>
                <PatientOverview />
              </div>
            </div>
          </div>
        </section>

        {/* Vital Signs Section */}
        <section id="vital-signs" className="scroll-mt-16">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-6">Vital Signs</h2>
            <VitalSignsCharts />
          </div>
        </section>

        {/* Video Feeds Section */}
        <section id="video-feeds" className="scroll-mt-16">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-6">Video Feeds</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LiveVideoFeed />
              <LatestSnapshot />
            </div>
          </div>
        </section>

        {/* Analytics Section */}
        <section id="analytics" className="scroll-mt-16">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-6">Analytics</h2>
            <VitalSignsCharts />
          </div>
        </section>

        {/* Alerts Section */}
        <section id="alerts" className="scroll-mt-16">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-6">Alerts</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AlertsPanel />
              <SystemStatus />
            </div>
          </div>
        </section>

        {/* Patients Section */}
        <section id="patients" className="scroll-mt-16">
          <div className="p-6">
            <h2 className="text-2xl font-bold text-foreground mb-6">Patients</h2>
            <PatientOverview />
          </div>
        </section>
      </div>
    </DashboardLayout>
  )
}
