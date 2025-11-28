import { DashboardLayout } from "@/components/layout/dashboard-layout"
import { UploadInterface } from "@/components/features/upload-interface"
import { AnalysisOverview } from "@/components/features/analysis-overview"
import { QuickActions } from "@/components/features/quick-actions"

export default function HomePage() {
  return (
    <DashboardLayout>
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight text-balance">Cash Flow Analysis</h1>
          <p className="text-muted-foreground text-balance">
          Financial flow analysis and reporting
          </p>
        </div>

        <UploadInterface />

        <QuickActions />

        <AnalysisOverview />
      </div>
    </DashboardLayout>
  )
}

