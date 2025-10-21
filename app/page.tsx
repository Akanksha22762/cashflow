import { DashboardLayout } from "@/components/dashboard-layout"
import { UploadInterface } from "@/components/upload-interface"
import { AnalysisOverview } from "@/components/analysis-overview"
import { QuickActions } from "@/components/quick-actions"

export default function HomePage() {
  return (
    <DashboardLayout>
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight text-balance">Cash Flow Analysis</h1>
          <p className="text-muted-foreground text-balance">
            AI-powered transaction categorization and financial insights
          </p>
        </div>

        <UploadInterface />

        <QuickActions />

        <AnalysisOverview />
      </div>
    </DashboardLayout>
  )
}
