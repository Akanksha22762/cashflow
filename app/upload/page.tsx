import { DashboardLayout } from "@/components/dashboard-layout"
import { UploadInterface } from "@/components/upload-interface"

export default function UploadPage() {
  return (
    <DashboardLayout>
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight">Upload Data</h1>
          <p className="text-muted-foreground">Upload your bank statement for AI-powered analysis</p>
        </div>

        <UploadInterface />
      </div>
    </DashboardLayout>
  )
}
