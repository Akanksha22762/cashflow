"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FileSpreadsheet, CheckCircle2, Loader2 } from "lucide-react"

export function UploadSection() {
  const [uploading, setUploading] = useState(false)
  const [uploaded, setUploaded] = useState(false)

  const handleUpload = async () => {
    setUploading(true)
    // Simulate upload
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setUploading(false)
    setUploaded(true)
  }

  return (
    <Card className="border-2 border-dashed border-border bg-muted/30">
      <div className="flex flex-col items-center justify-center gap-4 p-12">
        {!uploaded ? (
          <>
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <Upload className="h-8 w-8 text-primary" />
            </div>
            <div className="text-center">
              <h3 className="text-lg font-semibold">Upload Bank Statement</h3>
              <p className="text-sm text-muted-foreground mt-1">Upload your bank statement file to begin analysis</p>
            </div>
            <div className="flex gap-3">
              <Button onClick={handleUpload} disabled={uploading} className="gap-2">
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <FileSpreadsheet className="h-4 w-4" />
                    Select File
                  </>
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">Supports XLSX, XLS, and CSV formats</p>
          </>
        ) : (
          <>
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <CheckCircle2 className="h-8 w-8 text-primary" />
            </div>
            <div className="text-center">
              <h3 className="text-lg font-semibold">File Uploaded Successfully</h3>
              <p className="text-sm text-muted-foreground mt-1">Your data is ready for analysis</p>
            </div>
            <Button variant="outline" onClick={() => setUploaded(false)}>
              Upload Another File
            </Button>
          </>
        )}
      </div>
    </Card>
  )
}
