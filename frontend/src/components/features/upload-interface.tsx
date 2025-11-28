"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Upload, FileSpreadsheet, CheckCircle2, Loader2, X } from "lucide-react"

export function UploadInterface() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<"idle" | "processing" | "complete" | "error">("idle")
  const [error, setError] = useState<string>("")
  const router = useRouter()

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setStatus("idle")
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setStatus("processing")
    setProgress(0)
    setError("")

    try {
      const form = new FormData()
      // Backend expects 'bank_file' (primary) and optionally 'sap_file'
      form.append("bank_file", file)

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      })

      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `Upload failed (${res.status})`)
      }

      // Basic progress simulation while waiting
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval)
            return 100
          }
          return prev + 10
        })
      }, 150)

      // Parse to ensure JSON and then mark complete
      await res.json().catch(() => ({}))
      setStatus("complete")
      setUploading(false)

      // Navigate to Transactions to view parsed rows
      setTimeout(() => router.push("/transactions"), 500)
    } catch (e: any) {
      setUploading(false)
      setStatus("error")
      setError(e?.message || "Upload failed")
    }
  }

  const handleReset = () => {
    setFile(null)
    setStatus("idle")
    setProgress(0)
  }

  return (
    <div className="grid gap-6">
      <Card className="p-8">
        <h2 className="text-xl font-semibold mb-6">File Upload</h2>

        {!file ? (
          <label className="flex flex-col items-center justify-center gap-4 p-12 border-2 border-dashed border-border rounded-lg cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <Upload className="h-8 w-8 text-primary" />
            </div>
            <div className="text-center">
              <p className="font-medium">Click to upload or drag and drop</p>
              <p className="text-sm text-muted-foreground mt-1">XLSX, XLS, or CSV (max 10MB)</p>
            </div>
            <input type="file" className="hidden" accept=".xlsx,.xls,.csv" onChange={handleFileSelect} />
          </label>
        ) : (
          <div className="space-y-6">
            <div className="flex items-start gap-4 p-4 bg-muted rounded-lg">
              <FileSpreadsheet className="h-10 w-10 text-primary flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{file.name}</p>
                <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              {status === "idle" && (
                <Button variant="ghost" size="icon" onClick={handleReset}>
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>

            {status === "processing" && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Processing...</span>
                  <span className="font-medium">{progress}%</span>
                </div>
                <Progress value={progress} />
                {error && <div className="text-red-600 text-sm mt-2">{error}</div>}
              </div>
            )}

            {status === "complete" && (
              <div className="flex items-center gap-3 p-4 bg-primary/10 rounded-lg">
                <CheckCircle2 className="h-5 w-5 text-primary" />
                <div className="flex-1">
                  <p className="font-medium">Upload Complete</p>
                  <p className="text-sm text-muted-foreground">File processed successfully</p>
                </div>
              </div>
            )}

            {status === "error" && (
              <div className="text-red-600 text-sm">{error || "Upload failed"}</div>
            )}

            {status === "idle" && (
              <Button onClick={handleUpload} className="w-full">
                Process File
              </Button>
            )}

            {status === "complete" && (
              <Button onClick={handleReset} variant="outline" className="w-full bg-transparent">
                Upload Another File
              </Button>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}

