import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { TrendingUp, Users, BarChart3 } from "lucide-react"

const actions = [
  {
    title: "Transaction Analysis",
    description: "Detailed study of monetary movements",
    icon: TrendingUp,
    color: "text-chart-1",
    bgColor: "bg-chart-1/10",
    href: "/transactions",
  },
  {
    title: "Vendor Analysis",
    description: "In-depth view of supplier interactions",
    icon: Users,
    color: "text-chart-2",
    bgColor: "bg-chart-2/10",
    href: "/vendors",
  },
  {
    title: "Cash Flow Forecast",
    description: "Forward cash perspective",
    icon: BarChart3,
    color: "text-chart-3",
    bgColor: "bg-chart-3/10",
    href: "/analytics",
  },
]

export function QuickActions() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3">
      {actions.map((action) => (
        <Card key={action.title} className="p-6 hover:shadow-md transition-shadow">
          <div className="flex flex-col gap-4">
            <div className={`flex h-12 w-12 items-center justify-center rounded-lg ${action.bgColor}`}>
              <action.icon className={`h-6 w-6 ${action.color}`} />
            </div>
            <div className="flex flex-col gap-1">
              <h3 className="font-semibold">{action.title}</h3>
              <p className="text-sm text-muted-foreground">{action.description}</p>
            </div>
            <Button asChild variant="outline" size="sm" className="w-full bg-transparent">
              <Link href={action.href}>Start Analysis</Link>
            </Button>
          </div>
        </Card>
      ))}
    </div>
  )
}

