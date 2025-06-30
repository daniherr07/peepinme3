// --- VERCEL DEPLOYMENT CONFIGURATION ---
export const runtime = "nodejs"
export const dynamic = "force-dynamic"
export const maxDuration = 60 // 60 segundos para evitar timeouts

import { AIService } from "../../lib/ai-service"
import type { ChatbotResponse } from "../../lib/ai-service"

// --- MAIN API HANDLER for POST requests ---
export async function POST(request: Request) {
  try {
    console.log(`\n--- New API Request ---`)
    console.log(`[${new Date().toISOString()}] 🚀 Starting request processing`)

    // Parse request body
    const body = await request.json()
    const { query } = body

    // Validate input
    if (!query || typeof query !== "string" || query.trim().length === 0) {
      console.log(`[ERROR] Invalid query received:`, query)
      return Response.json(
        {
          introMessage: "Por favor, proporciona una consulta válida.",
        },
        { status: 400 },
      )
    }

    // Process query with AI service
    const response: ChatbotResponse = await AIService.processQuery(query.trim())

    console.log(`[LOG] ✅ Request completed successfully`)
    return Response.json(response)
  } catch (error) {
    console.error("API Route Error:", error)

    // Return user-friendly error message
    return Response.json(
      {
        introMessage: "Estoy teniendo algunos problemas técnicos. Por favor, inténtalo de nuevo en unos momentos.",
      },
      { status: 500 },
    )
  }
}

// Optional: Add GET handler for health check
export async function GET() {
  return Response.json({
    status: "ok",
    message: "AI Chatbot API is running",
    timestamp: new Date().toISOString(),
  })
}
