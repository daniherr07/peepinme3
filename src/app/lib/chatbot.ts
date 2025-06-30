// Este archivo es ahora un helper CLIENTE simple.
// Su único trabajo es hacer una solicitud de red a nuestra propia API del servidor.
// Todo el trabajo pesado de AI se ha eliminado de aquí.

// Importar los tipos desde el servicio de AI (no desde la ruta API)
import type { ChatbotResponse } from "./ai-service"
export type { Store, StoreGroup } from "./ai-service"

export async function processQuery(query: string): Promise<ChatbotResponse> {
  try {
    console.log(`[CLIENT] 🚀 Enviando consulta: "${query}"`)

    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    })

    if (!response.ok) {
      console.error(`[CLIENT] ❌ Error del servidor: ${response.status} ${response.statusText}`)

      // Si el servidor devuelve un error, usar su mensaje
      try {
        const errorData = await response.json()
        return {
          introMessage: errorData.introMessage || "Ocurrió un error desconocido.",
        }
      } catch (parseError) {
        console.error("[CLIENT] Error parsing error response:", parseError)
        return {
          introMessage: "Error de comunicación con el servidor.",
        }
      }
    }

    // Si es exitoso, devolver los datos JSON del servidor
    const result = await response.json()
    console.log(`[CLIENT] ✅ Respuesta recibida exitosamente`)

    return result as ChatbotResponse
  } catch (error) {
    console.error("[CLIENT] Error de red:", error)
    return {
      introMessage: "Tengo problemas para conectarme. Por favor verifica tu internet e inténtalo de nuevo.",
    }
  }
}

// Función helper adicional para validar la consulta antes de enviarla
export function validateQuery(query: string): { isValid: boolean; error?: string } {
  if (!query || typeof query !== "string") {
    return { isValid: false, error: "La consulta debe ser una cadena de texto válida" }
  }

  if (query.trim().length === 0) {
    return { isValid: false, error: "La consulta no puede estar vacía" }
  }

  if (query.length > 500) {
    return { isValid: false, error: "La consulta es demasiado larga (máximo 500 caracteres)" }
  }

  return { isValid: true }
}

// Función helper para procesar consulta con validación
export async function processQuerySafe(query: string): Promise<ChatbotResponse> {
  const validation = validateQuery(query)

  if (!validation.isValid) {
    return {
      introMessage: validation.error || "Consulta inválida",
    }
  }

  return processQuery(query.trim())
}
