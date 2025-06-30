export async function processQuery(query) {
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

    return result
  } catch (error) {
    console.error("[CLIENT] Error de red:", error)
    return {
      introMessage: "Tengo problemas para conectarme. Por favor verifica tu internet e inténtalo de nuevo.",
    }
  }
}
