

// --- MAIN AI SERVICE CLASS ---
export class AIService {
  static async processQuery(query) {
  try{
      response = {
        introMessage: "Basado en tu consulta, encontré estas tiendas muy relevantes:"
      }
      return response
    } catch (error) {
      console.error("AI Service Error:", error)
      throw new Error(`Error procesando la consulta: ${error instanceof Error ? error.message : "Error desconocido"}`)
    }
  }
}
