import {
  pipeline,
  type ZeroShotClassificationPipeline,
  type FeatureExtractionPipeline,
  type ZeroShotClassificationOutput,
} from "@xenova/transformers"
import { dot, norm } from "mathjs"
import storesData from "../data/stores_with_embeddings.json"

// --- TYPE DEFINITIONS ---
interface ProductEmbedding {
  product: string
  embedding: number[]
}

export interface Store {
  id: number
  name: string
  category: string
  location: {
    province: string
    city: string
  }
  product_embeddings: ProductEmbedding[]
  product_types: string[]
  hours: string
  contact: string
}

interface StoreWithScore extends Store {
  score: number
}

export interface StoreGroup {
  category: string
  stores: Store[]
}

export interface ChatbotResponse {
  introMessage: string
  storeGroups?: StoreGroup[]
}

// --- In-Memory Index (For Performance) ---
console.log("🚀 Building In-Memory Index...")

const categoryIndex = (storesData as Store[]).reduce((index, store) => {
  if (!index.has(store.category)) index.set(store.category, [])
  index.get(store.category)!.push(store)
  return index
}, new Map<string, Store[]>())

console.log(`✅ In-Memory Index built with ${categoryIndex.size} categories.`)

// --- AI Model Management ---
class AIModels {
  private static classifier: Promise<ZeroShotClassificationPipeline> | null = null
  private static extractor: Promise<FeatureExtractionPipeline> | null = null

  static async getClassifier(): Promise<ZeroShotClassificationPipeline> {
    if (!this.classifier) {
      console.log("🤖 Loading classification model...")
      this.classifier = pipeline("zero-shot-classification", "Xenova/bart-large-mnli")
    }
    return this.classifier
  }

  static async getExtractor(): Promise<FeatureExtractionPipeline> {
    if (!this.extractor) {
      console.log("🔍 Loading feature extraction model...")
      this.extractor = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")
    }
    return this.extractor
  }
}

// --- HELPER FUNCTIONS ---
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0

  const dotProduct = Number(dot(vecA, vecB))
  const normA = Number(norm(vecA))
  const normB = Number(norm(vecB))

  if (normA === 0 || normB === 0) return 0

  return dotProduct / (normA * normB)
}

// --- MAIN AI SERVICE CLASS ---
export class AIService {
  static async processQuery(query: string): Promise<ChatbotResponse> {
    const startTime = Date.now()

    console.log(`\n--- Processing Query ---`)
    console.log(`[${new Date().toISOString()}] ➡️ Query: "${query}"`)

    try {
      // Load AI models
      const classifier = await AIModels.getClassifier()
      const extractor = await AIModels.getExtractor()

      // Get candidate labels (categories)
      const candidateLabels = Array.from(categoryIndex.keys())

      // Classify query into categories
      const rawCategoryResults: ZeroShotClassificationOutput | ZeroShotClassificationOutput[] = await classifier(
        query,
        candidateLabels,
        { multi_label: true },
      )

      if (Array.isArray(rawCategoryResults)) {
        throw new Error("Classifier returned an array for a single query, which is unexpected.")
      }

      const categoryResults: ZeroShotClassificationOutput = rawCategoryResults
      const categoryScores = new Map(categoryResults.labels.map((label, i) => [label, categoryResults.scores[i]]))

      console.log(`[LOG] AI Category Scores:`, Object.fromEntries(categoryScores))

      // Get candidate stores from relevant categories
      const candidateStores: Store[] = []
      for (const [category, score] of categoryScores.entries()) {
        if (score > 0.4) {
          candidateStores.push(...(categoryIndex.get(category) || []))
        }
      }

      const uniqueCandidateStores = [...new Map(candidateStores.map((item) => [item["id"], item])).values()]

      console.log(`[LOG] Found ${uniqueCandidateStores.length} candidate stores from relevant categories.`)

      if (uniqueCandidateStores.length === 0) {
        return {
          introMessage: "Lo siento, no pude encontrar tiendas relacionadas con ese tema.",
        }
      }

      // Generate query embedding
      const queryEmbeddingOutput = await extractor(query, {
        pooling: "mean",
        normalize: true,
      })
      const queryVector = Array.from(queryEmbeddingOutput.data)

      // Score stores based on similarity
      const scoredStores: StoreWithScore[] = uniqueCandidateStores.map((store) => {
        let bestProductSimilarity = 0

        for (const { embedding } of store.product_embeddings) {
          const similarity = cosineSimilarity(queryVector, embedding)
          if (similarity > bestProductSimilarity) {
            bestProductSimilarity = similarity
          }
        }

        const finalScore = (categoryScores.get(store.category) || 0) * (1 + bestProductSimilarity)
        return { ...store, score: finalScore }
      })

      // Get top stores
      const topStores = scoredStores.sort((a, b) => b.score - a.score).slice(0, 5)

      console.log(
        `[LOG] Top 5 stores:`,
        topStores.map((s) => ({
          name: s.name,
          score: s.score,
        })),
      )

      // Prepare response
      let response: ChatbotResponse

      if (topStores.length === 0) {
        response = {
          introMessage:
            "Encontré algunas categorías relacionadas, pero no productos específicos. ¿Podrías ser más específico?",
        }
      } else {
        const groups = new Map<string, Store[]>()

        for (const store of topStores) {
          if (!groups.has(store.category)) {
            groups.set(store.category, [])
          }
          groups.get(store.category)!.push(store)
        }

        const finalStoreGroups = Array.from(groups.entries()).map(([category, stores]) => ({
          category,
          stores,
        }))

        response = {
          introMessage: "Basado en tu consulta, encontré estas tiendas muy relevantes:",
          storeGroups: finalStoreGroups,
        }
      }

      const endTime = Date.now()
      console.log(`[LOG] ✅ Query processed in ${endTime - startTime}ms.`)

      return response
    } catch (error) {
      console.error("AI Service Error:", error)
      throw new Error(`Error procesando la consulta: ${error instanceof Error ? error.message : "Error desconocido"}`)
    }
  }
}
