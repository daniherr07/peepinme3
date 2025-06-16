/**
 * PeepInMe Chatbot Logic - V8 (Highly Optimized Semantic Search)
 *
 * This version introduces major performance and correctness improvements:
 * 1. High-Speed Performance: The AI classifier is now called only ONCE per query.
 * 2. Type-Safe Math: Correctly handles numeric types from the mathjs library.
 * 3. Two-stage AI filtering remains for high accuracy.
 */

import { pipeline, ZeroShotClassificationPipeline, FeatureExtractionPipeline, ZeroShotClassificationOutput } from '@xenova/transformers';
import { dot, norm } from 'mathjs'; // Import math functions
import storesData from '../data/stores_with_embeddings.json'; // Use the data with pre-calculated embeddings

// --- TYPE DEFINITIONS (No changes needed here) ---
interface ProductEmbedding {
  product: string;
  embedding: number[];
}
export interface Store {
  id: number;
  name: string;
  category: string;
  location: { province: string; city: string; };
  product_embeddings: ProductEmbedding[];
  product_types: string[];
  hours: string;
  contact: string;
}
interface StoreWithScore extends Store {
  score: number;
}
export interface StoreGroup {
    category: string;
    stores: Store[];
}
export interface ChatbotResponse {
  introMessage: string;
  storeGroups?: StoreGroup[];
}

// --- AI MODEL MANAGEMENT (Singleton for multiple models) ---
class AIModels {
  private static classifier: Promise<ZeroShotClassificationPipeline> | null = null;
  private static extractor: Promise<FeatureExtractionPipeline> | null = null;

  static async getClassifier(): Promise<ZeroShotClassificationPipeline> {
    if (this.classifier === null) {
      this.classifier = pipeline('zero-shot-classification', 'Xenova/bart-large-mnli') as Promise<ZeroShotClassificationPipeline>;
    }
    return this.classifier;
  }
  
  static async getExtractor(): Promise<FeatureExtractionPipeline> {
    if (this.extractor === null) {
      this.extractor = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2') as Promise<FeatureExtractionPipeline>;
    }
    return this.extractor;
  }
}

// --- HELPER FUNCTIONS ---
const getRandomMessage = (messages: string[]) => messages[Math.floor(Math.random() * messages.length)];

// CORRECTED Cosine Similarity function
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

    // Cast the results from mathjs functions to 'number' to satisfy TypeScript
    const dotProduct = Number(dot(vecA, vecB));
    const normA = Number(norm(vecA));
    const normB = Number(norm(vecB));

    if (normA === 0 || normB === 0) return 0; // Prevent division by zero

    return dotProduct / (normA * normB);
}

// --- MAIN QUERY PROCESSING FUNCTION (Optimized) ---
export async function processQuery(query: string): Promise<ChatbotResponse> {
  if (!query || query.trim() === '') return { introMessage: "Please ask me something..." };

  try {
    const classifier = await AIModels.getClassifier();
    const extractor = await AIModels.getExtractor();
    const candidateLabels = [...new Set(storesData.map(store => store.category))];

    // --- AI Stage 1: Fast Category Classification (Run ONCE for performance) ---
    const categoryResults: ZeroShotClassificationOutput = await classifier(query, candidateLabels, { multi_label: true });
    // Create a score map for O(1) lookups
    const categoryScores = new Map(categoryResults.labels.map((label, i) => [label, categoryResults.scores[i]]));

    // --- AI Stage 2: Deep Semantic Analysis of User Query (Run ONCE) ---
    const queryEmbeddingOutput = await extractor(query, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(queryEmbeddingOutput.data);

    // --- Scoring Logic: Combine both AI stages in a single, fast loop ---
    const scoredStores: StoreWithScore[] = storesData.map(store => {
      // 1. Get the base score from the category filter.
      const categoryScore = categoryScores.get(store.category) || 0;
      
      // 2. Get the semantic relevance score by finding the best product match.
      let bestProductSimilarity = 0;
      for (const { embedding } of store.product_embeddings) {
          const similarity = cosineSimilarity(queryVector, embedding);
          if (similarity > bestProductSimilarity) {
              bestProductSimilarity = similarity;
          }
      }
      
      // 3. Combine scores. The semantic score heavily boosts the base category relevance.
      const finalScore = categoryScore * (1 + bestProductSimilarity);
      return { ...store, score: finalScore };
    });

    // --- Filtering, Sorting, and Grouping ---
    const topStores = scoredStores
      .filter(store => store.score > 0.5) // A fine-tuned threshold for relevance
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    if (topStores.length === 0) {
      return { introMessage: "I'm sorry, I couldn't find any stores that specifically match your request. Could you try rephrasing?" };
    }

    const groups = new Map<string, Store[]>();
    for (const store of topStores) {
        if (!groups.has(store.category)) groups.set(store.category, []);
        groups.get(store.category)!.push(store);
    }

    const finalStoreGroups = Array.from(groups.entries()).map(([category, stores]) => ({ category, stores }));

    return {
      introMessage: "Based on your request, I found these highly relevant stores:",
      storeGroups: finalStoreGroups,
    };
  } catch (error) {
    console.error("Error processing query:", error); // Keep for debugging, can be removed later
    return { introMessage: "I'm having a little trouble thinking right now. Please try again." };
  }
}