/**
 * PeepInMe Chatbot Logic - V7 (Semantic Search AI)
 *
 * This version uses a powerful two-stage AI filtering system:
 * 1. AI Category Filter: A fast, high-level pass to find relevant store types.
 * 2. AI Semantic Filter: A deep-meaning search to find the most relevant products.
 */

import { pipeline, ZeroShotClassificationPipeline, FeatureExtractionPipeline } from '@xenova/transformers';
import { dot, norm, cos, transpose } from 'mathjs'; // Import math functions for vector comparison
import storesData from '../data/stores_with_embeddings.json'; // IMPORTANT: Use the new data file

// --- TYPE DEFINITIONS ---
interface ProductEmbedding {
  product: string;
  embedding: number[];
}

// Update the Store interface to use the new product structure
export interface Store {
  id: number;
  name: string;
  category: string;
  location: { province: string; city: string; };
  product_embeddings: ProductEmbedding[]; // This is the new field
  product_types: string[]; // Keep this for display purposes
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

// --- AI MODEL MANAGEMENT (Singleton pattern for multiple models) ---
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

function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    // Using mathjs for a robust calculation: (A . B) / (||A|| * ||B||)
    return dot(vecA, vecB) / (norm(vecA) * norm(vecB));
}

// --- MAIN QUERY PROCESSING FUNCTION ---
export async function processQuery(query: string): Promise<ChatbotResponse> {
  if (!query || query.trim() === '') return { introMessage: "Please ask me something..." };

  try {
    const classifier = await AIModels.getClassifier();
    const extractor = await AIModels.getExtractor();
    const candidateLabels = [...new Set(storesData.map(store => store.category))];

    // --- AI Stage 1: Fast Category Classification ---
    const categoryResults = await classifier(query, candidateLabels, { multi_label: true });
    const categoryScores = new Map(categoryResults.labels.map((label, i) => [label, categoryResults.scores[i]]));

    // --- AI Stage 2: Deep Semantic Analysis of User Query ---
    const queryEmbeddingOutput = await extractor(query, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(queryEmbeddingOutput.data);

    // --- Scoring Logic: Combine both AI stages ---
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
      
      // 3. Combine scores: The category score acts as a gate, and the semantic score refines the ranking.
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
    return { introMessage: "I'm having a little trouble thinking right now. Please try again." };
  }
}