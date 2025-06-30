// This file is now a simple CLIENT-SIDE helper.
// Its only job is to make a network request to our own server-side API.
// All the heavy AI work is gone from here.

// Export the same types so the UI components don't need to change.
export type { Store, StoreGroup, ChatbotResponse } from '../api/chat/route';

export async function processQuery(query: string): Promise<any> {
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            // If the server returns an error, use its message
            const errorData = await response.json();
            return { introMessage: errorData.introMessage || "An unknown error occurred." };
        }

        // If successful, return the JSON data from the server
        return await response.json();

    } catch (error) {
        console.error("Network error:", error);
        return { introMessage: "I'm having trouble connecting. Please check your internet and try again." };
    }
}