import { GoogleGenAI, Chat, GenerateContentResponse } from "@google/genai";
import { AspectRatio, ImageResolution } from '../types';

// Initialize the API client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export interface ReferenceImages {
  product?: string;
  model?: string;
  background?: string;
}

/**
 * Helper to extract mime type and base64 data correctly.
 */
const processBase64 = (base64String: string) => {
  const mimeMatch = base64String.match(/^data:([^;]+);base64,/);
  const mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
  const data = base64String.replace(/^data:([^;]+);base64,/, '');
  return { mimeType, data };
};

/**
 * Generate a high-quality fashion image using Gemini.
 * Supports switching between Gemini 2.5 Flash (Nano Banana) and Gemini 3 Pro (Nano Banana Pro).
 */
export const generateFashionImage = async (
  prompt: string,
  aspectRatio: AspectRatio,
  resolution: ImageResolution,
  references: ReferenceImages,
  modelVersion: '2.5' | '3' = '2.5'
): Promise<string> => {
  try {
    const parts: any[] = [];

    // Construct the multimodal prompt with labeled references
    if (references.product) {
      const { mimeType, data } = processBase64(references.product);
      parts.push({ text: "Primary Product Reference (The garment/item to feature):" });
      parts.push({
        inlineData: { mimeType, data }
      });
    }

    if (references.model) {
      const { mimeType, data } = processBase64(references.model);
      parts.push({ text: "Model Reference (Person/Pose style):" });
      parts.push({
        inlineData: { mimeType, data }
      });
    }

    if (references.background) {
      const { mimeType, data } = processBase64(references.background);
      parts.push({ text: "Background/Scene Reference:" });
      parts.push({
        inlineData: { mimeType, data }
      });
    }

    // Add the main text prompt last
    parts.push({ text: `Instructions: ${prompt}` });

    // Determine model based on selection
    const modelName = modelVersion === '3' ? 'gemini-3-pro-image-preview' : 'gemini-2.5-flash-image';

    const config: any = {
      imageConfig: {
        aspectRatio: aspectRatio,
      }
    };

    // Image resolution (1K, 2K, 4K) is only supported by gemini-3-pro-image-preview
    if (modelVersion === '3') {
      config.imageConfig.imageSize = resolution;
    }

    const response = await ai.models.generateContent({
      model: modelName,
      contents: {
        parts: parts,
      },
      config: config,
    });

    // Extract image from response
    if (response.candidates && response.candidates[0].content.parts) {
      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData && part.inlineData.data) {
          return `data:image/png;base64,${part.inlineData.data}`;
        }
      }
    }
    throw new Error("No image generated.");
  } catch (error) {
    console.error("Gemini Generation Error:", error);
    throw error;
  }
};

/**
 * Edit an existing image using Gemini 2.5 Flash Image.
 * Ideal for "Add a retro filter", "Remove background person".
 * Supports optional sketch mask.
 */
export const editFashionImage = async (
  imageBase64: string,
  prompt: string,
  maskBase64?: string
): Promise<string> => {
  try {
    const { mimeType: imageMime, data: imageData } = processBase64(imageBase64);
    
    const parts: any[] = [
      {
        inlineData: {
          mimeType: imageMime,
          data: imageData,
        },
      }
    ];

    if (maskBase64) {
       const { mimeType: maskMime, data: maskData } = processBase64(maskBase64);
       parts.push({
         inlineData: {
           mimeType: maskMime,
           data: maskData
         }
       });
       parts.push({ text: "Use the provided sketch/mask image as a guide for the edit." });
    }

    parts.push({ text: prompt });

    // Using gemini-2.5-flash-image for editing tasks
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: parts,
      },
    });

    if (response.candidates && response.candidates[0].content.parts) {
      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData && part.inlineData.data) {
          return `data:image/png;base64,${part.inlineData.data}`;
        }
      }
    }
    throw new Error("No edited image generated.");
  } catch (error) {
    console.error("Gemini Edit Error:", error);
    throw error;
  }
};

/**
 * Analyze an uploaded image (e.g., flat lay garment) to generate a description.
 * Uses Gemini 3 Pro Preview for deep understanding.
 */
export const analyzeImage = async (imageBase64: string): Promise<string> => {
  try {
    const { mimeType, data } = processBase64(imageBase64);

    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: mimeType,
              data: data,
            },
          },
          {
            text: "Analyze this fashion image. Describe the garment style, material, color, and key details in a concise paragraph suitable for a fashion catalog.",
          },
        ],
      },
    });

    return response.text || "Could not analyze image.";
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    return "Analysis failed. Please try again.";
  }
};

/**
 * Chat assistant for the app.
 */
export const createChatSession = (): Chat => {
  return ai.chats.create({
    model: 'gemini-3-pro-preview',
    config: {
      systemInstruction: "You are an expert Fashion Director and AI Technical Assistant. You help users design outfits, suggest campaign ideas, and troubleshoot the FashionGen Studio app. Keep answers professional, chic, and concise.",
    },
  });
};