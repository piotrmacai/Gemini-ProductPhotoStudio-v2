export enum ModuleType {
  STUDIO = 'STUDIO',
  CAMPAIGN = 'CAMPAIGN'
}

export enum AspectRatio {
  SQUARE = '1:1',
  PORTRAIT = '3:4',
  LANDSCAPE = '4:3',
  VERTICAL = '9:16',
  WIDE = '16:9',
  CINEMATIC = '21:9',
  STD_PORTRAIT = '2:3',
  STD_LANDSCAPE = '3:2'
}

export enum ImageResolution {
  RES_1K = '1K',
  RES_2K = '2K',
  RES_4K = '4K'
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export interface AdOverlay {
  id: string;
  text: string;
  x: number;
  y: number;
  type: 'headline' | 'subtext';
  fontFamily: string;
  color: string;
}

export interface GeneratedImage {
  id: string;
  url: string; // Base64 or external URL
  prompt: string;
  timestamp: number;
}