import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download, Sparkles, Layers, Type, Trash2, Wand2, Loader2, ImagePlus, Plus, X, User, Image as ImageIcon, Shirt, Zap, Eraser, Brush, RotateCcw } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatAssistant from './components/ChatAssistant';
import { ModuleType, AspectRatio, ImageResolution, AdOverlay, GeneratedImage } from './types';
import { generateFashionImage, editFashionImage, analyzeImage } from './services/geminiService';

// Default placeholder image
const DEFAULT_IMAGE = "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?auto=format&fit=crop&q=80&w=2574&ixlib=rb-4.0.3";

const App: React.FC = () => {
  // --- State ---
  const [activeModule, setActiveModule] = useState<ModuleType>(ModuleType.STUDIO);
  const [mainImage, setMainImage] = useState<string | null>(null); 
  
  // Studio Inputs
  const [productImage, setProductImage] = useState<string | null>(null);
  const [modelRefImage, setModelRefImage] = useState<string | null>(null);
  const [bgRefImage, setBgRefImage] = useState<string | null>(null);
  
  // Generation Config
  const [prompt, setPrompt] = useState<string>('');
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>(AspectRatio.PORTRAIT);
  const [resolution, setResolution] = useState<ImageResolution>(ImageResolution.RES_2K);
  const [imageCount, setImageCount] = useState<number>(1);
  const [modelVersion, setModelVersion] = useState<'2.5' | '3'>('2.5');
  
  // Edit / Retouch State
  const [editPrompt, setEditPrompt] = useState<string>('');
  const [isBrushActive, setIsBrushActive] = useState(false);
  const [brushSize, setBrushSize] = useState(20);
  
  // App Status
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // History / Variations
  const [generatedHistory, setGeneratedHistory] = useState<GeneratedImage[]>([]);

  // Overlays (Campaign Mode)
  const [overlays, setOverlays] = useState<AdOverlay[]>([]);

  // Refs
  const productInputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);
  const bgInputRef = useRef<HTMLInputElement>(null);
  
  // Canvas Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Initialize with a default image if history is empty
  useEffect(() => {
    if (!mainImage && generatedHistory.length === 0) {
        setMainImage(DEFAULT_IMAGE);
    }
  }, []);

  // Sync canvas size with image size
  useEffect(() => {
    const syncCanvas = () => {
      if (imageRef.current && canvasRef.current) {
        canvasRef.current.width = imageRef.current.naturalWidth;
        canvasRef.current.height = imageRef.current.naturalHeight;
        // Also clear canvas on resize/image load
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    };
    
    if (imageRef.current?.complete) {
        syncCanvas();
    }
  }, [mainImage]);

  // --- Handlers ---

  const handleNewSession = () => {
    if (window.confirm("Start a new session? This will clear your current workspace.")) {
      setMainImage(DEFAULT_IMAGE);
      setProductImage(null);
      setModelRefImage(null);
      setBgRefImage(null);
      setPrompt('');
      setEditPrompt('');
      setOverlays([]);
      setError(null);
      setGeneratedHistory([]); 
      setActiveModule(ModuleType.STUDIO);
      clearCanvas();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'product' | 'model' | 'bg') => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64 = reader.result as string;
        
        if (type === 'product') {
          setProductImage(base64);
          setIsAnalyzing(true);
          const analysis = await analyzeImage(base64);
          if (analysis) {
            setPrompt(prev => {
                const prefix = prev ? prev + "\n\n" : "";
                return `${prefix}Product Analysis: ${analysis}`;
            });
          }
          setIsAnalyzing(false);
        } else if (type === 'model') {
          setModelRefImage(base64);
        } else if (type === 'bg') {
          setBgRefImage(base64);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerate = async () => {
    if (!prompt && !productImage) {
        setError("Please enter a prompt or upload a product image.");
        return;
    }
    setIsGenerating(true);
    setError(null);

    try {
        for (let i = 0; i < imageCount; i++) {
            const resultImage = await generateFashionImage(
                prompt, 
                aspectRatio, 
                resolution, 
                {
                    product: productImage || undefined,
                    model: modelRefImage || undefined,
                    background: bgRefImage || undefined
                },
                modelVersion
            );

            const newImageObj = {
                id: Date.now().toString() + i,
                url: resultImage,
                prompt: prompt,
                timestamp: Date.now()
            };

            setGeneratedHistory(prev => [newImageObj, ...prev]);
            setMainImage(resultImage);
            clearCanvas(); // Clear sketches on new generation
        }
    } catch (err: any) {
      setError(err.message || "Generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleEdit = async () => {
     if (!mainImage || !editPrompt) return;
     setIsGenerating(true);
     setError(null);
     
     try {
        let maskBase64 = undefined;
        // Check if canvas has content
        if (canvasRef.current) {
            // Check if user drew anything (simple check: is canvas blank?)
            // We can just send it. If empty transparent, it's fine.
            // Convert canvas to base64
            maskBase64 = canvasRef.current.toDataURL("image/png");
            
            // Optimization: check if empty to avoid sending useless data
            const context = canvasRef.current.getContext('2d');
            const pixelBuffer = new Uint32Array(
                context!.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height).data.buffer
            );
            if (!pixelBuffer.some(color => color !== 0)) {
                maskBase64 = undefined; // Canvas is empty
            }
        }

        const resultImage = await editFashionImage(mainImage, editPrompt, maskBase64);
        const newImageObj = {
            id: Date.now().toString(),
            url: resultImage,
            prompt: `Edit: ${editPrompt}`,
            timestamp: Date.now()
        };
        setGeneratedHistory(prev => [newImageObj, ...prev]);
        setMainImage(resultImage);
        clearCanvas();
     } catch (err: any) {
        setError(err.message || "Edit failed");
     } finally {
        setIsGenerating(false);
     }
  };

  // --- Canvas Drawing Logic ---
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isBrushActive) return;
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Get mouse position relative to canvas
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(236, 72, 153, 0.6)'; // Pink transparent brush
    ctx.lineWidth = brushSize;
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !isBrushActive) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // --- Campaign Overlay Logic ---
  const addOverlay = (type: 'headline' | 'subtext') => {
    setOverlays(prev => [...prev, {
      id: Date.now().toString(),
      text: type === 'headline' ? 'NEW COLLECTION' : 'Shop Now',
      x: 50,
      y: 50,
      type,
      fontFamily: type === 'headline' ? 'Playfair Display' : 'Inter',
      color: '#ffffff'
    }]);
  };

  const updateOverlay = (id: string, updates: Partial<AdOverlay>) => {
    setOverlays(prev => prev.map(o => o.id === id ? { ...o, ...updates } : o));
  };

  const removeOverlay = (id: string) => {
    setOverlays(prev => prev.filter(o => o.id !== id));
  };

  const getAspectRatioCSS = (ar: AspectRatio) => {
    const [w, h] = ar.split(':').map(Number);
    return `${w}/${h}`;
  };

  // --- Render ---

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans overflow-hidden">
      {/* 1. Navigation Sidebar */}
      <Sidebar activeModule={activeModule} setActiveModule={setActiveModule} />

      {/* 2. Left Panel: Generation Inputs (Only visible in STUDIO) */}
      {activeModule === ModuleType.STUDIO && (
          <aside className="w-[340px] bg-zinc-950 border-r border-zinc-800 flex flex-col p-5 z-20 shrink-0 overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
                    <Sparkles size={12} className="text-pink-500"/> Model & Scene
                  </h3>
                  {/* Model Switcher */}
                  <div className="flex bg-zinc-900 rounded-lg p-0.5 border border-zinc-800">
                    <button 
                        onClick={() => setModelVersion('2.5')}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-bold rounded-md transition-all ${modelVersion === '2.5' ? 'bg-zinc-800 text-white shadow-sm ring-1 ring-zinc-700' : 'text-zinc-500 hover:text-zinc-300'}`}
                        title="Gemini 2.5 Flash Image"
                    >
                        <Zap size={10} className={modelVersion === '2.5' ? 'text-yellow-400' : ''}/>
                        <span>Flash 2.5</span>
                    </button>
                    <button 
                        onClick={() => setModelVersion('3')}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-bold rounded-md transition-all ${modelVersion === '3' ? 'bg-gradient-to-r from-pink-600 to-violet-600 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                        title="Gemini 3 Pro Image"
                    >
                        <Sparkles size={10} className={modelVersion === '3' ? 'text-white' : ''}/>
                        <span>Pro 3</span>
                    </button>
                  </div>
              </div>

              {/* Product Upload */}
              <div className="mb-4 p-3 border border-zinc-800 rounded-xl bg-zinc-900/30">
                <div className="flex justify-between items-center mb-2">
                    <label className="text-xs font-semibold text-zinc-300 flex items-center gap-1">
                        <Shirt size={12} /> Main Product <span className="text-pink-500">*</span>
                    </label>
                    {productImage && <button onClick={() => setProductImage(null)} className="text-zinc-600 hover:text-red-500"><X size={12} /></button>}
                </div>
                <div 
                  onClick={() => productInputRef.current?.click()}
                  className={`border border-dashed rounded-lg h-28 flex flex-col items-center justify-center cursor-pointer transition-all relative overflow-hidden group
                    ${productImage ? 'border-zinc-700' : 'border-zinc-700 hover:border-pink-500 hover:bg-zinc-800'}`}
                >
                  {productImage ? (
                    <img src={productImage} className="w-full h-full object-cover opacity-80 group-hover:opacity-40 transition-opacity" />
                  ) : (
                    <>
                       {isAnalyzing ? <Loader2 className="animate-spin text-pink-500 mb-2" size={20} /> : <Upload className="text-zinc-500 mb-2 group-hover:text-pink-500" size={20} />}
                       <span className="text-xs text-zinc-400">{isAnalyzing ? 'Analyzing...' : 'Upload Product'}</span>
                    </>
                  )}
                </div>
                <input type="file" ref={productInputRef} className="hidden" onChange={(e) => handleFileUpload(e, 'product')} />
              </div>

              {/* Optional References */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                 <div onClick={() => modelInputRef.current?.click()} className="cursor-pointer">
                    <label className="text-[10px] text-zinc-400 mb-1 flex justify-between">
                        <span>Model (Opt)</span> {modelRefImage && <X size={10} onClick={(e) => {e.stopPropagation(); setModelRefImage(null);}}/>}
                    </label>
                    <div className="h-16 border border-dashed border-zinc-800 rounded-lg bg-zinc-900/30 flex items-center justify-center overflow-hidden">
                       {modelRefImage ? <img src={modelRefImage} className="w-full h-full object-cover opacity-80" /> : <Plus size={14} className="text-zinc-700"/>}
                    </div>
                    <input type="file" ref={modelInputRef} className="hidden" onChange={(e) => handleFileUpload(e, 'model')} />
                 </div>
                 <div onClick={() => bgInputRef.current?.click()} className="cursor-pointer">
                    <label className="text-[10px] text-zinc-400 mb-1 flex justify-between">
                        <span>Scene (Opt)</span> {bgRefImage && <X size={10} onClick={(e) => {e.stopPropagation(); setBgRefImage(null);}}/>}
                    </label>
                    <div className="h-16 border border-dashed border-zinc-800 rounded-lg bg-zinc-900/30 flex items-center justify-center overflow-hidden">
                       {bgRefImage ? <img src={bgRefImage} className="w-full h-full object-cover opacity-80" /> : <Plus size={14} className="text-zinc-700"/>}
                    </div>
                    <input type="file" ref={bgInputRef} className="hidden" onChange={(e) => handleFileUpload(e, 'bg')} />
                 </div>
              </div>

              {/* Controls */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                 <div>
                    <label className="text-[10px] text-zinc-500 mb-1 block">ASPECT RATIO</label>
                    <select 
                      value={aspectRatio} onChange={(e) => setAspectRatio(e.target.value as AspectRatio)}
                      className="w-full bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs rounded-lg p-2 focus:border-pink-500 outline-none"
                    >
                       {Object.values(AspectRatio).map(ar => <option key={ar} value={ar}>{ar}</option>)}
                    </select>
                 </div>
                 <div>
                    <label className="text-[10px] text-zinc-500 mb-1 block">AMOUNT</label>
                    <div className="flex bg-zinc-900 rounded-lg border border-zinc-800 p-0.5">
                        {[1, 2, 3].map(num => (
                            <button key={num} onClick={() => setImageCount(num)} className={`flex-1 text-xs py-1.5 rounded-md ${imageCount === num ? 'bg-zinc-700 text-white' : 'text-zinc-500'}`}>{num}</button>
                        ))}
                    </div>
                 </div>
              </div>
              
              {modelVersion === '3' && (
                 <div className="mb-4">
                    <label className="text-[10px] text-zinc-500 mb-1 block">RESOLUTION</label>
                    <select value={resolution} onChange={(e) => setResolution(e.target.value as ImageResolution)} className="w-full bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs rounded-lg p-2 outline-none">
                       <option value={ImageResolution.RES_1K}>1K Standard</option>
                       <option value={ImageResolution.RES_2K}>2K High Def</option>
                       <option value={ImageResolution.RES_4K}>4K Ultra</option>
                    </select>
                 </div>
              )}

              <div className="mt-auto">
                 <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2">Prompt</h3>
                 <textarea 
                    value={prompt} onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the styling..."
                    className="w-full h-28 bg-zinc-900/50 border border-zinc-800 rounded-xl p-3 text-sm text-zinc-200 outline-none focus:border-pink-500 resize-none mb-3"
                 />
                 <button
                    onClick={handleGenerate} disabled={isGenerating || (!prompt && !productImage)}
                    className="w-full py-3 bg-gradient-to-r from-pink-600 to-violet-600 hover:opacity-90 text-white rounded-xl font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                 >
                    {isGenerating ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
                    <span>Generate</span>
                 </button>
              </div>
          </aside>
      )}

      {/* 3. Center Panel: Canvas & Campaign */}
      <main className="flex-1 flex flex-col relative min-w-0 bg-[#09090b]">
        {/* Header */}
        <header className="h-16 border-b border-zinc-800 flex items-center justify-between px-6 bg-zinc-950/90 z-10 shrink-0">
           <div className="flex items-center gap-2">
             <span className="font-serif font-bold text-xl text-white">FASHIONGEN</span>
           </div>
           <div className="flex items-center gap-4">
              {error && <div className="text-xs text-red-400 px-3 py-1 bg-red-900/20 rounded-full border border-red-500/20">{error}</div>}
              <button onClick={handleNewSession} className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-lg text-xs hover:bg-zinc-800 text-zinc-300">
                 <Plus size={14} /> New Session
              </button>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 to-violet-600 flex items-center justify-center text-[10px] font-bold">JD</div>
           </div>
        </header>

        {/* Content */}
        <div className="flex-1 relative flex flex-col overflow-hidden">
            {activeModule === ModuleType.CAMPAIGN && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 bg-zinc-900 border border-zinc-800 rounded-lg p-2 flex gap-2 shadow-xl">
                   <button onClick={() => addOverlay('headline')} className="px-3 py-1 bg-zinc-800 text-xs rounded hover:bg-zinc-700">Add Headline</button>
                   <button onClick={() => addOverlay('subtext')} className="px-3 py-1 bg-zinc-800 text-xs rounded hover:bg-zinc-700">Add Subtext</button>
                </div>
            )}

            <div className="flex-1 flex items-center justify-center p-8 overflow-hidden bg-[#0c0c0e]">
               {/* Pattern */}
               <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'linear-gradient(#444 1px, transparent 1px), linear-gradient(90deg, #444 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
               
               {/* Image Wrapper */}
               <div className="relative shadow-2xl transition-all duration-300 group" style={{ aspectRatio: getAspectRatioCSS(aspectRatio), height: 'auto', maxHeight: '100%', maxWidth: '100%' }}>
                  {mainImage ? (
                    <>
                        <img 
                            ref={imageRef}
                            src={mainImage} 
                            className="w-full h-full object-contain bg-zinc-900" 
                            alt="Main Work" 
                            onLoad={() => {
                                // Trigger canvas sync
                                if (imageRef.current && canvasRef.current) {
                                    canvasRef.current.width = imageRef.current.naturalWidth;
                                    canvasRef.current.height = imageRef.current.naturalHeight;
                                }
                            }}
                        />
                        {/* Sketch Canvas Overlay - Only in Studio */}
                        {activeModule === ModuleType.STUDIO && (
                            <canvas
                                ref={canvasRef}
                                className={`absolute inset-0 w-full h-full cursor-crosshair transition-opacity ${isBrushActive ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
                                onMouseDown={startDrawing}
                                onMouseMove={draw}
                                onMouseUp={stopDrawing}
                                onMouseLeave={stopDrawing}
                            />
                        )}
                    </>
                  ) : (
                    <div className="w-full h-full bg-zinc-900/50 border border-zinc-800 flex flex-col items-center justify-center text-zinc-600 aspect-[3/4] max-h-[600px]">
                       <ImagePlus size={48} className="mb-4 opacity-30" />
                       <p className="text-sm">Start a new session</p>
                    </div>
                  )}

                  {/* Campaign Overlays */}
                  {activeModule === ModuleType.CAMPAIGN && overlays.map(overlay => (
                     <div
                        key={overlay.id}
                        className="absolute cursor-move select-none p-2 border border-transparent hover:border-pink-500/50 rounded"
                        style={{ left: `${overlay.x}%`, top: `${overlay.y}%`, transform: 'translate(-50%, -50%)', fontFamily: overlay.fontFamily, color: overlay.color }}
                     >
                        <span className={overlay.type === 'headline' ? 'text-4xl font-bold uppercase' : 'text-lg font-medium'}>
                           {overlay.text}
                        </span>
                        <button onClick={() => removeOverlay(overlay.id)} className="absolute -top-3 -right-3 bg-red-500 text-white p-1 rounded-full opacity-0 hover:opacity-100"><X size={10}/></button>
                     </div>
                  ))}
               </div>
            </div>

            {/* Filmstrip */}
            <div className="h-28 bg-zinc-950 border-t border-zinc-800 shrink-0 flex flex-col">
              <div className="px-4 py-1.5 border-b border-zinc-800/50 flex justify-between items-center">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2"><Layers size={10} /> History</span>
              </div>
              <div className="flex-1 overflow-x-auto p-2 flex gap-2 items-center">
                 {generatedHistory.map((item) => (
                    <button key={item.id} onClick={() => { setMainImage(item.url); setPrompt(item.prompt); }} className={`h-full aspect-[3/4] rounded-md overflow-hidden border ${mainImage === item.url ? 'border-pink-500' : 'border-zinc-800'} relative`}>
                       <img src={item.url} className="w-full h-full object-cover" loading="lazy" />
                    </button>
                 ))}
              </div>
            </div>
        </div>
      </main>

      {/* 4. Right Panel: AI Editor (Studio Mode) */}
      {activeModule === ModuleType.STUDIO && (
          <aside className="w-[300px] bg-zinc-950 border-l border-zinc-800 flex flex-col p-5 z-20 shrink-0">
             <div className="mb-6">
                <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                   <Wand2 size={12} className="text-violet-500"/> AI Editor
                </h3>
                
                {/* Sketch Controls */}
                <div className="bg-zinc-900/50 p-3 rounded-xl border border-zinc-800 mb-4">
                    <label className="text-[10px] text-zinc-400 mb-2 block uppercase flex justify-between">
                        <span>Masking Tool</span>
                        <span className="text-violet-400 text-[10px]">{isBrushActive ? 'Active' : 'Inactive'}</span>
                    </label>
                    <div className="flex gap-2 mb-3">
                        <button 
                           onClick={() => setIsBrushActive(!isBrushActive)}
                           className={`flex-1 py-2 text-xs rounded-lg border transition-all flex items-center justify-center gap-2 ${isBrushActive ? 'bg-violet-600 border-violet-500 text-white' : 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:text-zinc-200'}`}
                        >
                           <Brush size={14} /> {isBrushActive ? 'Draw Mask' : 'Enable Brush'}
                        </button>
                        <button onClick={clearCanvas} className="p-2 bg-zinc-900 border border-zinc-700 rounded-lg text-zinc-400 hover:text-white" title="Clear Mask">
                           <RotateCcw size={14} />
                        </button>
                    </div>
                    {isBrushActive && (
                        <div>
                            <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                <span>Size</span>
                                <span>{brushSize}px</span>
                            </div>
                            <input 
                                type="range" min="5" max="100" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))}
                                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
                            />
                        </div>
                    )}
                </div>

                <p className="text-[10px] text-zinc-500 mb-2">EDIT INSTRUCTIONS</p>
                <textarea 
                   value={editPrompt} onChange={(e) => setEditPrompt(e.target.value)}
                   placeholder="E.g., Change the background to a beach, remove the hat..."
                   className="w-full h-24 bg-zinc-900/50 border border-zinc-800 rounded-xl p-3 text-sm text-zinc-200 outline-none focus:border-violet-500 resize-none mb-3"
                />
                
                <button
                   onClick={handleEdit} disabled={isGenerating || !editPrompt}
                   className="w-full py-3 bg-zinc-100 hover:bg-white text-black rounded-xl font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                >
                   {isGenerating ? <Loader2 className="animate-spin" size={16} /> : <Wand2 size={16} />}
                   <span>Apply Edit</span>
                </button>
             </div>
             
             {/* Info */}
             <div className="mt-auto p-4 bg-zinc-900/30 rounded-xl border border-zinc-800/50">
                 <h4 className="text-xs font-semibold text-zinc-400 mb-1">Pro Tip</h4>
                 <p className="text-[10px] text-zinc-500 leading-relaxed">
                    Use the <strong>Brush Tool</strong> to highlight specific areas you want to change. If no mask is drawn, the AI will attempt to follow your instructions for the whole image.
                 </p>
             </div>
          </aside>
      )}

      {/* Campaign Right Sidebar */}
      {activeModule === ModuleType.CAMPAIGN && (
          <aside className="w-[300px] bg-zinc-950 border-l border-zinc-800 flex flex-col p-5 z-20 shrink-0">
             <div className="mb-6">
                <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-4">Export Options</h3>
                <button className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-medium flex items-center justify-center gap-2 mb-4">
                   <Download size={16} /> Export Final Asset
                </button>
                
                <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-zinc-400">Layers</h4>
                    {overlays.map(layer => (
                        <div key={layer.id} className="p-2 bg-zinc-900 border border-zinc-800 rounded text-xs flex justify-between items-center">
                            <span className="truncate max-w-[150px]">{layer.text}</span>
                            <button onClick={() => removeOverlay(layer.id)}><Trash2 size={12} className="text-zinc-600 hover:text-red-500"/></button>
                        </div>
                    ))}
                    {overlays.length === 0 && <p className="text-[10px] text-zinc-600 italic">No text layers added.</p>}
                </div>
             </div>
          </aside>
      )}

      {/* Chat Assistant */}
      <ChatAssistant />
    </div>
  );
};

export default App;