import React, { useState, useEffect } from 'react';
import { jsPDF } from 'jspdf';

export default function Home() {
  const [image, setImage] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [convertedText, setConvertedText] = useState('');
  const [showResult, setShowResult] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState(null);
  const [isMultiLine, setIsMultiLine] = useState(false);
  const [showLineSeparator, setShowLineSeparator] = useState(false);

  // if backend runs on this port
  const API_URL = 'http://localhost:3001';

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB.');
        return;
      }
      setImage(file);
      setPreviewURL(prev => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setShowResult(false);
      setConvertedText('');
      setError(null);
      
      // If multi-line mode bhaye separate garne window dekhaune
      if (isMultiLine) {
        setShowLineSeparator(true);
      }
    } else {
      alert('Please upload a valid image file.');
    }
  };

  const handleLinesExtracted = async (lineImages) => {
    setShowLineSeparator(false);
    setIsProcessing(true);
    setError(null);

    try {
      console.log(`Processing ${lineImages.length} lines...`);
      
      const allTexts = [];
      const allConfidences = [];

      // Process each line
      for (let i = 0; i < lineImages.length; i++) {
        const formData = new FormData();
        formData.append('file', lineImages[i].blob, `line_${i}.png`);

        console.log(`Processing line ${i + 1}/${lineImages.length}...`);

        const response = await fetch(`${API_URL}/api/predict`, {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || `Failed to process line ${i + 1}`);
        }

        if (data.success) {
          allTexts.push(data.text);
          allConfidences.push(data.confidence);
          console.log(`Line ${i + 1}: ${data.text} (${(data.confidence * 100).toFixed(1)}%)`);
        }
      }

      // Combine all lines
      const fullText = allTexts.join('\n');
      const avgConfidence = allConfidences.reduce((a, b) => a + b, 0) / allConfidences.length;

      setConvertedText(fullText);
      setConfidence(avgConfidence);
      setShowResult(true);

      console.log('Multi-line processing complete!');

    } catch (error) {
      console.error('Multi-line OCR Error:', error);
      setError(error.message || 'Failed to process lines');
      alert(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Updated TTS function using Web Speech API for Nepali
  const speakText = () => {
    if (!convertedText) return;

    // Stop speaking if already speaking
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    // Stop any ongoing speech first
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(convertedText);
    
    // Set language to Nepali
    utterance.lang = 'ne-NP';
    
    // Try to find a Nepali voice
    const voices = window.speechSynthesis.getVoices();
    const nepaliVoice = voices.find(voice => 
      voice.lang.startsWith('ne') || voice.lang.includes('NP')
    );
    
    if (nepaliVoice) {
      utterance.voice = nepaliVoice;
    } else {
      // Fallback to Hindi or English-India voice if Nepali not available
      const fallbackVoice = voices.find(v => 
        v.lang.includes('hi') || v.lang.includes('en-IN')
      ) || voices[0];
      if (fallbackVoice) {
        utterance.voice = fallbackVoice;
      }
    }

    // Set speech parameters
    utterance.rate = 0.7; // Slightly slower for better clarity
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Event handlers
    utterance.onstart = () => {
      setIsSpeaking(true);
      console.log('Started speaking');
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      console.log('Finished speaking');
    };

    utterance.onerror = (event) => {
      console.error('Speech error:', event);
      setIsSpeaking(false);
      alert('सुन्न सकेन। कृपया फेरि प्रयास गर्नुहोस्।');
    };

    // Speak the text
    window.speechSynthesis.speak(utterance);
  };

  // Load voices when component mounts
  useEffect(() => {
    // Load voices
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      console.log('Available voices:', voices.length);
      const nepaliVoices = voices.filter(v => v.lang.startsWith('ne'));
      console.log('Nepali voices:', nepaliVoices);
    };

    loadVoices();
    
    // Chrome loads voices asynchronously
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    // Cleanup: stop speaking when component unmounts
    return () => {
      window.speechSynthesis.cancel();
    };
  }, []);

  const handleConvertToText = async () => {
    if (!image) {
      alert('Please upload an image first.');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Create FormData to send image
      const formData = new FormData();
      formData.append('file', image);

      console.log('Sending image to backend...');

      // Call backend API
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to process image');
      }

      if (data.success) {
        console.log('OCR Success:', data);
        setConvertedText(data.text);
        setConfidence(data.confidence);
        setShowResult(true);
      } else {
        throw new Error('Prediction failed');
      }

    } catch (error) {
      console.error('OCR Error:', error);
      setError(error.message || 'Failed to process image. Please try again.');
      alert(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadText = () => {
    const blob = new Blob([convertedText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'converted-text.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSaveAsPDF = () => {
    const doc = new jsPDF();
    
    // Add Nepali font support for jsPDF (basic support)
    // Note: For proper Nepali rendering in PDF, you might need custom font
    const lines = doc.splitTextToSize(convertedText, 180);
    doc.text(lines, 10, 10);
    doc.save('converted-text.pdf');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 pt-16">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md overflow-hidden">
          <div className="bg-blue-600 py-6 px-8">
            <h1 className="text-2xl md:text-3xl font-bold text-white text-center">
              Image to Text Converter
            </h1>
            <p className="text-blue-100 text-center mt-2">
              Upload an image and extract Nepali handwritten text with AI
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="mx-8 mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          <div className="p-6 md:p-8">
            <div className="flex flex-col md:flex-row gap-8">
              {/* Left side - Image Upload */}
              <div className="flex-1">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors duration-300">
                  <div className="flex flex-col items-center justify-center space-y-4">
                    <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <div>
                      <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-300">
                        Choose an Image
                        <input 
                          type="file" 
                          accept=".jpg,.jpeg,.png,.bmp,.tiff,.webp" 
                          onChange={handleImageUpload} 
                          className="hidden" 
                        />
                      </label>
                      <p className="text-gray-500 text-sm mt-2">
                        JPG, PNG, BMP, TIFF, WebP (Max 5MB)
                      </p>
                      
                      {/* Multi-line toggle */}
                      <div className="mt-4 flex items-center justify-center">
                        <label className="flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={isMultiLine}
                            onChange={(e) => setIsMultiLine(e.target.checked)}
                            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                          />
                          <span className="ml-2 text-sm text-gray-700">Multi-line text (manual separation)</span>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>

                {previewURL && (
                  <div className="mt-6">
                    <h3 className="text-lg font-medium text-gray-700 mb-2">Image Preview</h3>
                    <div className="relative group">
                      <img 
                        src={previewURL} 
                        alt="Preview" 
                        className="w-full h-auto rounded-lg shadow-sm border border-gray-200" 
                      />
                      <div className="absolute inset-0 bg-opacity-0 group-hover:bg-opacity-10 transition-all duration-300 rounded-lg" />
                    </div>
                    <button 
                      onClick={handleConvertToText} 
                      disabled={isProcessing} 
                      className={`mt-4 w-full py-3 px-4 rounded-lg font-medium text-white transition-colors duration-300 ${
                        isProcessing 
                          ? 'bg-blue-400 cursor-not-allowed' 
                          : 'bg-blue-600 hover:bg-blue-700'
                      }`}
                    >
                      {isProcessing ? (
                        <span className="flex items-center justify-center">
                          <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Processing with AI...
                        </span>
                      ) : (
                        'Convert to Text'
                      )}
                    </button>
                  </div>
                )}
              </div>

              {/* Right side - Results */}
              <div className="flex-1">
                {showResult ? (
                  <div className="h-full flex flex-col">
                    <div className="flex justify-between items-center mb-2">
                      <div>
                        <h3 className="text-lg font-medium text-gray-700">Extracted Text</h3>
                        {confidence > 0 && (
                          
                          <p className="text-sm text-gray-500">
                            Confidence: {(confidence * 100).toFixed(1)}%
                          </p>
                          
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        <button 
                          onClick={speakText} 
                          title={isSpeaking ? "Stop speaking" : "Listen to text"} 
                          className={`text-sm flex items-center px-3 py-1.5 rounded-lg transition-colors ${
                            isSpeaking 
                              ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          }`}
                        >
                          {isSpeaking ? (
                            <>
                              <svg className="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                              </svg>
                              Stop
                            </>
                          ) : (
                            <>
                              <svg className="w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9 4.804A7.968 7.968 0 015 4a8 8 0 100 16 7.968 7.968 0 014-.804V4.804zM11 6v8l4 4V2l-4 4z" />
                              </svg>
                              Listen
                            </>
                          )}
                        </button>

                        <button 
                          onClick={() => {
                            navigator.clipboard.writeText(convertedText);
                            alert('Text copied to clipboard!');
                          }} 
                          className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
                        >
                          <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                          </svg>
                          Copy
                        </button>
                      </div>
                    </div>
                    <div className="flex-1 bg-gray-50 p-4 rounded-lg border border-gray-200 overflow-auto">
                      <textarea 
                        value={convertedText} 
                        onChange={(e) => setConvertedText(e.target.value)} 
                        className="w-full h-full p-2 bg-transparent resize-none focus:outline-none text-gray-700" 
                        rows="12"
                        placeholder="Extracted text will appear here..."
                      />
                    </div>
                    <div className="mt-4 flex space-x-3">
                      <button 
                        onClick={handleSaveAsPDF} 
                        className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors duration-300"
                      >
                        Save as PDF
                      </button>
                      <button 
                        onClick={handleDownloadText} 
                        className="flex-1 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg transition-colors duration-300"
                      >
                        Download Text
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <div className="text-center p-6">
                      <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <h3 className="mt-2 text-sm font-medium text-gray-900">
                        {previewURL ? 'Click "Convert to Text"' : 'Upload an image to get started'}
                      </h3>
                      <p className="mt-1 text-sm text-gray-500">
                        {previewURL ? 'AI will extract Nepali handwritten text' : 'Supports handwritten Nepali text recognition'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-gray-50 px-8 py-4 border-t border-gray-200">
            <p className="text-center text-sm text-gray-500">
              Powered by AI • Need help? <a href="/aboutus" className="text-blue-600 hover:text-blue-800">Learn more</a>
            </p>
          </div>
        </div>

        {/* Manual Line Separator Modal */}
        {showLineSeparator && previewURL && (
          <ManualLineSeparator
            imageUrl={previewURL}
            onLinesExtracted={handleLinesExtracted}
            onCancel={() => setShowLineSeparator(false)}
          />
        )}
      </div>
    </div>
  );
}

// Manual Line Separator Component
function ManualLineSeparator({ imageUrl, onLinesExtracted, onCancel }) {
  const canvasRef = React.useRef(null);
  const imageRef = React.useRef(null);
  const [lines, setLines] = React.useState([]);
  const [isProcessing, setIsProcessing] = React.useState(false);

  React.useEffect(() => {
    if (imageUrl && canvasRef.current) {
      loadImage();
    }
  }, [imageUrl]);

  const loadImage = () => {
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      const canvas = canvasRef.current;
      canvas.width = img.width;
      canvas.height = img.height;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
    };
    img.src = imageUrl;
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const y = (e.clientY - rect.top) * scaleY;
    
    const newLines = [...lines, y].sort((a, b) => a - b);
    setLines(newLines);
    redrawCanvas(newLines);
  };

  const redrawCanvas = (linePositions) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0);
    }
    
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    
    linePositions.forEach((y, index) => {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
      
      ctx.fillStyle = '#ff0000';
      ctx.font = '20px Arial';
      ctx.fillText(`Line ${index + 1}`, 10, y - 10);
    });
  };

  const handleRemoveLastLine = () => {
    const newLines = lines.slice(0, -1);
    setLines(newLines);
    redrawCanvas(newLines);
  };

  const handleClearAll = () => {
    setLines([]);
    redrawCanvas([]);
  };

  const extractLineImages = async () => {
    if (lines.length < 2) {
      alert('Please add at least 2 lines to create 1 text line');
      return;
    }

    setIsProcessing(true);

    try {
      const canvas = canvasRef.current;
      const lineImages = [];

      for (let i = 0; i < lines.length - 1; i++) {
        const startY = Math.floor(lines[i]);
        const endY = Math.floor(lines[i + 1]);
        const height = endY - startY;

        const lineCanvas = document.createElement('canvas');
        lineCanvas.width = canvas.width;
        lineCanvas.height = height;
        
        const lineCtx = lineCanvas.getContext('2d');
        lineCtx.drawImage(
          canvas,
          0, startY, canvas.width, height,
          0, 0, canvas.width, height
        );

        const blob = await new Promise(resolve => {
          lineCanvas.toBlob(resolve, 'image/png');
        });

        lineImages.push({ blob, index: i, startY, endY });
      }

      onLinesExtracted(lineImages);
      
    } catch (error) {
      console.error('Error extracting lines:', error);
      alert('Failed to extract lines. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Separate Text Lines</h2>
          <p className="text-sm text-gray-600 mt-1">
            Click on the image to add separation lines between text lines
          </p>
        </div>

        <div className="p-4 bg-gray-100">
          <div className="bg-white border-2 border-gray-300 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className="w-full cursor-crosshair"
              style={{ maxHeight: '60vh' }}
            />
          </div>
        </div>

        <div className="px-4 py-3 bg-blue-50 border-y border-blue-200">
          <div className="text-sm text-blue-800">
            <strong>Instructions:</strong>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>Click ABOVE the first line of text</li>
              <li>Click BELOW each line of text</li>
              <li>Need at least 2 lines to create 1 text line</li>
            </ul>
          </div>
        </div>

        <div className="px-4 py-2 bg-gray-50">
          <p className="text-sm text-gray-700">
            Lines marked: <strong>{lines.length}</strong>
            {lines.length >= 2 && (
              <span className="ml-2 text-green-600">
                → Will extract {lines.length - 1} text line(s)
              </span>
            )}
          </p>
        </div>

        <div className="p-4 border-t border-gray-200 flex flex-wrap gap-3">
          <button
            onClick={handleRemoveLastLine}
            disabled={lines.length === 0 || isProcessing}
            className="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Remove Last Line
          </button>

          <button
            onClick={handleClearAll}
            disabled={lines.length === 0 || isProcessing}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Clear All
          </button>

          <div className="flex-grow"></div>

          <button
            onClick={onCancel}
            disabled={isProcessing}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg disabled:cursor-not-allowed transition-colors"
          >
            Cancel
          </button>

          <button
            onClick={extractLineImages}
            disabled={lines.length < 2 || isProcessing}
            className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {isProcessing ? 'Processing...' : `Extract & Process ${lines.length >= 2 ? lines.length - 1 : 0} Line(s)`}
          </button>
        </div>
      </div>
    </div>
  );
}