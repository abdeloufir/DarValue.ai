'use client'

import { useState } from 'react'
import PropertyForm from '@/components/PropertyForm'
import PredictionResults from '@/components/PredictionResults'
import { PropertyPrediction } from '@/lib/api'

export default function Home() {
  const [prediction, setPrediction] = useState<PropertyPrediction | null>(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-blue-600">DarValue.ai</h1>
          <p className="text-gray-600 mt-2">AI-Powered Property Valuation for Morocco</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Form Section */}
          <div className="card">
            <h2 className="text-2xl font-bold mb-6 text-gray-800">Property Details</h2>
            <PropertyForm 
              onPrediction={setPrediction}
              setLoading={setLoading}
            />
          </div>

          {/* Results Section */}
          <div>
            {loading && (
              <div className="card flex items-center justify-center h-96">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Analyzing property...</p>
                </div>
              </div>
            )}
            {prediction && !loading && (
              <PredictionResults prediction={prediction} />
            )}
            {!prediction && !loading && (
              <div className="card flex items-center justify-center h-96">
                <p className="text-gray-500 text-center">
                  Fill in the property details to get predictions
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
