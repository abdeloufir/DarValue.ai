import { PropertyPrediction } from '@/lib/api'

interface PredictionResultsProps {
  prediction: PropertyPrediction
}

export default function PredictionResults({ prediction }: PredictionResultsProps) {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('fr-MA', {
      style: 'currency',
      currency: 'MAD',
      maximumFractionDigits: 0,
    }).format(value)
  }

  const getRecommendationColor = (action: string) => {
    switch (action.toUpperCase()) {
      case 'BUY':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'SELL':
        return 'bg-red-100 text-red-800 border-red-300'
      case 'HOLD':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const getDeviationColor = (deviation: number) => {
    if (deviation < -0.2) return 'text-green-600'
    if (deviation > 0.2) return 'text-red-600'
    return 'text-gray-600'
  }

  return (
    <div className="space-y-4">
      {/* Data Quality Warning */}
      {prediction.data_quality === 'DATA_QUALITY_ISSUE' && (
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-800 px-4 py-3 rounded">
          ⚠️ Data Quality Warning: This property has suspicious pricing data. Predictions may be unreliable.
        </div>
      )}

      {/* Predicted Value */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Predicted Property Value</h3>
        <p className="text-4xl font-bold text-blue-600">
          {formatCurrency(prediction.predicted_value)}
        </p>
        <p className="text-gray-600 mt-2">
          Price per m²: <span className="font-semibold">{formatCurrency(prediction.predicted_price_per_m2)}</span>
        </p>
      </div>

      {/* Valuation Status */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Market Valuation</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-gray-600">Market Price</p>
            <p className="text-xl font-bold">{formatCurrency(prediction.valuation.market_price)}</p>
          </div>
          <div>
            <p className="text-gray-600">Deviation</p>
            <p className={`text-xl font-bold ${getDeviationColor(prediction.valuation.price_deviation)}`}>
              {(prediction.valuation.price_deviation * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <div className="mt-4 p-3 bg-gray-50 rounded">
          <p className="text-sm font-semibold text-gray-800">Status: {prediction.valuation.status}</p>
        </div>
      </div>

      {/* 3-Year Appreciation */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">3-Year Appreciation Forecast</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-gray-600 text-sm">Annual Rate</p>
            <p className="text-2xl font-bold text-green-600">{(prediction.appreciation_3_years.annual_rate * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-gray-600 text-sm">Forecast Price</p>
            <p className="text-lg font-bold">{formatCurrency(prediction.appreciation_3_years.forecast_price)}</p>
          </div>
          <div>
            <p className="text-gray-600 text-sm">Total Gain</p>
            <p className="text-lg font-bold text-green-600">{formatCurrency(prediction.appreciation_3_years.total_appreciation)}</p>
          </div>
        </div>
      </div>

      {/* Rental Yield */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Rental Yield Analysis</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-gray-600 text-sm">Gross Yield</p>
            <p className="text-2xl font-bold text-blue-600">{(prediction.rental_yield.gross_yield).toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-gray-600 text-sm">Net Yield</p>
            <p className="text-2xl font-bold text-blue-600">{(prediction.rental_yield.net_yield).toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-gray-600 text-sm">Monthly Rental</p>
            <p className="text-lg font-bold">{formatCurrency(prediction.rental_yield.monthly_rental)}</p>
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className={`card border-2 ${getRecommendationColor(prediction.recommendation.action)}`}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold">Investment Recommendation</h3>
          <span className="text-3xl font-bold">{prediction.recommendation.action}</span>
        </div>
        <p className="text-sm mb-3">{prediction.recommendation.reasoning}</p>
        <div className="flex items-center justify-between pt-3 border-t">
          <span className="text-gray-600">Confidence Score</span>
          <div className="w-32 bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full"
              style={{ width: `${prediction.recommendation.confidence}%` }}
            ></div>
          </div>
          <span className="font-bold">{prediction.recommendation.confidence}%</span>
        </div>
      </div>
    </div>
  )
}
