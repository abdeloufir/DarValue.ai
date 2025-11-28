'use client'

import { useState, useEffect } from 'react'
import { predictProperty, getCities, getNeighborhoods, PropertyInput, PropertyPrediction } from '@/lib/api'

interface PropertyFormProps {
  onPrediction: (prediction: PropertyPrediction) => void
  setLoading: (loading: boolean) => void
}

export default function PropertyForm({ onPrediction, setLoading }: PropertyFormProps) {
  const [formData, setFormData] = useState<PropertyInput>({
    price: 0,
    surface_m2: 0,
    rooms: 1,
    bathrooms: 1,
    city: 'Casablanca',
    neighborhood: '',
    property_type: 'apartment',
    condition: 'Standard',
    furnishing: 'Unknown',
  })

  const [cities, setCities] = useState<string[]>([])
  const [neighborhoods, setNeighborhoods] = useState<string[]>([])
  const [error, setError] = useState<string>('')

  useEffect(() => {
    // Fetch cities on mount
    const fetchCities = async () => {
      try {
        const data = await getCities()
        setCities(data)
      } catch (err) {
        console.error('Failed to fetch cities')
      }
    }
    fetchCities()
  }, [])

  useEffect(() => {
    // Fetch neighborhoods when city changes
    const fetchNeighborhoods = async () => {
      try {
        const data = await getNeighborhoods(formData.city)
        setNeighborhoods(data)
        setFormData(prev => ({ ...prev, neighborhood: data[0] || '' }))
      } catch (err) {
        console.error('Failed to fetch neighborhoods')
      }
    }
    if (formData.city) {
      fetchNeighborhoods()
    }
  }, [formData.city])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: ['price', 'surface_m2', 'rooms', 'bathrooms'].includes(name) 
        ? parseFloat(value) 
        : value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      if (!formData.price || !formData.surface_m2) {
        throw new Error('Please fill in all required fields')
      }

      const prediction = await predictProperty(formData)
      onPrediction(prediction)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Price */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Price (MAD) *
        </label>
        <input
          type="number"
          name="price"
          value={formData.price}
          onChange={handleChange}
          className="input-field"
          placeholder="e.g., 2500000"
          required
        />
      </div>

      {/* Surface */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Surface (m²) *
        </label>
        <input
          type="number"
          name="surface_m2"
          value={formData.surface_m2}
          onChange={handleChange}
          className="input-field"
          placeholder="e.g., 120"
          required
        />
      </div>

      {/* Rooms */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Rooms
        </label>
        <input
          type="number"
          name="rooms"
          value={formData.rooms}
          onChange={handleChange}
          className="input-field"
          min="1"
        />
      </div>

      {/* Bathrooms */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Bathrooms
        </label>
        <input
          type="number"
          name="bathrooms"
          value={formData.bathrooms}
          onChange={handleChange}
          className="input-field"
          min="1"
        />
      </div>

      {/* City */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          City
        </label>
        <select
          name="city"
          value={formData.city}
          onChange={handleChange}
          className="input-field"
        >
          {cities.map(city => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>
      </div>

      {/* Neighborhood */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Neighborhood
        </label>
        <select
          name="neighborhood"
          value={formData.neighborhood}
          onChange={handleChange}
          className="input-field"
        >
          {neighborhoods.map(neighborhood => (
            <option key={neighborhood} value={neighborhood}>{neighborhood}</option>
          ))}
        </select>
      </div>

      {/* Property Type */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Property Type
        </label>
        <select
          name="property_type"
          value={formData.property_type}
          onChange={handleChange}
          className="input-field"
        >
          <option value="apartment">Apartment</option>
          <option value="villa">Villa</option>
          <option value="house">House</option>
          <option value="land">Land</option>
          <option value="commercial">Commercial</option>
        </select>
      </div>

      {/* Condition */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Condition
        </label>
        <select
          name="condition"
          value={formData.condition}
          onChange={handleChange}
          className="input-field"
        >
          <option value="New">New</option>
          <option value="Renovated">Renovated</option>
          <option value="Standard">Standard</option>
          <option value="Old">Old</option>
        </select>
      </div>

      {/* Furnishing */}
      <div>
        <label className="block text-gray-700 font-semibold mb-2">
          Furnishing
        </label>
        <select
          name="furnishing"
          value={formData.furnishing}
          onChange={handleChange}
          className="input-field"
        >
          <option value="Furnished">Furnished</option>
          <option value="Unfurnished">Unfurnished</option>
          <option value="Unknown">Unknown</option>
        </select>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        className="btn-primary w-full"
      >
        Get Property Prediction
      </button>
    </form>
  )
}
