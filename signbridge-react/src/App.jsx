
import { useRef, useState, useEffect } from 'react'
import './App.css'

function App() {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(null)
  const requestRef = useRef(false)
  const historyRef = useRef([])

  const [cameraActive, setCameraActive] = useState(false)
  const [status, setStatus] = useState('Camera is currently off')
  const [prediction, setPrediction] = useState('')
  const [apiStatus, setApiStatus] = useState('Waiting for camera')

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true
      })

      streamRef.current = stream
      videoRef.current.srcObject = stream

      await videoRef.current.play()

      setCameraActive(true)
      setStatus('Camera is active')
      setApiStatus('Connecting to SignBridge...')
      setPrediction('')
      historyRef.current = []

    } catch (error) {
      console.error('Camera error:', error)
      setStatus('Camera access denied or unavailable')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setCameraActive(false)
    setStatus('Camera is currently off')
    setApiStatus('Waiting for camera')
    setPrediction('')
    historyRef.current = []
  }

  // Send camera frames to Python backend
  useEffect(() => {
    if (!cameraActive) return

    const interval = setInterval(async () => {
      if (requestRef.current) return

      const video = videoRef.current
      const canvas = canvasRef.current

      if (!video || !canvas || video.videoWidth === 0) return

      requestRef.current = true

      try {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        const context = canvas.getContext('2d')
        context.drawImage(video, 0, 0, canvas.width, canvas.height)

        const blob = await new Promise(resolve => {
          canvas.toBlob(resolve, 'image/jpeg', 0.8)
        })

        if (!blob) throw new Error('Could not capture camera frame')

        const formData = new FormData()
        formData.append('file', blob, 'sign.jpg')

        const response = await fetch(
          'http://127.0.0.1:8000/predict',
          {
            method: 'POST',
            body: formData
          }
        )

        if (!response.ok) {
          throw new Error('Prediction API request failed')
        }

        const data = await response.json()

        if (data.prediction) {
          historyRef.current.push(data.prediction)

          if (historyRef.current.length > 5) {
            historyRef.current.shift()
          }

          const counts = {}

          historyRef.current.forEach(letter => {
            counts[letter] = (counts[letter] || 0) + 1
          })

          const stablePrediction = Object.keys(counts).reduce(
            (a, b) => counts[a] >= counts[b] ? a : b
          )

          setPrediction(stablePrediction)
          setApiStatus('Sign recognition is active')

        } else {
          historyRef.current = []
          setPrediction('')
          setApiStatus('Show your hand to the camera')
        }

      } catch (error) {
        console.error('Prediction error:', error)
        setApiStatus('Backend connection error')
      } finally {
        requestRef.current = false
      }

    }, 500)

    return () => clearInterval(interval)

  }, [cameraActive])

  // Stop camera when component is removed
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>SignBridge</h1>
        <p>Breaking barriers, one sign at a time.</p>
      </header>

      <main className="dashboard">
        <section className="camera-section">
          <h2>Sign Recognition</h2>

          <div className={`camera-box ${cameraActive ? 'active' : ''}`}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
            />

            {!cameraActive && (
              <div className="camera-placeholder">
                <div className="camera-icon">📷</div>
                <h3>Ready to Recognize</h3>
                <p>Position your hand in front of the camera</p>
                <span>Camera is currently off</span>
              </div>
            )}

            {cameraActive && (
              <div className="camera-live-label">
                <span className="live-dot"></span>
                LIVE
              </div>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: 'none' }} />

          <button onClick={cameraActive ? stopCamera : startCamera}>
            {cameraActive ? 'Stop Camera' : 'Start Camera'}
          </button>

          <p className="language">{status}</p>
          <p className="language">{apiStatus}</p>
        </section>

        <section className="result-section">
          <h2>Translation</h2>

          <div className="result-box">
            {prediction ? (
              <h1>{prediction}</h1>
            ) : (
              <p>Your recognized sign will appear here.</p>
            )}
          </div>

          <p className="language">
            Language: Indian Sign Language (ISL)
          </p>
        </section>
      </main>
    </div>
  )
}

export default App
