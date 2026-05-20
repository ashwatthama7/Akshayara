import { useState } from 'react'
import './App.css'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Header from './components/Header'
import Home from './pages/Home'
import AboutUs from './pages/AboutUs'
function App() {
   return (
    <BrowserRouter>
    <Header/>
    <Routes>
    <Route path="/" element={<Home />} />
    <Route path="/home" element={<Home />} />
    <Route path="/aboutus" element={<AboutUs />} />
    </Routes>
    
    </BrowserRouter>
  )
}

export default App