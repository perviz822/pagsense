import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { Link } from 'react-router-dom'

function Register() {
  const [email,setEmail] = useState("")
  const [password,setPassword] = useState("")

   const handleSubmit = (e) => {
    e.preventDefault()
    console.log("Email:", email)
    console.log("Password:", password)
  }

  return (
      <form
      onSubmit={handleSubmit}
      className="max-w-sm mx-auto mt-20 p-6 bg-white rounded-xl shadow-md space-y-4"
    >
      <h2 className="text-2xl font-bold text-left">Register</h2>

      <div>
        <label htmlFor="email" className="block text-sm font-medium mb-1 text-left">
          Email
        </label>
        <input
          id="email"
          type="email"
          className="w-full border border-gray-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 focus:outline-none "
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          placeholder='Enter your email'
        />
      </div>

      <div>
        <label htmlFor="password" className="block text-sm font-medium mb-1 text-left">
          Password
        </label>
        <input
          id="password"
          type="password"
          className="w-full border border-gray-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 focus:outline-none "
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder= "Enter your password"
          required
        />
      </div>

      <button
        type="submit"
        className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-900 transition duration-200"

      >
        Sign Up
      </button>
       Already a member? <Link  to ="/login"> Login </Link > 
    </form>

     
  )
}

export default Register
