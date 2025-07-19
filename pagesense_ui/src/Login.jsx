import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { useNavigate,Link } from "react-router-dom";
function Login({setUser}) {
  const [email,setEmail] = useState("")
  const [password,setPassword] = useState("")
  const navigate = useNavigate();

   const handleSubmit = (e) => {
    e.preventDefault();

    if (email && password) {
      setUser(true); // simulate login
      navigate("/"); // redirect to homepage
    } else {
      alert("Please enter email and password");
    }
  }

  return (
      <form
      onSubmit={handleSubmit}
      className="max-w-sm mx-auto mt-20 p-6 bg-white rounded-xl shadow-md space-y-4"
    >
      <h2 className="text-2xl font-bold text-left">Login</h2>

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
        className="w-full bg-black text-white py-2 px-4 rounded-md hover:bg-blue-900 transition duration-200"

      >
        Sign In
      </button>
      Not a member ? <Link  to ="/register"> Register </Link > 
    </form>

     
  )
}

export default Login
