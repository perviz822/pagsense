
import { Link, useNavigate } from "react-router-dom";
import { AuthContext } from "./AuthContext";
import { useContext } from "react";
export default function Header() {
  const navigate = useNavigate();
    const { setUser } = useContext(AuthContext);

  const handleLogout = () => {
     setUser(false);
     localStorage.removeItem("user"); // simulate logout
     navigate("/login");
  };

  return (
  <header className="bg-white text-black px-6 py-4 flex justify-between items-center border-b border-gray-200">
  <div className="text-xl font-medium tracking-wide">page  <span className="font-bold text-2xl">sense</span></div>
  <nav className="flex gap-8 text-base font-medium">
    <Link
      to="/test"
      className="text-gray-700 hover:text-white hover:bg-black transition-colors duration-400 px-2 py-1 rounded-xl"
    >
      Take a Test
    </Link>
    <Link
      to="/gallery"
      className="text-gray-700 hover:text-white hover:bg-black transition-colors duration-400 px-2 py-1 rounded-xl"
    >
      My Word Lists
    </Link>
    <button
      onClick={handleLogout}
      className="text-gray-500 hover:text-white hover:bg-black transition-colors duration-400 px-2 py-1 rounded-xl"
    >
      Logout
    </button>
  </nav>
</header>

  );
}
