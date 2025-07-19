
import { Link, useNavigate } from "react-router-dom";

export default function Header({ setUser }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    setUser(false); // simulate logout
    navigate("/login");
  };

  return (
  <header className="bg-white text-black px-6 py-4 flex justify-between items-center border-b border-gray-200">
  <div className="text-xl font-medium tracking-wide">Word App</div>
  <nav className="flex gap-8 text-base font-medium">
    <Link
      to="/test"
      className="text-gray-700 hover:text-black transition-colors duration-200"
    >
      Take a Test
    </Link>
    <Link
      to="/gallery"
      className="text-gray-700 hover:text-black transition-colors duration-200"
    >
      My Word Lists
    </Link>
    <button
      onClick={handleLogout}
      className="text-gray-500 hover:text-black transition-colors duration-200"
    >
      Logout
    </button>
  </nav>
</header>

  );
}
