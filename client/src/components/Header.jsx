import { Link } from "react-router-dom";

export default function Header() {
  return (
    <header className="bg-white shadow-md fixed top-0 left-0 w-full z-10">
      <nav className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
        <div className="space-x-4 flex items-center">
          <Link
            to="/"
            className="text-xl font-bold text-gray-800 hover:text-blue-600"
          >
          Akshayara
          </Link>
        </div>
          <Link
            to="/aboutus"
            className="text-gray-700 hover:text-blue-600 font-medium"
          >
            About Us
          </Link>
        
      </nav>
    </header>
  );
}
