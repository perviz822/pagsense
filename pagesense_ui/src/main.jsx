import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import App from "./App.jsx";
import Gallery from "./Gallery.jsx";
import Login from "./Login.jsx";
import Register from "./Register.jsx";
import Test from "./Test.jsx"; // add this
import ProtectedRoutes from "./ProtectedRoutes.jsx";
import "./index.css";


function RootApp() {
  const [user, setUser] = useState(false);

  const router = createBrowserRouter([
    {
      path: "/",
      element: <ProtectedRoutes user={user} />, // secure route
      children: [
        {
          index: true,
          element: <App setUser={setUser} />,
        },
        {
          path: "test",
          element: <Test />,
        },
        {
          path: "gallery",
          element: <Gallery />,
        },
      ],
    },
    {
      path: "/login",
      element: <Login setUser={setUser} />,
    },
    {
      path: "/register",
      element: <Register setUser={setUser} />,
    },
  ]);

  return <RouterProvider router={router} />;
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RootApp />
  </StrictMode>
);
