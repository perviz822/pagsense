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
import { AuthProvider } from "./AuthContext.jsx";
import BookContent from "./BookContent.jsx";
import UploadBook from "./UploadBook.jsx";


function RootApp() {

  const router = createBrowserRouter([
    {
      path: "/",
      element: 
          <ProtectedRoutes/>,
               
      children: [
        {
          index: true,
          element: <App  />,
        },
        {
          path: "test",
          element: <Test />,
        },
        {
          path: "gallery",
          element: <Gallery />,
        },
          {
      path: "/gallery/:id",
      element: <BookContent  />,
    },

      {
      path: "/upload_book",
      element: <UploadBook  />,
    },
      ],
    },
    {
      path: "/login",
      element: <Login  />,
    },
    {
      path: "/register",
      element: <Register  />,
    },

   
  ]);

  return <RouterProvider router={router} />;
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
   <AuthProvider>
     <RootApp />
   </AuthProvider>
  </StrictMode>
);
