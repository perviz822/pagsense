import { Outlet, Navigate } from "react-router-dom";


function ProtectedRoutes({ user }) {
  return user ? <Outlet /> : <Navigate to="/login" />;
}

export default ProtectedRoutes;
