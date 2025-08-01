import { Outlet, Navigate } from "react-router-dom";
import { AuthContext } from "./AuthContext";
import { useContext } from "react";
import Header from "./Header";


function ProtectedRoutes() {
  const {user} = useContext(AuthContext)
  console.log(user)
  return (

    <>
   
    {user ? <> <Header/> <Outlet/>  </>: <Navigate to="/login" />}
    
    
    </>
  );
}

export default ProtectedRoutes;
