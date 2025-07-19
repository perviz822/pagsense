import { createContext ,useState} from "react";


export const AuthContext = createContext()

 export function AuthProvider({children}){
    const [user,setUser] = useState(()=> {
        return localStorage.getItem("user") === "true"
    });

    return  <AuthContext.Provider value = {{user,setUser}}>
               {children}
    </AuthContext.Provider>
}

