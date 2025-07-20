
import { useParams } from "react-router-dom"


function BookContent(){
    const {id}  = useParams();
    return(

          <>
    
         <h1>This page will contain the word list for the book for book wih id :{id} </h1>
    
          </>
    )
   
}



export default BookContent

