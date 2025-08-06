
import { useParams } from "react-router-dom"
import DualFlashcardCarousels from "./Flashcards";


function BookContent(){
    const {id}  = useParams();
    return(
          <>
         <DualFlashcardCarousels />
          </>
    )
   
}



export default BookContent

