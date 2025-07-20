
function Book ({data}){

    return (
           <div
            key={data.id}
            className="bg-white rounded-2xl shadow-md overflow-hidden hover:shadow-xl transition duration-300 cursor-pointer"
          
          >
            <img
              src={data.image}
              alt={data.title}
              className="h-48 w-full object-cover"
            />
            <div className="p-4">
              <h2 className="text-xl font-semibold mb-1">{data.title}</h2>
              <p className="text-gray-600 text-sm">{data.description}</p>
            </div>
          </div>

    )
}



export default Book;