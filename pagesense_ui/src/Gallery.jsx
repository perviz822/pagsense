
import React from "react";
import Book from "./Book";
import { Navigate } from "react-router-dom";

const books = [
  {
    id: 1,
    title: "The Great Gatsby",
    description: "A classic novel by F. Scott Fitzgerald.",
    image: "src/assets/images/gatsby.png", // Replace with your actual image path
  },
  {
    id: 2,
    title: "1984",
    description: "A dystopian story by George Orwell.",
    image: "src/assets/images/1984.png",
  },
  {
    id: 3,
    title: "To Kill a Mockingbird",
    description: "A powerful novel by Harper Lee.",
    image: "src/assets/images/mockingbird-e1666245397419.webp",
  },
];




const Gallery = () => {
  return (
    <div className="min-h-screen bg-gray-100 py-10 px-4">
      <h1 className="text-3xl font-bold text-center mb-8">Book Gallery</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
        {books.map((book) => (
         <Book data = {book}/>
        ))}
      </div>
    </div>
  );
};

export default Gallery;