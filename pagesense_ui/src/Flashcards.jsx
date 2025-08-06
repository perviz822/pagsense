import { useState } from "react";
import FlashcardCarousel from "./FlashCardCarpusel";

export default function DualFlashcardCarousels() {
  const initialSimple = [
    { word: "Run", definition: "To move swiftly on foot." },
    { word: "Big", definition: "Of considerable size." },
     { word: "see", definition: "To see something." }
  ];

  const initialComplex = [
    { word: "Ubiquitous", definition: "Present, appearing, or found everywhere." },
    { word: "Serendipity", definition: "The occurrence of events by chance in a happy way." },
    { word: "Enchantress", definition: "The occurrence of events by chance in a happy way." }
  ];

  const [simpleCards, setSimpleCards] = useState(initialSimple);
  const [complexCards, setComplexCards] = useState(initialComplex);
  const [simpleIndex, setSimpleIndex] = useState(0);
  const [complexIndex, setComplexIndex] = useState(0);

  const moveToComplex = (index) => {
    const card = simpleCards[index];
    setSimpleCards((prev) => prev.filter((card, i) => i !== index));
    setComplexCards((prev) => [...prev, card]);
    setSimpleIndex((prev) => Math.max(0, prev - 1));
  };

  const moveToSimple = (index) => {
    const card = complexCards[index];
    setComplexCards((prev) => prev.filter((_, i) => i !== index));
    setSimpleCards((prev) => [...prev, card]);
    setComplexIndex((prev) => Math.max(0, prev - 1));
  };

  return (
    <div className="flex flex-col md:flex-row justify-center items-start gap-8 p-4">
      {/* Simple Carousel */}
      <div className="w-full md:w-1/2">
        <h3 className="text-xl font-bold text-center mb-4">Simple Words</h3>
        <FlashcardCarousel
          cards={simpleCards}
          currentIndex={simpleIndex}
          setCurrentIndex={setSimpleIndex}
          onMove={moveToComplex}
          buttonLabel="Move to Complex"
          position="left"
        />
      </div>

      {/* Complex Carousel */}
      <div className="w-full md:w-1/2 bg-gray">
        <h3 className="text-xl font-bold text-center mb-4">Complex Words</h3>
        <FlashcardCarousel
          cards={complexCards}
          currentIndex={complexIndex}
          setCurrentIndex={setComplexIndex}
          onMove={moveToSimple}
          buttonLabel="Move to Simple"
          position="right"
        />
      </div>
    </div>
  );
}

