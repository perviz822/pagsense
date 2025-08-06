export default function FlashcardCarousel({
  cards,
  currentIndex,
  setCurrentIndex,
  onMove,
  position,
}) {
  const goLeft = () => {
    setCurrentIndex((prev) => (prev === 0 ? cards.length - 1 : prev - 1));
  };

  const goRight = () => {
    setCurrentIndex((prev) => (prev === cards.length - 1 ? 0 : prev + 1));
  };

  const currentCard = cards[currentIndex];

  if (!currentCard)
    return <p className="text-center text-gray-500">No cards available</p>;

  return (
    <div className="relative  w-full max-w-md mx-auto border border-gray-300 rounded-xl p-6 text-center min-h-[220px] flex flex-col justify-between">
      <div className="absolute top-4 right-4">
        <button
          onClick={() => onMove(currentIndex)}
          className="p-2 bg-blue-500 text-white rounded-1/4 hover:bg-blue-600"
        >
          <span className="block md:hidden">
            {position === "left" ? "↓" : "↑"}
          </span>
          <span className="hidden md:block">
            {position === "left" ? "→" : "←"}
          </span>
        </button>
      </div>

      <div>
        <h2 className="text-2xl font-semibold mb-2">{currentCard.word}</h2>
        <p className="text-gray-600 mb-4">{currentCard.definition}</p>
      </div>

      <div className="space-y-4 ">
        {cards.length > 1 && (
          <div className="flex justify-between">
            <button
              onClick={goLeft}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
            >
              ←
            </button>
            <button
              onClick={goRight}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
            >
              →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
