import Header from "./Header";

function App({ setUser }) {
  return (
    <>
      <Header setUser={setUser} />
      <div className="p-6">
        <h1 className="text-2xl font-bold">Welcome to the App!</h1>
      </div>
    </>
  );
}

export default App;
