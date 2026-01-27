import React from 'react';

const Navigation = ({ selected, setSelected }) => {
  const options = ["Home", "F1", "F1 Data Archives", "Chat"];
  return (
    <nav className="flex justify-center space-x-4 p-4 bg-black text-red-500 rounded-b-lg shadow-lg">
      {options.map(option => (
        <button
          key={option}
          className={`px-4 py-2 rounded-lg text-lg font-semibold transition-colors ${selected === option ? 'bg-red-600 text-white' : 'hover:bg-red-500 hover:text-white'}`}
          onClick={() => setSelected(option)}
        >
          {option}
        </button>
      ))}
    </nav>
  );
};

export default Navigation;
