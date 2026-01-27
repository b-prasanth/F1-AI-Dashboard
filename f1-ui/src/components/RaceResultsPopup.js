import React from 'react';

const RaceResultsPopup = ({ year, grandPrix, onClose }) => {
  return (
    <div className="fixed top-0 left-0 w-full h-full bg-gray-800 bg-opacity-50 flex justify-center items-center">
      <div className="bg-white p-8 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold mb-4">{grandPrix} Race Results - {year}</h2>
        <p>Race results will be displayed here.</p>
        <button onClick={onClose} className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded mt-4">
          Close
        </button>
      </div>
    </div>
  );
};

export default RaceResultsPopup;
