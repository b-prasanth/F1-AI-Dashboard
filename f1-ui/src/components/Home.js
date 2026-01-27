import React, { useState } from 'react';
import ChatWindow from './ChatWindow';

const Home = ({ setSelected }) => {
  const [chatOpen, setChatOpen] = useState(false);
  const [initialMessage, setInitialMessage] = useState('');
  const [message, setMessage] = useState('');

  const handleSendMessage = () => {
    if (message.trim() === '') return;
    setInitialMessage(message);
    setChatOpen(true);
    setMessage(''); // Clear the input after sending
  };

  return (
    <div className="max-w-6xl mx-auto mt-8 p-8 bg-white rounded-xl shadow-xl relative min-h-screen">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-4 text-white-800">Formula 1 Dashboard</h1>
        <p className="text-xl text-white-600 mb-8">Your ultimate destination for F1 data, statistics, and predictions</p>
        <div className="w-24 h-1 bg-red-500 mx-auto rounded"></div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        <div
          className="p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border-l-4 border-red-500 cursor-pointer transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl"
          onClick={() => setSelected('F1')}
        >
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center mr-4">
              <span className="text-white text-xl font-bold">🏁</span>
            </div>
            <h3 className="text-2xl font-bold text-white-800">Current F1 Season</h3>
          </div>
          <p className="text-white-700 leading-relaxed">Stay updated with the thrill of high-speed racing! Get the latest F1 standings, live statistics, Grand Prix predictions, qualifying results, and race outcomes.</p>
          <div className="mt-4 text-red-600 font-semibold">Click to explore →</div>
        </div>

        <div
          className="p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-xl border-l-4 border-green-500 cursor-pointer transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl"
          onClick={() => setSelected('F1 Data Archives')}
        >
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mr-4">
              <span className="text-white text-xl font-bold">📊</span>
            </div>
            <h3 className="text-2xl font-bold text-white-800">F1 Historical Archives</h3>
          </div>
          <p className="text-white-700 leading-relaxed">Dive into F1 history! Explore comprehensive archives of past qualifying sessions, race results, championship standings, and statistical analysis from previous seasons.</p>
          <div className="mt-4 text-green-600 font-semibold">Click to explore →</div>
        </div>
      </div>

      <div className="mt-12 text-center">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <div
            className="p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border-l-4 border-red-500 cursor-pointer transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl"
            onClick={() => {
              setSelected('F1');
              setTimeout(() => {
                window.f1ChampionshipsTab();
              }, 100);
            }}
          >
            <div className="text-3xl mb-2">🏆</div>
            <h4 className="font-semibold text-white-800">Championships</h4>
            <p className="text-sm text-white-600">Driver & Constructor standings</p>
          </div>
          <div
            className="p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border-l-4 border-red-500 cursor-pointer transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl"
            onClick={() => {
              setSelected('F1');
              setTimeout(() => {
                window.f1StatsTab();
              }, 100);
            }}
          >
            <div className="text-3xl mb-2">📈</div>
            <h4 className="font-semibold text-white-800">Analytics</h4>
            <p className="text-sm text-white-600">Season Statistics Graphs</p>
          </div>
          <div
            className="p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border-l-4 border-red-500 cursor-pointer transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl"
            onClick={() => {
              setSelected('F1');
              setTimeout(() => {
                window.f1PredictionsTab();
              }, 100);
            }}
          >
            <div className="text-3xl mb-2">🔮</div>
            <h4 className="font-semibold text-white-800">Predictions</h4>
            <p className="text-sm text-white-600">AI-powered race forecasts</p>
          </div>
        </div>
      </div>



      {chatOpen && (
        <ChatWindow initialMessage={initialMessage} onClose={() => setChatOpen(false)} />
      )}
    </div>
  );
};

export default Home;
