import React, { useState } from 'react';
import './App.css';
import Navigation from './components/Navigation';
import Home from './components/Home';
import F1 from './components/F1';
import F1DataArchives from './components/F1DataArchives';
import ChatWindow from './components/ChatWindow';

// Global functions to set F1 tabs
window.f1ChampionshipsTab = () => {
  const tabButtons = document.querySelectorAll('.f1-tab-button');
  tabButtons.forEach(button => {
    if (button.textContent.trim() === 'Championships') {
      button.click();
    }
  });
};

window.f1PredictionsTab = () => {
  const tabButtons = document.querySelectorAll('.f1-tab-button');
  tabButtons.forEach(button => {
    if (button.textContent.trim() === 'Predictions') {
      button.click();
    }
  });
};

// Main App Component
const App = () => {
  const [selected, setSelected] = useState("Home");

  return (
    <div className="font-sans">
      <Navigation selected={selected} setSelected={setSelected} />
      <main className="p-4">
        {selected === "Home" && <Home setSelected={setSelected} />}
        {selected === "F1" && <F1 />}
        {selected === "F1 Data Archives" && <F1DataArchives />}
        {selected === "Chat" && (
          <div className="w-full max-w-4xl mx-auto">
            <ChatWindow initialMessage="" />
          </div>
        )}
      </main>
    </div>
  );
};

export default App;